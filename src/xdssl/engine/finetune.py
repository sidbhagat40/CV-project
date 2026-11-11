"""Supervised fine-tuning on a (fraction of a) single source domain, followed by multi-domain evaluation."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.labeled_datasets import DetDataset, SegDataset, det_collate, load_records
from ..data.taxonomy import DET_CLASSES, SEG_CLASSES, SEG_IGNORE
from ..metrics.robustness import summarize_domains
from ..models.backbone import build_backbone
from ..models.detection import DetectionModel
from ..models.segmentation import SegmentationModel
from ..utils.config import resolve_path, to_dict
from ..utils.misc import JsonlLogger, autocast, get_device, infinite_loader, load_json, save_json, seed_everything
from .evaluate import evaluate_domains
from .optim import build_optimizer_and_scheduler, param_groups

PRIMARY_METRIC = {"seg": "mIoU", "det": "AP"}


def build_task_model(cfg, vit=None):
    m = cfg.model
    if vit is None:
        vit = build_backbone(m.arch, m.init, m.get("ckpt"), m.get("drop_path", 0.1))
    if cfg.task == "seg":
        return SegmentationModel(vit, len(SEG_CLASSES), m.get("head", "fpn"), m.get("freeze_backbone", False))
    if cfg.task == "det":
        return DetectionModel(vit, len(DET_CLASSES), m.get("freeze_backbone", False), cfg.data.min_size, cfg.data.max_size)
    raise ValueError(f"unknown task {cfg.task}")


def _train_loader(cfg, records):
    d = cfg.data
    if cfg.task == "seg":
        ds = SegDataset(records, True, d.crop_size, d.base_short, d.scale_range)
        collate = None
    else:
        ds = DetDataset(records, True)
        collate = det_collate
    return DataLoader(ds, batch_size=cfg.optim.batch_size, shuffle=True, num_workers=d.num_workers, drop_last=True,
                      pin_memory=True, persistent_workers=d.num_workers > 0, collate_fn=collate)


def _step_loss(model, batch, task, device, amp):
    with autocast(device, amp):
        if task == "seg":
            x, y = batch
            logits = model(x.to(device, non_blocking=True))
            return F.cross_entropy(logits.float(), y.to(device, non_blocking=True), ignore_index=SEG_IGNORE)
        imgs, targets = batch
        targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
        losses = model([im.to(device, non_blocking=True) for im in imgs], targets)
        return sum(losses.values())


def run_finetune(cfg) -> dict:
    out_dir = resolve_path(cfg.output_dir)
    results_path = out_dir / "results.json"
    if results_path.exists() and not cfg.get("force", False):
        print(f"[finetune] {results_path} exists, skipping")
        return load_json(results_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    device = get_device(cfg.get("device", "auto"))

    d = cfg.data
    records = load_records(resolve_path(d.train), to_dict(d.train_filter) if d.get("train_filter") else None,
                           d.label_fraction, d.subset_seed, d.get("max_records"))
    if len(records) < cfg.optim.batch_size:
        raise ValueError(f"only {len(records)} training records; check data.train / train_filter")
    print(f"[finetune] task={cfg.task} init={cfg.model.init} train_records={len(records)}")

    model = build_task_model(cfg).to(device)
    o = cfg.optim
    groups = param_groups(model, o.lr, o.weight_decay, backbone=model.backbone, layer_decay=o.layer_decay,
                          head_lr_mult=o.get("head_lr_mult", 1.0))
    opt, sched = build_optimizer_and_scheduler(groups, o.total_steps, o.warmup_steps)
    logger = JsonlLogger(out_dir / "log.jsonl")
    it = infinite_loader(_train_loader(cfg, records))
    model.train()
    for step in range(o.total_steps):
        loss_acc = 0.0
        for _ in range(o.grad_accum):
            loss = _step_loss(model, next(it), cfg.task, device, o.amp)
            (loss / o.grad_accum).backward()
            loss_acc += loss.item() / o.grad_accum
        if o.get("clip_grad"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), o.clip_grad)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if not math.isfinite(loss_acc):
            raise FloatingPointError(f"loss is {loss_acc} at step {step}")
        if step % cfg.log_every == 0 or step == o.total_steps - 1:
            logger.log({"step": step, "loss": loss_acc, "lr": opt.param_groups[-1]["lr"]})

    torch.save({"model": model.state_dict(), "config": to_dict(cfg), "encoder_config": model.backbone.config},
               out_dir / "model.pt")
    domain_results = evaluate_domains(model, cfg.task, cfg.eval, device, o.amp)
    results = {
        "meta": {**to_dict(cfg.get("meta", {})), "task": cfg.task, "init": cfg.model.init, "arch": cfg.model.arch,
                 "head": cfg.model.get("head"), "freeze_backbone": cfg.model.get("freeze_backbone", False),
                 "label_fraction": d.label_fraction, "seed": cfg.seed, "n_train_records": len(records),
                 "ssl_ckpt": cfg.model.get("ckpt")},
        "domains": domain_results,
        "summary": summarize_domains(domain_results, PRIMARY_METRIC[cfg.task], cfg.eval.in_domain),
    }
    save_json(results, results_path)
    print(f"[finetune] saved {results_path}")
    return results


def load_finetuned(model_path: str | Path, device="cpu"):
    """Rebuild a fine-tuned model from model.pt (used by re-evaluation and demo scripts)."""
    from omegaconf import OmegaConf

    from ..models.vit import VisionTransformer

    state = torch.load(resolve_path(model_path), map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(state["config"])
    model = build_task_model(cfg, vit=VisionTransformer(**state["encoder_config"]))
    model.load_state_dict(state["model"])
    return model.to(device).eval(), cfg

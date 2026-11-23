"""Self-supervised pretraining on a pool. Training length is fixed in optimizer *steps* (not epochs) so that
every pool gets the same compute regardless of its size."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data.pretrain_dataset import FramePoolDataset, collate_views, contrastive_transforms, mae_transform
from ..models.mae import MAE
from ..models.moco import MoCoV3
from ..models.vit import create_vit, load_timm_weights
from ..utils.config import resolve_path, to_dict
from ..utils.misc import JsonlLogger, autocast, get_device, infinite_loader, seed_everything
from .optim import build_optimizer_and_scheduler, param_groups


def build_pretrain_model(cfg):
    size = tuple(cfg.data.input_size)
    encoder = create_vit(cfg.model.arch, img_size=size, patch_size=cfg.model.get("patch_size", 16))
    if cfg.model.get("init_from") == "imagenet":
        load_timm_weights(encoder, cfg.model.arch)
    if cfg.method == "mae":
        model = MAE(encoder, **to_dict(cfg.mae))
        dataset_kwargs = dict(transform=mae_transform(size, cfg.data.crop_scale))
    elif cfg.method == "moco":
        m = to_dict(cfg.moco)
        pair_mode = m.pop("pair_mode", "same")
        max_dt = m.pop("max_dt_ms", 2000)
        m.pop("momentum", None)
        model = MoCoV3(encoder, **m)
        t1, t2 = contrastive_transforms(size, cfg.data.crop_scale)
        dataset_kwargs = dict(transform=t1, transform2=t2, pair_mode=pair_mode, max_dt_ms=max_dt)
    else:
        raise ValueError(f"unknown method {cfg.method}")
    return model, dataset_kwargs


def run_pretrain(cfg) -> Path:
    out_dir = resolve_path(cfg.output_dir)
    final_path = out_dir / "final.pt"
    if final_path.exists() and not cfg.get("force", False):
        print(f"[pretrain] {final_path} exists, skipping")
        return final_path
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    device = get_device(cfg.get("device", "auto"))

    pool = pd.read_parquet(resolve_path(cfg.data.pool))
    model, ds_kwargs = build_pretrain_model(cfg)
    model.to(device)
    ds = FramePoolDataset(pool, resolve_path(cfg.paths.frames_root), **ds_kwargs)
    o = cfg.optim
    loader = DataLoader(ds, batch_size=o.batch_size, shuffle=True, num_workers=cfg.data.num_workers, drop_last=True,
                        pin_memory=device.type == "cuda", persistent_workers=cfg.data.num_workers > 0,
                        collate_fn=collate_views)
    eff_bs = o.batch_size * o.grad_accum
    lr = o.blr * eff_bs / 256
    opt, sched = build_optimizer_and_scheduler(param_groups(model, lr, o.weight_decay), o.total_steps, o.warmup_steps,
                                               betas=o.betas)
    start = 0
    last_path = out_dir / "last.pt"
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start = state["step"]
        print(f"[pretrain] resumed from step {start}")

    logger = JsonlLogger(out_dir / "log.jsonl", print_every=1)
    it = infinite_loader(loader)
    model.train()
    for step in range(start, o.total_steps):
        loss_acc = 0.0
        for _ in range(o.grad_accum):
            batch = next(it)
            batch = batch.to(device, non_blocking=True) if torch.is_tensor(batch) else [b.to(device, non_blocking=True) for b in batch]
            with autocast(device, o.amp):
                losses, _ = model(batch)
            (losses["loss"] / o.grad_accum).backward()
            loss_acc += losses["loss"].item() / o.grad_accum
        if o.get("clip_grad"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), o.clip_grad)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if cfg.method == "moco":
            m0 = cfg.moco.get("momentum", 0.99)
            model.update_momentum(1 - (1 - m0) * (math.cos(math.pi * step / o.total_steps) + 1) / 2)
        if not math.isfinite(loss_acc):
            raise FloatingPointError(f"loss is {loss_acc} at step {step}")
        if step % cfg.log_every == 0 or step == o.total_steps - 1:
            logger.log({"step": step, "loss": loss_acc, "lr": opt.param_groups[0]["lr"],
                        "epoch": round((step + 1) * eff_bs / len(ds), 3)})
        if (step + 1) % cfg.ckpt_every == 0 and step + 1 < o.total_steps:
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                        "step": step + 1}, last_path)

    final = {"encoder": model.encoder.state_dict(), "encoder_config": model.encoder.config,
             "method": cfg.method, "pool": cfg.get("pool_name"), "config": to_dict(cfg)}
    if cfg.method == "mae":
        final["decoder"] = {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}
    torch.save(final, final_path)
    if last_path.exists():
        last_path.unlink()
    print(f"[pretrain] saved {final_path}")
    return final_path

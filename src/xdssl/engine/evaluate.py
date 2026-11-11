"""Evaluation of a fine-tuned model on a list of target domains (configs/eval/*.yaml)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.labeled_datasets import DetDataset, SegDataset, det_collate, load_records, seg_eval_collate
from ..data.taxonomy import DET_CLASSES, SEG_CLASSES, SEG_IGNORE
from ..metrics.detection import coco_eval
from ..metrics.segmentation import bootstrap_miou_ci, confusion_matrix, iou_from_confusion, miou
from ..utils.config import resolve_path, to_dict
from ..utils.misc import autocast


@torch.no_grad()
def evaluate_seg(model, records, device, amp=None, eval_short=512, num_workers=2, bootstrap=1000) -> dict:
    model.eval()
    loader = DataLoader(SegDataset(records, train=False, eval_short=eval_short), batch_size=1,
                        num_workers=num_workers, collate_fn=seg_eval_collate)
    K = len(SEG_CLASSES)
    per_image = []
    for img, mask in tqdm(loader, desc="eval seg", leave=False):
        x = img.unsqueeze(0).to(device)
        H, W = x.shape[-2:]
        x = F.pad(x, (0, (-W) % 16, 0, (-H) % 16))
        with autocast(device, amp):
            logits = model(x)[..., :H, :W].float()
        logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        per_image.append(confusion_matrix(logits.argmax(1)[0].cpu(), mask, K, SEG_IGNORE))
    cm = sum(per_image)
    ious = iou_from_confusion(cm)
    return {"mIoU": miou(cm), "mIoU_ci95": bootstrap_miou_ci(per_image, bootstrap) if bootstrap else None,
            "per_class_IoU": {c: (None if v != v else float(v)) for c, v in zip(SEG_CLASSES, ious)},
            "n_images": len(per_image)}


@torch.no_grad()
def evaluate_det(model, records, device, amp=None, batch_size=4, num_workers=2) -> dict:
    model.eval()
    loader = DataLoader(DetDataset(records, train=False), batch_size=batch_size, num_workers=num_workers,
                        collate_fn=det_collate)
    gts, preds = [], []
    for imgs, targets in tqdm(loader, desc="eval det", leave=False):
        with autocast(device, amp):
            outs = model([im.to(device) for im in imgs])
        for o, t in zip(outs, targets):
            preds.append({k: o[k].float().cpu().numpy() for k in ("boxes", "labels", "scores")})
            gts.append({"boxes": t["boxes"].numpy(), "labels": t["labels"].numpy()})
    return coco_eval(gts, preds, DET_CLASSES)


def evaluate_domains(model, task: str, eval_cfg, device, amp=None) -> dict:
    """Missing annotation files are skipped with a warning so partial dataset setups still run."""
    results = {}
    for d in eval_cfg.domains:
        d = to_dict(d)
        path = resolve_path(d["file"])
        if not path.exists():
            print(f"[eval] skipping domain '{d['name']}': {path} not found")
            continue
        recs = load_records(path, d.get("filter"), max_records=d.get("max_images") or eval_cfg.get("max_images"))
        if not recs:
            print(f"[eval] skipping domain '{d['name']}': no records after filtering")
            continue
        print(f"[eval] {d['name']}: {len(recs)} images")
        if task == "seg":
            results[d["name"]] = evaluate_seg(model, recs, device, amp, eval_cfg.get("eval_short", 512),
                                              eval_cfg.get("num_workers", 2), eval_cfg.get("bootstrap", 1000))
        else:
            results[d["name"]] = evaluate_det(model, recs, device, amp, eval_cfg.get("batch_size", 4),
                                              eval_cfg.get("num_workers", 2))
    return results

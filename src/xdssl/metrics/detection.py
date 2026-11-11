"""COCO-style box AP via pycocotools, computed from in-memory predictions."""

from __future__ import annotations

import contextlib
import io

import numpy as np


def coco_eval(gts: list[dict], preds: list[dict], class_names: list[str]) -> dict:
    """gts[i] = {"boxes": (n,4) xyxy, "labels": (n,) 1-based}; preds[i] = same + "scores"."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images, anns, dets = [], [], []
    for img_id, (g, p) in enumerate(zip(gts, preds)):
        images.append({"id": img_id})
        for b, l in zip(np.asarray(g["boxes"]).reshape(-1, 4), np.asarray(g["labels"])):
            w, h = float(b[2] - b[0]), float(b[3] - b[1])
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": int(l), "bbox": [float(b[0]), float(b[1]), w, h],
                         "area": w * h, "iscrowd": 0})
        for b, l, s in zip(np.asarray(p["boxes"]).reshape(-1, 4), np.asarray(p["labels"]), np.asarray(p["scores"])):
            dets.append({"image_id": img_id, "category_id": int(l), "score": float(s),
                         "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(class_names)]
    if not anns:
        return {"AP": float("nan"), "AP50": float("nan"), "n_images": len(images), "n_gt": 0}

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"images": images, "annotations": anns, "categories": cats}
        gt.createIndex()
        dt = gt.loadRes(dets) if dets else None
        if dt is None:
            return {"AP": 0.0, "AP50": 0.0, "n_images": len(images), "n_gt": len(anns)}
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    out = {"AP": float(ev.stats[0]), "AP50": float(ev.stats[1]), "AP75": float(ev.stats[2]),
           "AP_small": float(ev.stats[3]), "AP_medium": float(ev.stats[4]), "AP_large": float(ev.stats[5]),
           "n_images": len(images), "n_gt": len(anns)}
    precision = ev.eval["precision"]  # [T, R, K, A, M]
    for k, name in enumerate(class_names):
        p = precision[:, :, k, 0, -1]
        out[f"AP_{name}"] = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
    return out

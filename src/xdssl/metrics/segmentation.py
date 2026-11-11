from __future__ import annotations

import numpy as np
import torch


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255) -> np.ndarray:
    valid = target != ignore_index
    idx = target[valid].long() * num_classes + pred[valid].long()
    return torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes).cpu().numpy()


def iou_from_confusion(cm: np.ndarray) -> np.ndarray:
    """Per-class IoU; NaN for classes absent from both GT and prediction (excluded from the mean)."""
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(0) + cm.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, tp / denom, np.nan)


def miou(cm: np.ndarray) -> float:
    return float(np.nanmean(iou_from_confusion(cm)))


def bootstrap_miou_ci(per_image_cms: list[np.ndarray], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap over images (resample images, re-sum confusion matrices)."""
    if len(per_image_cms) < 2:
        return None
    cms = np.stack(per_image_cms)
    rng = np.random.default_rng(seed)
    stats = [miou(cms[rng.integers(0, len(cms), len(cms))].sum(0)) for _ in range(n_boot)]
    return [float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2)))]

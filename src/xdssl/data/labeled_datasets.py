"""Torch datasets over the unified JSONL records (see labeled_sources.py)."""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..utils.misc import read_jsonl
from .pretrain_dataset import IMAGENET_MEAN, IMAGENET_STD
from .taxonomy import LABEL_LUTS, SEG_IGNORE

_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def match_attrs(rec: dict, include: dict | None) -> bool:
    attrs = rec.get("attrs", {})
    for k, v in (include or {}).items():
        vals = v if isinstance(v, (list, tuple)) else [v]
        if attrs.get(k) not in vals:
            return False
    return True


def load_records(path, include: dict | None = None, fraction: float = 1.0, subset_seed: int = 0,
                 max_records: int | None = None) -> list[dict]:
    """Filter by attrs, then take a deterministic `fraction` subset. The subset depends only on
    (file, filter, fraction, subset_seed) - never on the model - so all inits fine-tune on identical labels."""
    recs = [r for r in read_jsonl(path) if match_attrs(r, include)]
    recs.sort(key=lambda r: r["image"])
    if fraction < 1.0:
        rng = np.random.default_rng(subset_seed)
        n = max(1, int(round(len(recs) * fraction)))
        recs = [recs[i] for i in sorted(rng.permutation(len(recs))[:n])]
    if max_records:
        recs = recs[:max_records]
    return recs


def to_tensor_normalized(img: Image.Image) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
    return (t - _MEAN) / _STD


def load_mask(rec: dict) -> np.ndarray:
    m = np.array(Image.open(rec["mask"]))
    if m.ndim == 3:
        m = m[..., 0]
    lut = LABEL_LUTS[rec.get("label_format", "trainid")]
    return lut[m] if lut is not None else m.astype(np.uint8)


def _resize_short(img: Image.Image, short: int, resample) -> Image.Image:
    w, h = img.size
    s = short / min(w, h)
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), resample)


class SegDataset(Dataset):
    """train=True: random scale (short side = base_short * U(scale_range)), random crop, h-flip.
    train=False: resize the image so its short side is `eval_short`; the mask stays at full resolution and
    predictions are upsampled to it, so mIoU is always computed at native label resolution."""

    def __init__(self, records, train: bool, crop_size=(512, 512), base_short=512, scale_range=(0.5, 2.0),
                 eval_short=512):
        self.recs, self.train = records, train
        self.crop_size, self.base_short, self.scale_range, self.eval_short = tuple(crop_size), base_short, scale_range, eval_short

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        rec = self.recs[i]
        img = Image.open(rec["image"]).convert("RGB")
        mask = load_mask(rec)
        if not self.train:
            return to_tensor_normalized(_resize_short(img, self.eval_short, Image.BILINEAR)), torch.from_numpy(mask.astype(np.int64))
        short = int(self.base_short * random.uniform(*self.scale_range))
        img = _resize_short(img, short, Image.BILINEAR)
        mask = np.array(_resize_short(Image.fromarray(mask), short, Image.NEAREST))
        img, mask = self._random_crop(np.asarray(img), mask)
        if random.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        return to_tensor_normalized(Image.fromarray(np.ascontiguousarray(img))), torch.from_numpy(mask.astype(np.int64).copy())

    def _random_crop(self, img: np.ndarray, mask: np.ndarray):
        ch, cw = self.crop_size
        h, w = mask.shape
        ph, pw = max(0, ch - h), max(0, cw - w)
        if ph or pw:
            img = np.pad(img, ((0, ph), (0, pw), (0, 0)))
            mask = np.pad(mask, ((0, ph), (0, pw)), constant_values=SEG_IGNORE)
            h, w = mask.shape
        y, x = random.randint(0, h - ch), random.randint(0, w - cw)
        return img[y:y + ch, x:x + cw], mask[y:y + ch, x:x + cw]


class DetDataset(Dataset):
    """Returns (image in [0,1], target). Resizing/normalization happens inside torchvision's detector."""

    def __init__(self, records, train: bool):
        self.recs, self.train = records, train

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        rec = self.recs[i]
        img = Image.open(rec["image"]).convert("RGB")
        w, _ = img.size
        x = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
        boxes = torch.tensor(rec["boxes"], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(rec["labels"], dtype=torch.int64) + 1  # 0 is background in torchvision
        keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
        boxes, labels = boxes[keep], labels[keep]
        if self.train and random.random() < 0.5:
            x = x.flip(-1)
            boxes = torch.stack([w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]], dim=1)
        return x, {"boxes": boxes, "labels": labels, "image_id": i}


def det_collate(batch):
    return tuple(zip(*batch))


def seg_eval_collate(batch):
    return batch[0]

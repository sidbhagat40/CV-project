"""Datasets over a pretraining pool (unlabeled frames)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _crop(size, scale):
    h, w = size
    ar = w / h
    return T.RandomResizedCrop((h, w), scale=tuple(scale), ratio=(0.75 * ar, 1.333 * ar),
                               interpolation=T.InterpolationMode.BICUBIC)


def mae_transform(size=(224, 224), scale=(0.2, 1.0)):
    return T.Compose([_crop(size, scale), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class _Solarize:
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        return ImageOps.solarize(img) if random.random() < self.p else img


class _Blur:
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0))) if random.random() < self.p else img


def contrastive_transforms(size=(224, 224), scale=(0.2, 1.0)):
    """The asymmetric MoCo-v3 / BYOL augmentation pair."""
    def make(blur_p, sol_p):
        return T.Compose([
            _crop(size, scale), T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8), T.RandomGrayscale(0.2),
            _Blur(blur_p), _Solarize(sol_p), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return make(1.0, 0.0), make(0.1, 0.2)


class FramePoolDataset(Dataset):
    """Returns one augmented view (MAE) or two views (contrastive).

    pair_mode='temporal' draws the second view from a *different* frame of the same sequence within
    `max_dt_ms` (if the pool contains one), turning natural video motion into a positive pair.
    """

    def __init__(self, pool: pd.DataFrame, frames_root: str | Path, transform=None, transform2=None,
                 pair_mode: str | None = None, max_dt_ms: int = 2000):
        pool = pool.reset_index(drop=True)
        self.paths = pool["path"].tolist()
        self.root = Path(frames_root)
        self.transform, self.transform2 = transform, transform2
        self.pair_mode = pair_mode
        self.neighbors = None
        if pair_mode == "temporal":
            self.neighbors = self._build_neighbors(pool, max_dt_ms)

    @staticmethod
    def _build_neighbors(pool: pd.DataFrame, max_dt_ms: int) -> list[np.ndarray]:
        neighbors = [np.empty(0, dtype=np.int64)] * len(pool)
        for _, g in pool.groupby(["sequence_id", "camera"]):
            idx, ts = g.index.to_numpy(), g["timestamp"].to_numpy()
            for i, t in zip(idx, ts):
                close = idx[(np.abs(ts - t) <= max_dt_ms) & (idx != i)]
                neighbors[i] = close
        return neighbors

    def __len__(self):
        return len(self.paths)

    def _load(self, i):
        return Image.open(self.root / self.paths[i]).convert("RGB")

    def __getitem__(self, i):
        img = self._load(i)
        if self.pair_mode is None:
            return self.transform(img)
        img2 = img
        if self.neighbors is not None and len(self.neighbors[i]):
            img2 = self._load(int(np.random.choice(self.neighbors[i])))
        return self.transform(img), self.transform2(img2)


def collate_views(batch):
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch)
    return tuple(torch.stack(v) for v in zip(*batch))

"""Faster R-CNN (torchvision) on top of the ViT feature pyramid - a simplified ViTDet."""

from __future__ import annotations

from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN

from .backbone import ViTPyramid
from .vit import VisionTransformer


class _DetBackbone(nn.Module):
    def __init__(self, pyramid: ViTPyramid):
        super().__init__()
        self.pyramid = pyramid
        self.out_channels = pyramid.out_channels

    def forward(self, x):
        feats = self.pyramid(x)
        out = OrderedDict((str(i), f) for i, f in enumerate(feats))
        out["pool"] = F.max_pool2d(feats[-1], 1, 2, 0)
        return out


class DetectionModel(nn.Module):
    """Wraps FasterRCNN so train mode returns a loss dict and eval mode returns per-image detections."""

    def __init__(self, vit: VisionTransformer, num_classes: int, freeze_backbone: bool = False, min_size: int = 576,
                 max_size: int = 1024):
        super().__init__()
        self.pyramid = ViTPyramid(vit, 256, freeze_backbone=freeze_backbone)
        # torchvision's defaults (5 anchor sizes, RoIAlign over '0'-'3') match the 5 maps produced here.
        self.detector = FasterRCNN(_DetBackbone(self.pyramid), num_classes=num_classes + 1, min_size=min_size,
                                   max_size=max_size)

    @property
    def backbone(self) -> VisionTransformer:
        return self.pyramid.vit

    def forward(self, images, targets=None):
        return self.detector(images, targets)

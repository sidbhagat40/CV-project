from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ViTPyramid, _norm
from .vit import VisionTransformer


class SemanticFPNHead(nn.Module):
    """Panoptic-FPN style semantic head: top-down FPN fusion, then every level is brought to stride 4 and summed."""

    def __init__(self, channels: int, num_classes: int, strides=(4, 8, 16, 32)):
        super().__init__()
        self.smooth = nn.ModuleList([nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                                   _norm(channels), nn.ReLU(inplace=True)) for _ in strides])
        self.dropout = nn.Dropout2d(0.1)
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def forward(self, feats):
        for i in range(len(feats) - 1, 0, -1):  # top-down
            feats[i - 1] = feats[i - 1] + F.interpolate(feats[i], size=feats[i - 1].shape[-2:], mode="bilinear", align_corners=False)
        size = feats[0].shape[-2:]
        out = sum(F.interpolate(s(f), size=size, mode="bilinear", align_corners=False) if j else s(f)
                  for j, (f, s) in enumerate(zip(feats, self.smooth)))
        return self.classifier(self.dropout(out))


class SegmentationModel(nn.Module):
    """head='fpn': full dense head (use with fine-tuning). head='linear': BN + 1x1 conv on the last ViT layer -
    with freeze_backbone=True this is the standard linear-probe readout of representation quality."""

    def __init__(self, vit: VisionTransformer, num_classes: int, head: str = "fpn", freeze_backbone: bool = False,
                 channels: int = 256):
        super().__init__()
        self.head_type = head
        if head == "fpn":
            self.neck = ViTPyramid(vit, channels, freeze_backbone=freeze_backbone)
            self.head = SemanticFPNHead(channels, num_classes)
        elif head == "linear":
            self.vit, self.freeze = vit, freeze_backbone
            if freeze_backbone:
                vit.requires_grad_(False)
            self.head = nn.Sequential(nn.BatchNorm2d(vit.embed_dim), nn.Conv2d(vit.embed_dim, num_classes, 1))
        else:
            raise ValueError(f"unknown head {head}")

    @property
    def backbone(self) -> VisionTransformer:
        return self.neck.vit if self.head_type == "fpn" else self.vit

    def forward(self, x):
        H, W = x.shape[-2:]
        if self.head_type == "fpn":
            logits = self.head(self.neck(x))
        else:
            ctx = torch.no_grad() if self.freeze else torch.enable_grad()
            if self.freeze:
                self.vit.eval()
            with ctx:
                f = self.vit.forward_features(x, [self.vit.depth - 1])[0]
            logits = self.head(f)
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

"""Builds the ViT encoder for fine-tuning from one of three inits, plus a ViTDet-style feature pyramid."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.config import resolve_path
from .vit import VisionTransformer, create_vit, load_timm_weights

INITS = ("random", "imagenet", "ssl")


def build_backbone(arch: str, init: str, ckpt: str | None = None, drop_path_rate: float = 0.1) -> VisionTransformer:
    if init not in INITS:
        raise ValueError(f"init must be one of {INITS}, got {init}")
    if init == "ssl":
        if not ckpt:
            raise ValueError("init=ssl requires model.ckpt (a pretraining checkpoint)")
        state = torch.load(resolve_path(ckpt), map_location="cpu", weights_only=False)
        vit = VisionTransformer(**state["encoder_config"], drop_path_rate=drop_path_rate)
        vit.load_state_dict(state["encoder"])
    else:
        vit = create_vit(arch, img_size=(224, 224), drop_path_rate=drop_path_rate)
        if init == "imagenet":
            load_timm_weights(vit, arch)
    vit.pos_embed.requires_grad_(True)
    return vit


def _norm(c):
    return nn.GroupNorm(32 if c % 32 == 0 else 1, c)


class ViTPyramid(nn.Module):
    """Turns 4 intermediate ViT blocks (all stride 16) into maps at strides 4/8/16/32 with `out_channels`."""

    strides = (4, 8, 16, 32)

    def __init__(self, vit: VisionTransformer, out_channels: int = 256, out_indices=None, freeze_backbone=False):
        super().__init__()
        self.vit, self.freeze = vit, freeze_backbone
        d, D = vit.depth, vit.embed_dim
        self.out_indices = list(out_indices) if out_indices else [d // 4 - 1, d // 2 - 1, 3 * d // 4 - 1, d - 1]
        self.rescale = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(D, D, 2, 2), _norm(D), nn.GELU(), nn.ConvTranspose2d(D, D, 2, 2)),
            nn.ConvTranspose2d(D, D, 2, 2),
            nn.Identity(),
            nn.MaxPool2d(2),
        ])
        self.lateral = nn.ModuleList([
            nn.Sequential(nn.Conv2d(D, out_channels, 1, bias=False), _norm(out_channels),
                          nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), _norm(out_channels))
            for _ in range(4)
        ])
        self.out_channels = out_channels
        if freeze_backbone:
            self.vit.requires_grad_(False)

    def forward(self, x) -> list[torch.Tensor]:
        if self.freeze:
            self.vit.eval()
            with torch.no_grad():
                feats = self.vit.forward_features(x, self.out_indices)
        else:
            feats = self.vit.forward_features(x, self.out_indices)
        return [lat(rs(f)) for f, rs, lat in zip(feats, self.rescale, self.lateral)]

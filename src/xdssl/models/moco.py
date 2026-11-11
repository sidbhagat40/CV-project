"""MoCo v3 (Chen et al., 2021) for ViT - the contrastive alternative to MAE. With the `temporal` pair mode
of FramePoolDataset, the two views come from nearby frames of the same drive.

Contrastive learning benefits from large batches; on a single 8 GB GPU expect batch ~64-128 with bf16, so
treat MAE as the primary method and this as an ablation."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import VisionTransformer


def _mlp(in_dim, hidden, out_dim, layers, last_bn=True):
    mods = []
    for i in range(layers):
        d_in = in_dim if i == 0 else hidden
        d_out = out_dim if i == layers - 1 else hidden
        mods.append(nn.Linear(d_in, d_out, bias=False))
        if i < layers - 1:
            mods += [nn.BatchNorm1d(d_out), nn.ReLU(inplace=True)]
        elif last_bn:
            mods.append(nn.BatchNorm1d(d_out, affine=False))
    return nn.Sequential(*mods)


class MoCoV3(nn.Module):
    def __init__(self, encoder: VisionTransformer, dim=256, mlp_dim=4096, temperature=0.2):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
        self.projector = _mlp(encoder.embed_dim, mlp_dim, dim, 3)
        self.predictor = _mlp(dim, mlp_dim, dim, 2, last_bn=False)
        self.momentum_encoder = copy.deepcopy(encoder)
        self.momentum_projector = copy.deepcopy(self.projector)
        for p in list(self.momentum_encoder.parameters()) + list(self.momentum_projector.parameters()):
            p.requires_grad_(False)
        self.momentum = 0.99

    @torch.no_grad()
    def update_momentum(self, m: float):
        self.momentum = m
        for online, target in [(self.encoder, self.momentum_encoder), (self.projector, self.momentum_projector)]:
            for po, pt in zip(online.parameters(), target.parameters()):
                pt.mul_(m).add_(po.detach(), alpha=1 - m)

    def _contrastive(self, q, k):
        q, k = F.normalize(q.float(), dim=1), F.normalize(k.float(), dim=1)
        logits = q @ k.t() / self.temperature
        labels = torch.arange(q.shape[0], device=q.device)
        return F.cross_entropy(logits, labels) * 2 * self.temperature

    def forward(self, views):
        x1, x2 = views
        q1 = self.predictor(self.projector(self.encoder(x1)))
        q2 = self.predictor(self.projector(self.encoder(x2)))
        with torch.no_grad():
            k1 = self.momentum_projector(self.momentum_encoder(x1))
            k2 = self.momentum_projector(self.momentum_encoder(x2))
        loss = self._contrastive(q1, k2) + self._contrastive(q2, k1)
        return {"loss": loss}, {}

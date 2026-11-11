"""Plain ViT encoder shared by every pretraining method and downstream head.

Parameter names mirror timm's VisionTransformer so ImageNet checkpoints load directly. Position embeddings are
stored for the construction-time grid and bicubically resampled on the fly, so the same weights run at the
pretraining crop size (e.g. 224x224) and at wide fine-tuning resolutions (e.g. 512x910).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARCHS = {
    "vit_tiny": dict(embed_dim=192, depth=12, num_heads=3),
    "vit_small": dict(embed_dim=384, depth=12, num_heads=6),
    "vit_base": dict(embed_dim=768, depth=12, num_heads=12),
}
TIMM_NAMES = {
    "vit_tiny": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "vit_small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "vit_base": "vit_base_patch16_224.augreg_in21k_ft_in1k",
}


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = torch.empty((x.shape[0],) + (1,) * (x.ndim - 1), device=x.device, dtype=x.dtype).bernoulli_(1 - self.p)
        return x * keep / (1 - self.p)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, patch_size, patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2), x.shape[-2:]


def sincos_pos_embed(dim: int, grid_hw: tuple[int, int], cls_token: bool = True) -> torch.Tensor:
    """2D sin-cos position embedding (MAE), shape (1, [1+]H*W, dim)."""
    gh, gw = grid_hw
    gy, gx = np.meshgrid(np.arange(gh, dtype=np.float64), np.arange(gw, dtype=np.float64), indexing="ij")

    def emb_1d(d, pos):
        omega = 1.0 / 10000 ** (np.arange(d // 2, dtype=np.float64) / (d / 2.0))
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    emb = np.concatenate([emb_1d(dim // 2, gy), emb_1d(dim // 2, gx)], axis=1)
    if cls_token:
        emb = np.concatenate([np.zeros((1, dim)), emb], axis=0)
    return torch.from_numpy(emb).float().unsqueeze(0)


def resample_pos_embed(pos: torch.Tensor, src_hw, dst_hw) -> torch.Tensor:
    """pos: (1, H*W, D) patch-only embeddings."""
    if tuple(src_hw) == tuple(dst_hw):
        return pos
    D = pos.shape[-1]
    grid = pos.reshape(1, src_hw[0], src_hw[1], D).permute(0, 3, 1, 2)
    grid = F.interpolate(grid.float(), size=tuple(dst_hw), mode="bicubic", align_corners=False).to(pos.dtype)
    return grid.permute(0, 2, 3, 1).reshape(1, -1, D)


class VisionTransformer(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 drop_path_rate=0.0):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        self.patch_size, self.embed_dim, self.depth = patch_size, embed_dim, depth
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.config = dict(img_size=list(img_size), patch_size=patch_size, embed_dim=embed_dim, depth=depth,
                           num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.patch_embed = PatchEmbed(patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(sincos_pos_embed(embed_dim, self.grid_size))
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, dpr[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self._init_weights()

    def _init_weights(self):
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def patch_pos(self, grid_hw) -> torch.Tensor:
        return resample_pos_embed(self.pos_embed[:, 1:], self.grid_size, grid_hw)

    def embed(self, x):
        """Patchify + add position embedding (no cls token). Returns tokens (B, N, D) and grid (h, w)."""
        tokens, hw = self.patch_embed(x)
        return tokens + self.patch_pos(hw), hw

    def prepend_cls(self, tokens):
        cls = (self.cls_token + self.pos_embed[:, :1]).expand(tokens.shape[0], -1, -1)
        return torch.cat([cls, tokens], dim=1)

    def forward_features(self, x, out_indices=None):
        """Returns final normed tokens (B, 1+N, D), or a list of (B, D, h, w) maps from `out_indices` blocks."""
        tokens, (h, w) = self.embed(x)
        x = self.prepend_cls(tokens)
        feats = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if out_indices is not None and i in out_indices:
                t = self.norm(x) if i == self.depth - 1 else x
                feats.append(t[:, 1:].transpose(1, 2).reshape(x.shape[0], self.embed_dim, h, w))
        return feats if out_indices is not None else self.norm(x)

    def forward(self, x):
        return self.forward_features(x)[:, 0]

    def layer_id(self, name: str) -> int:
        """Depth index used for layer-wise LR decay."""
        if name.startswith(("cls_token", "pos_embed", "patch_embed")):
            return 0
        if name.startswith("blocks."):
            return int(name.split(".")[1]) + 1
        return self.depth + 1


def create_vit(arch: str, img_size=(224, 224), patch_size=16, drop_path_rate=0.0) -> VisionTransformer:
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch}; choose from {list(ARCHS)}")
    return VisionTransformer(img_size=img_size, patch_size=patch_size, drop_path_rate=drop_path_rate, **ARCHS[arch])


def load_timm_weights(vit: VisionTransformer, arch: str) -> None:
    import timm

    sd = timm.create_model(TIMM_NAMES[arch], pretrained=True).state_dict()
    if sd["pos_embed"].shape != vit.pos_embed.shape:
        n = sd["pos_embed"].shape[1] - 1
        side = int(math.isqrt(n))
        patch = resample_pos_embed(sd["pos_embed"][:, 1:], (side, side), vit.grid_size)
        sd["pos_embed"] = torch.cat([sd["pos_embed"][:, :1], patch], dim=1)
    missing, unexpected = vit.load_state_dict({k: v for k, v in sd.items() if not k.startswith("head")}, strict=False)
    if missing:
        print(f"[timm init] missing keys: {missing}")

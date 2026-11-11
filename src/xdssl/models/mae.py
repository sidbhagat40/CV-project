"""Masked Autoencoder (He et al., 2022). Memory-friendly SSL: the encoder only sees the visible 25% of
patches, which is what makes ViT pretraining feasible on an 8 GB GPU."""

from __future__ import annotations

import torch
import torch.nn as nn

from .vit import Block, VisionTransformer, sincos_pos_embed


class MAE(nn.Module):
    def __init__(self, encoder: VisionTransformer, decoder_dim=256, decoder_depth=4, decoder_heads=8,
                 mask_ratio=0.75, norm_pix_loss=True):
        super().__init__()
        self.encoder = encoder
        self.encoder.pos_embed.requires_grad_(False)  # fixed sin-cos during pretraining, as in MAE
        self.mask_ratio, self.norm_pix_loss = mask_ratio, norm_pix_loss
        p, D = encoder.patch_size, encoder.embed_dim
        self.decoder_embed = nn.Linear(D, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.register_buffer("decoder_pos_embed", sincos_pos_embed(decoder_dim, encoder.grid_size), persistent=False)
        self.decoder_blocks = nn.ModuleList([Block(decoder_dim, decoder_heads) for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_dim, eps=1e-6)
        self.decoder_pred = nn.Linear(decoder_dim, p * p * 3)
        nn.init.normal_(self.mask_token, std=0.02)
        for m in [self.decoder_embed, self.decoder_pred]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def patchify(self, imgs):
        p = self.encoder.patch_size
        B, C, H, W = imgs.shape
        x = imgs.reshape(B, C, H // p, p, W // p, p)
        return torch.einsum("nchpwq->nhwpqc", x).reshape(B, (H // p) * (W // p), p * p * C)

    @staticmethod
    def random_masking(x, ratio):
        B, N, D = x.shape
        keep = int(N * (1 - ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :keep]
        x_vis = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :keep] = 0
        return x_vis, torch.gather(mask, 1, ids_restore), ids_restore

    def forward_encoder(self, imgs):
        tokens, hw = self.encoder.embed(imgs)
        if tuple(hw) != tuple(self.encoder.grid_size):
            raise ValueError(f"MAE input grid {tuple(hw)} != encoder grid {self.encoder.grid_size}")
        x, mask, ids_restore = self.random_masking(tokens, self.mask_ratio)
        x = self.encoder.prepend_cls(x)
        for blk in self.encoder.blocks:
            x = blk(x)
        return self.encoder.norm(x), mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        B, N = ids_restore.shape
        mask_tokens = self.mask_token.expand(B, N + 1 - x.shape[1], -1)
        x_ = torch.cat([x[:, 1:], mask_tokens], dim=1)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1], x_], dim=1)
        pos = torch.cat([torch.zeros_like(self.decoder_pos_embed[:, :1]), self.decoder_pos_embed[:, 1:]], dim=1)
        x = x + pos
        for blk in self.decoder_blocks:
            x = blk(x)
        return self.decoder_pred(self.decoder_norm(x))[:, 1:]

    def forward(self, imgs):
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean, var = target.mean(-1, keepdim=True), target.var(-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        loss = ((pred.float() - target) ** 2).mean(-1)
        loss = (loss * mask).sum() / mask.sum()
        return {"loss": loss}, {"pred": pred, "mask": mask}

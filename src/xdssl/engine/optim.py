from __future__ import annotations

import torch

from ..models.vit import VisionTransformer
from ..utils.misc import warmup_cosine


def _no_weight_decay(name: str, p: torch.nn.Parameter) -> bool:
    return p.ndim <= 1 or name.endswith(("pos_embed", "cls_token", "mask_token"))


def param_groups(model: torch.nn.Module, lr: float, weight_decay: float, backbone: VisionTransformer | None = None,
                 layer_decay: float = 1.0, head_lr_mult: float = 1.0) -> list[dict]:
    """AdamW param groups with optional layer-wise LR decay over `backbone` (standard for fine-tuning MAE ViTs:
    block i gets lr * layer_decay^(depth+1-i); non-backbone params get lr * head_lr_mult)."""
    backbone_names = {id(p): n for n, p in backbone.named_parameters()} if backbone is not None else {}
    n_layers = backbone.depth + 1 if backbone is not None else 0
    groups: dict[tuple, dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in backbone_names:
            scale = layer_decay ** (n_layers - backbone.layer_id(backbone_names[id(p)]))
        else:
            scale = head_lr_mult
        no_wd = _no_weight_decay(name, p)
        key = (scale, no_wd)
        if key not in groups:
            groups[key] = {"params": [], "lr": lr * scale, "weight_decay": 0.0 if no_wd else weight_decay}
        groups[key]["params"].append(p)
    return list(groups.values())


def build_optimizer_and_scheduler(groups, total_steps: int, warmup_steps: int, betas=(0.9, 0.999), min_lr_ratio=0.0):
    opt = torch.optim.AdamW(groups, betas=tuple(betas))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, total_steps, warmup_steps, min_lr_ratio))
    return opt, sched

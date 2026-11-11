"""Input / masked / reconstruction grid from an MAE `final.pt` (pretraining sanity check and report figure).

  python scripts/visualize_mae.py --ckpt runs/pretrain/mae-bdd_mixed/final.pt --pool pools/bdd_mixed/pool.parquet \
      --n 6 --out outputs/mae_recon.png
"""

import argparse

import numpy as np
import pandas as pd
import torch

from omegaconf import OmegaConf

from xdssl.data.pretrain_dataset import IMAGENET_MEAN, IMAGENET_STD, FramePoolDataset, mae_transform
from xdssl.engine.pretrain import build_pretrain_model
from xdssl.utils.config import load_paths, resolve_path
from xdssl.utils.misc import get_device


def load_mae(ckpt_path, device):
    """Rebuilds the MAE (encoder + decoder) from a checkpoint written by `run_pretrain`."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ck.get("method") != "mae" or "decoder" not in ck:
        raise ValueError(f"{ckpt_path} has no MAE decoder (method={ck.get('method')}); re-run pretraining")
    cfg = OmegaConf.create(ck["config"])
    model, _ = build_pretrain_model(cfg)
    model.encoder.load_state_dict(ck["encoder"])
    model.load_state_dict(ck["decoder"], strict=False)
    return model.to(device).eval(), cfg


def to_uint8(x: torch.Tensor) -> np.ndarray:
    mean, std = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x.cpu() * std + mean).clamp(0, 1).permute(0, 2, 3, 1).mul(255).byte().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pool", required=True, help="pool.parquet to draw frames from")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/mae_recon.png")
    ap.add_argument("--paths", default=None)
    a = ap.parse_args()
    paths = load_paths(a.paths)
    device = get_device()
    model, cfg = load_mae(resolve_path(a.ckpt), device)
    pool = pd.read_parquet(resolve_path(a.pool))
    pool = pool.sample(min(a.n, len(pool)), random_state=a.seed)
    ds = FramePoolDataset(pool, resolve_path(paths.frames_root), transform=mae_transform(tuple(cfg.data.input_size), (1.0, 1.0)))
    torch.manual_seed(a.seed)
    imgs = torch.stack([ds[i] for i in range(len(ds))]).to(device)
    masked, _, paste = model.reconstruct(imgs)
    cols = [to_uint8(imgs), to_uint8(masked), to_uint8(paste)]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(imgs), 3, figsize=(7.5, 2.5 * len(imgs)), squeeze=False)
    for c, title in enumerate(["input", f"masked ({model.mask_ratio:.0%})", "reconstruction"]):
        axes[0, c].set_title(title)
        for r in range(len(imgs)):
            axes[r, c].imshow(cols[c][r]), axes[r, c].axis("off")
    out = resolve_path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

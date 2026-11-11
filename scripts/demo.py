"""Side-by-side qualitative comparison of fine-tuned models (the "fog vs clear" demo figure).

  python scripts/demo.py --models runs/finetune/imagenet__det_bdd_clearday__f1__s0/model.pt \
                                  runs/finetune/mae-bdd_nusc_waymo__det_bdd_clearday__f1__s0/model.pt \
      --names ImageNet "MAE diverse pool" --jsonl D:/xdssl_data/labeled/det/bdd_det_val.jsonl \
      --filter weather=fog --n 4 --out outputs/demo_fog.png
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from xdssl.data.labeled_datasets import _resize_short, load_records, to_tensor_normalized
from xdssl.data.taxonomy import DET_CLASSES, SEG_PALETTE
from xdssl.engine.finetune import load_finetuned
from xdssl.utils.config import resolve_path
from xdssl.utils.misc import get_device


@torch.no_grad()
def predict_seg(model, img: Image.Image, device, short=512):
    x = to_tensor_normalized(_resize_short(img, short, Image.BILINEAR)).unsqueeze(0).to(device)
    H, W = x.shape[-2:]
    logits = model(F.pad(x, (0, (-W) % 16, 0, (-H) % 16)))[..., :H, :W]
    pred = F.interpolate(logits.float(), size=img.size[::-1], mode="bilinear").argmax(1)[0].cpu().numpy()
    overlay = (0.45 * np.asarray(img) + 0.55 * SEG_PALETTE[pred]).astype(np.uint8)
    return Image.fromarray(overlay)


@torch.no_grad()
def predict_det(model, img: Image.Image, device, thresh=0.5):
    import matplotlib.pyplot as plt
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div(255).to(device)
    out = model([x])[0]
    fig = plt.figure(figsize=(img.width / 100, img.height / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img), ax.axis("off")
    colors = {1: "deepskyblue", 2: "orangered"}
    for b, l, s in zip(out["boxes"].cpu(), out["labels"].cpu(), out["scores"].cpu()):
        if s < thresh:
            continue
        x1, y1, x2, y2 = b.tolist()
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=colors.get(int(l), "y"), lw=1.5))
        ax.text(x1, y1, f"{DET_CLASSES[int(l) - 1]} {s:.2f}", color="w", fontsize=6, bbox=dict(fc=colors.get(int(l), "y"), pad=0.5, lw=0))
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return Image.fromarray(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--images", nargs="*", default=None)
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--filter", nargs="*", default=[], help="attrs filters, e.g. weather=fog")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/demo.png")
    a = ap.parse_args()
    device = get_device()
    models = [load_finetuned(m, device) for m in a.models]
    names = a.names or [Path(m).parent.name for m in a.models]
    if a.images:
        imgs = a.images
    else:
        recs = load_records(resolve_path(a.jsonl), dict(f.split("=", 1) for f in a.filter))
        idx = np.random.default_rng(a.seed).permutation(len(recs))[: a.n]
        imgs = [recs[i]["image"] for i in idx]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(imgs), len(models) + 1, figsize=(4 * (len(models) + 1), 2.4 * len(imgs)), squeeze=False)
    for r, path in enumerate(imgs):
        img = Image.open(path).convert("RGB")
        panels = [img] + [predict_seg(m, img, device) if cfg.task == "seg" else predict_det(m, img, device) for m, cfg in models]
        for c, (ax, p) in enumerate(zip(axes[r], panels)):
            ax.imshow(p), ax.axis("off")
            if r == 0:
                ax.set_title((["input"] + names)[c], fontsize=9)
    fig.tight_layout()
    out = resolve_path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

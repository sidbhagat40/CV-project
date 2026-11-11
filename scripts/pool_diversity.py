"""Measure pool diversity (x-axis of the main claim) and pool->target distances.

  python scripts/pool_diversity.py --pools bdd_clearday bdd_mixed bdd_nusc_waymo \
      --targets configs/eval/seg_targets.yaml configs/eval/det_targets.yaml
Writes pools_root/<pool>/diversity.json with: vendi, metadata entropies, and Frechet distance to each target.
The reference encoder is a fixed ImageNet ViT-S (needs timm); use --encoder random for a dependency-free run.
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from xdssl.data.labeled_datasets import load_records
from xdssl.data.pools import pool_dir
from xdssl.data.pretrain_dataset import IMAGENET_MEAN, IMAGENET_STD
from xdssl.metrics.diversity import frechet_distance, vendi_score
from xdssl.models.vit import create_vit, load_timm_weights
from xdssl.utils.config import load_config, load_paths, resolve_path, to_dict
from xdssl.utils.misc import get_device, save_json

TF = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class _Images(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return TF(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def embed(model, paths, device, bs=64, workers=4):
    feats = []
    for x in DataLoader(_Images(paths), batch_size=bs, num_workers=workers):
        feats.append(model(x.to(device)).float().cpu().numpy())
    return np.concatenate(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", required=True)
    ap.add_argument("--targets", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=2000, help="frames sampled per pool")
    ap.add_argument("--n-target", type=int, default=500)
    ap.add_argument("--encoder", choices=["imagenet", "random"], default="imagenet")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--paths", default=None)
    a = ap.parse_args()
    paths = load_paths(a.paths)
    device = get_device()
    model = create_vit("vit_small")
    if a.encoder == "imagenet":
        load_timm_weights(model, "vit_small")
    model.to(device).eval()

    target_feats = {}
    for tcfg in a.targets:
        ecfg = load_config(tcfg, paths_file=a.paths).eval
        for d in ecfg.domains:
            d = to_dict(d)
            if d["name"] in target_feats or not resolve_path(d["file"]).exists():
                continue
            recs = load_records(resolve_path(d["file"]), d.get("filter"))
            if len(recs) < 10:
                continue
            idx = np.random.default_rng(0).permutation(len(recs))[: a.n_target]
            target_feats[d["name"]] = embed(model, [recs[i]["image"] for i in idx], device, workers=a.workers)
            print(f"target {d['name']}: {len(idx)} images")

    frames_root = resolve_path(paths.frames_root)
    for name in a.pools:
        pdir = pool_dir(resolve_path(paths.pools_root), name)
        pool = pd.read_parquet(pdir / "pool.parquet")
        sample = pool.sample(n=min(a.n, len(pool)), random_state=0)
        feats = embed(model, [str(frames_root / p) for p in sample["path"]], device, workers=a.workers)
        summary = json.loads((pdir / "summary.json").read_text()) if (pdir / "summary.json").exists() else {}
        out = {"vendi": vendi_score(feats), "n_embedded": len(feats), "encoder": a.encoder,
               **{f"entropy_{k}": v for k, v in summary.get("entropy", {}).items()},
               "n_sequences": summary.get("n_sequences"),
               "frechet_to_target": {t: frechet_distance(feats, f) for t, f in target_feats.items()}}
        save_json(out, pdir / "diversity.json")
        print(name, json.dumps({k: v for k, v in out.items() if k != "frechet_to_target"}))


if __name__ == "__main__":
    main()

"""Prepare frames, manifest, detection JSONL and the leakage-exclusion list from the Hugging Face BDD100K-val mirror.

  python scripts/prepare_bdd_hf.py --raw ../data/raw_bdd_val
"""

import argparse

from xdssl.data.sources.bdd100k_hf import prepare
from xdssl.utils.config import load_paths, resolve_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--paths", default=None)
    a = ap.parse_args()
    p = load_paths(a.paths)
    print(prepare(resolve_path(a.raw), resolve_path(p.frames_root), resolve_path(p.labeled_root), seed=a.seed,
                  workers=a.workers))


if __name__ == "__main__":
    main()

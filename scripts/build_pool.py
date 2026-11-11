"""Sample a pretraining pool from the frame index.

  python scripts/build_pool.py --spec configs/pools/bdd_nusc_waymo.yaml
Writes pools_root/<name>/pool.parquet and summary.json.
"""

import argparse
import json

from xdssl.data.pools import build_pool, pool_dir, summarize_pool
from xdssl.utils.config import load_config, resolve_path
from xdssl.utils.misc import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--paths", default=None)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    spec = load_config(a.spec, a.overrides, a.paths)
    out = pool_dir(resolve_path(spec.paths.pools_root), spec.name)
    if (out / "pool.parquet").exists() and not a.force:
        print(f"{out / 'pool.parquet'} exists (use --force to rebuild)")
        return
    pool = build_pool({k: v for k, v in spec.items() if k != "paths"}, resolve_path(spec.paths.frames_root))
    out.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(out / "pool.parquet", index=False)
    summary = summarize_pool(pool)
    save_json(summary, out / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

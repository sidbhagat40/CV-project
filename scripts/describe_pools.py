"""Composition bar charts (dataset / weather / time of day) for the pools' `summary.json` files (report figure).

  python scripts/describe_pools.py --pools bdd_clearday bdd_mixed bdd_nusc_waymo --out outputs/pools.png
"""

import argparse

from xdssl.analysis import plot_pool_composition
from xdssl.utils.config import load_paths, resolve_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/pools.png")
    ap.add_argument("--paths", default=None)
    a = ap.parse_args()
    paths = load_paths(a.paths)
    out = resolve_path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_pool_composition(resolve_path(paths.pools_root), a.pools, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

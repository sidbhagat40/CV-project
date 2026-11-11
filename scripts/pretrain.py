"""Self-supervised pretraining.

  python scripts/pretrain.py --config configs/pretrain/mae_vit_small.yaml \
      data.pool=D:/xdssl_data/pools/bdd_mixed/pool.parquet output_dir=runs/pretrain/mae_vit_small_bdd_mixed
"""

import argparse

from xdssl.engine.pretrain import run_pretrain
from xdssl.utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--paths", default=None)
    ap.add_argument("overrides", nargs="*", help="dot-list overrides, e.g. optim.total_steps=1000")
    a = ap.parse_args()
    run_pretrain(load_config(a.config, a.overrides, a.paths))


if __name__ == "__main__":
    main()

"""Fine-tune on the source domain and evaluate on all target domains.

  python scripts/finetune.py --config configs/finetune/seg_bdd.yaml model.init=imagenet output_dir=runs/finetune/seg_in1k
  python scripts/finetune.py --config configs/finetune/det_bdd_clearday.yaml model.init=ssl \
      model.ckpt=runs/pretrain/mae_vit_small_bdd_mixed/final.pt data.label_fraction=0.1
"""

import argparse

from xdssl.engine.finetune import run_finetune
from xdssl.utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--paths", default=None)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    r = run_finetune(load_config(a.config, a.overrides, a.paths))
    print(r.get("summary"))


if __name__ == "__main__":
    main()

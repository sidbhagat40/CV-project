"""Generate a tiny synthetic dataset tree for trying the pipeline end-to-end.

  python scripts/make_synthetic_data.py --out data_synth
  python scripts/run_experiment.py --exp configs/smoke/experiment.yaml --paths data_synth/paths.yaml
  python scripts/analyze.py --exp configs/smoke/experiment.yaml --paths data_synth/paths.yaml
"""

import argparse

from xdssl.data.synthetic import make_synthetic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_synth")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(f"paths file: {make_synthetic(a.out, a.seed)}")


if __name__ == "__main__":
    main()

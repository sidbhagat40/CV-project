"""Aggregate results into CSV tables + figures, one set per fine-tuning protocol.

  python scripts/analyze.py --exp configs/experiments/e1_diversity.yaml
  python scripts/analyze.py            # everything under runs/finetune
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from xdssl import analysis as A
from xdssl.experiments import run_names
from xdssl.utils.config import load_paths, resolve_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--paths", default=None)
    a = ap.parse_args()
    paths = load_paths(a.paths)
    names = run_names(a.exp, a.paths) if a.exp else None
    exp_name = OmegaConf.load(resolve_path(a.exp)).get("name", Path(a.exp).stem) if a.exp else "all"
    out = resolve_path(a.out or f"outputs/{exp_name}")
    out.mkdir(parents=True, exist_ok=True)

    df = A.collect(resolve_path(paths.runs_root), names)
    if df.empty:
        print("no results found")
        return
    df.to_csv(out / "results_long.csv", index=False)
    diversity = A.load_pool_diversity(resolve_path(paths.pools_root))
    for proto, d in df.groupby("protocol"):
        table = A.summary_table(d)
        table.to_csv(out / f"{proto}_summary.csv", index=False)
        print(f"\n=== {proto} ===\n{table.round(4).to_string(index=False)}")
        er = A.add_effective_robustness(d)
        if not er.empty:
            er_tab = er.groupby(["init_name", "domain"])["eff_robustness"].mean().unstack()
            er_tab.to_csv(out / f"{proto}_effective_robustness.csv")
            print(f"\neffective robustness (OOD residual vs baseline fit):\n{er_tab.round(4).to_string()}")
        A.plot_domain_bars(d, out / f"{proto}_domains.png", label_fraction=d.label_fraction.max())
        A.plot_id_vs_ood(d, out / f"{proto}_id_vs_ood.png")
        if d.label_fraction.nunique() > 1:
            A.plot_label_scaling(d, out / f"{proto}_label_scaling.png")
        A.plot_diversity_vs_ood(d, diversity, out / f"{proto}_diversity_vs_ood.png")
    print(f"\nwrote tables and figures to {out}")


if __name__ == "__main__":
    main()

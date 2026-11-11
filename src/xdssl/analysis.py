"""Collects fine-tuning results into tables and produces the paper figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics.robustness import effective_robustness, fit_baseline

PRIMARY = {"seg": "mIoU", "det": "AP"}


def collect(runs_root: str | Path, run_names: list[str] | None = None) -> pd.DataFrame:
    """Long table: one row per (run, domain) with the task's primary metric."""
    rows = []
    for p in sorted((Path(runs_root) / "finetune").glob("*/results.json")):
        if run_names is not None and p.parent.name not in run_names:
            continue
        r = json.loads(p.read_text())
        m, metric = r["meta"], PRIMARY[r["meta"]["task"]]
        in_domain = r.get("summary", {}).get("in_domain")
        for dom, res in r["domains"].items():
            rows.append({"run": p.parent.name, "task": m["task"], "protocol": m.get("protocol", m["task"]),
                         "init_name": m.get("init_name", m["init"]),
                         "init": m["init"], "pool": m.get("pool"), "label_fraction": m["label_fraction"],
                         "seed": m["seed"], "domain": dom, "is_id": dom == in_domain, "metric": metric,
                         "value": res.get(metric)})
    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +- std over seeds, domains as columns, plus mean/worst OOD."""
    g = df.groupby(["task", "init_name", "label_fraction", "domain"])["value"].agg(["mean", "std", "count"]).reset_index()
    wide = g.pivot_table(index=["task", "init_name", "label_fraction"], columns="domain", values="mean")
    per_run = df.groupby(["task", "init_name", "label_fraction", "seed"]).apply(_run_summary, include_groups=False)
    agg = per_run.groupby(level=[0, 1, 2]).agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return wide.join(agg).reset_index()


def _run_summary(g: pd.DataFrame) -> pd.Series:
    idv = g.loc[g.is_id, "value"]
    ood = g.loc[~g.is_id, "value"].dropna()
    return pd.Series({"id": idv.iloc[0] if len(idv) else np.nan,
                      "mean_ood": ood.mean() if len(ood) else np.nan,
                      "worst_ood": ood.min() if len(ood) else np.nan})


def add_effective_robustness(df: pd.DataFrame, baseline_inits=("random", "imagenet")) -> pd.DataFrame:
    """Per (task, OOD domain): fit OOD~ID over baseline runs (all label fractions/seeds), then report every run's
    residual. Requires >= 2 distinct baseline points per domain (e.g. several label fractions)."""
    out = []
    for (task, dom), g in df[~df.is_id].groupby(["task", "domain"]):
        ids = df[df.is_id & (df.task == task)].set_index("run")["value"]
        g = g.assign(id_value=g["run"].map(ids)).dropna(subset=["id_value", "value"])
        base = g[g.init_name.isin(baseline_inits)]
        if base["id_value"].nunique() < 2:
            continue
        fit = fit_baseline(base["id_value"], base["value"])
        g = g.assign(eff_robustness=[effective_robustness(i, o, fit) for i, o in zip(g.id_value, g.value)],
                     fit_slope=fit[0], fit_intercept=fit[1])
        out.append(g)
    return pd.concat(out) if out else pd.DataFrame()


def paired_comparison(df: pd.DataFrame, init_a: str, init_b: str, n_boot: int = 10000, seed: int = 0) -> pd.DataFrame:
    """Paired test of `init_a` vs `init_b`: pairs are runs matching on (task, label_fraction, seed, domain), one
    row per task. Reports mean difference (a - b) with a bootstrap 95% CI and a Wilcoxon signed-rank p-value."""
    from scipy.stats import wilcoxon

    key = ["task", "label_fraction", "seed", "domain"]
    rng = np.random.default_rng(seed)
    rows = []
    for task, d in df[~df.is_id].groupby("task"):
        a = d[d.init_name == init_a].set_index(key)["value"]
        b = d[d.init_name == init_b].set_index(key)["value"]
        diff = (a - b).dropna().to_numpy()
        if len(diff) == 0:
            continue
        boots = rng.choice(diff, size=(n_boot, len(diff))).mean(1)
        p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
        rows.append({"task": task, "a": init_a, "b": init_b, "n_pairs": len(diff), "mean_diff": diff.mean(),
                     "ci95_lo": np.percentile(boots, 2.5), "ci95_hi": np.percentile(boots, 97.5), "wilcoxon_p": p})
    return pd.DataFrame(rows)


def load_pool_diversity(pools_root: str | Path) -> pd.DataFrame:
    rows = []
    for p in Path(pools_root).glob("*/diversity.json"):
        d = json.loads(p.read_text())
        rows.append({"pool": p.parent.name, **{k: v for k, v in d.items() if isinstance(v, (int, float))}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- plots
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_pool_composition(pools_root: str | Path, pools: list[str], out: Path):
    """One stacked bar per pool (fraction of frames) for each of dataset / weather / time of day."""
    plt = _plt()
    summaries = [json.loads((Path(pools_root) / p / "summary.json").read_text()) for p in pools]
    fields = [("by_dataset", "dataset"), ("by_weather", "weather"), ("by_timeofday", "time of day")]
    fig, axes = plt.subplots(1, len(fields), figsize=(4 * len(fields), 3.2), squeeze=False)
    for ax, (key, title) in zip(axes[0], fields):
        cats = sorted({c for s in summaries for c in s[key]})
        bottom = np.zeros(len(pools))
        for c in cats:
            frac = np.array([s[key].get(c, 0) / s["n_frames"] for s in summaries])
            ax.bar(pools, frac, bottom=bottom, label=c)
            bottom += frac
        ax.set_title(title), ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout(), fig.savefig(out, dpi=150), plt.close(fig)


def plot_domain_bars(df: pd.DataFrame, out: Path, label_fraction: float | None = None):
    plt = _plt()
    d = df if label_fraction is None else df[df.label_fraction == label_fraction]
    g = d.groupby(["init_name", "domain"])["value"].agg(["mean", "std"]).reset_index()
    domains = list(dict.fromkeys(d.sort_values("is_id", ascending=False)["domain"]))
    inits = sorted(g.init_name.unique())
    width = 0.8 / max(1, len(inits))
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(domains)), 4))
    for i, init in enumerate(inits):
        s = g[g.init_name == init].set_index("domain").reindex(domains)
        ax.bar(np.arange(len(domains)) + i * width, 100 * s["mean"], width, yerr=100 * s["std"].fillna(0), label=init)
    ax.set_xticks(np.arange(len(domains)) + width * (len(inits) - 1) / 2, domains, rotation=30, ha="right")
    ax.set_ylabel(df.metric.iloc[0]), ax.set_title("Per-domain performance (first = in-domain)")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout(), fig.savefig(out, dpi=150), plt.close(fig)


def plot_id_vs_ood(df: pd.DataFrame, out: Path, baseline_inits=("random", "imagenet")):
    """The effective-robustness plot: points above the baseline line generalize better than ID accuracy predicts."""
    plt = _plt()
    per_run = df.groupby(["run", "init_name"]).apply(_run_summary, include_groups=False).reset_index().dropna()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for init, g in per_run.groupby("init_name"):
        ax.scatter(100 * g["id"], 100 * g["mean_ood"], label=init, s=25, marker="s" if init in baseline_inits else "o")
    base = per_run[per_run.init_name.isin(baseline_inits)]
    if base["id"].nunique() >= 2:
        a, b = fit_baseline(base["id"], base["mean_ood"])
        xs = np.linspace(per_run["id"].min(), per_run["id"].max(), 50)
        z = a * np.log(xs / (1 - xs)) + b
        ax.plot(100 * xs, 100 / (1 + np.exp(-z)), "k--", lw=1, label="baseline fit (logit)")
    ax.set_xlabel("in-domain"), ax.set_ylabel("mean out-of-domain"), ax.legend(fontsize=7)
    fig.tight_layout(), fig.savefig(out, dpi=150), plt.close(fig)


def plot_diversity_vs_ood(df: pd.DataFrame, diversity: pd.DataFrame, out: Path, x: str = "vendi"):
    plt = _plt()
    if diversity.empty or x not in diversity:
        return
    per_run = df[df.init == "ssl"].groupby(["run", "pool"]).apply(_run_summary, include_groups=False).reset_index()
    m = per_run.merge(diversity, on="pool").groupby("pool").agg(
        div=(x, "first"), ood=("mean_ood", "mean"), ood_sd=("mean_ood", "std"), idm=("id", "mean"), id_sd=("id", "std"),
    ).reset_index()
    if m.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(m["div"], 100 * m["ood"], yerr=100 * m["ood_sd"].fillna(0), fmt="o", label="mean OOD")
    ax.errorbar(m["div"], 100 * m["idm"], yerr=100 * m["id_sd"].fillna(0), fmt="^", alpha=0.6, label="ID")
    for _, r in m.iterrows():
        ax.annotate(r["pool"], (r["div"], 100 * r["ood"]), fontsize=6)
    ax.set_xlabel(f"pool diversity ({x})"), ax.set_ylabel(df.metric.iloc[0]), ax.legend(fontsize=7)
    fig.tight_layout(), fig.savefig(out, dpi=150), plt.close(fig)


def plot_label_scaling(df: pd.DataFrame, out: Path):
    plt = _plt()
    per_run = df.groupby(["run", "init_name", "label_fraction"]).apply(_run_summary, include_groups=False).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True)
    for ax, col, title in zip(axes, ["id", "mean_ood"], ["in-domain", "mean out-of-domain"]):
        for init, g in per_run.groupby("init_name"):
            s = g.groupby("label_fraction")[col].agg(["mean", "std"])
            ax.errorbar(s.index, 100 * s["mean"], yerr=100 * s["std"].fillna(0), marker="o", label=init, capsize=2)
        ax.set_xscale("log"), ax.set_xlabel("labeled fraction of source data"), ax.set_title(title)
    axes[0].set_ylabel(df.metric.iloc[0]), axes[1].legend(fontsize=7)
    fig.tight_layout(), fig.savefig(out, dpi=150), plt.close(fig)

"""Robustness summaries.

* relative robustness: OOD / ID performance.
* effective robustness (Taori et al., 2020): OOD performance *beyond* what a model's ID performance predicts,
  using a linear fit (in logit space) over a set of baseline models. This separates "more robust" from
  "simply better", which is essential for the diversity claim - a pool could raise OOD numbers only because it
  raises ID numbers too.
"""

from __future__ import annotations

import numpy as np


def _logit(p, eps=1e-4):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def summarize_domains(results: dict[str, dict], metric: str, in_domain: str) -> dict:
    vals = {d: r[metric] for d, r in results.items() if metric in r and np.isfinite(r[metric])}
    if in_domain not in vals:
        return {}
    ood = {d: v for d, v in vals.items() if d != in_domain}
    out = {"metric": metric, "in_domain": in_domain, "id": vals[in_domain]}
    if ood:
        out.update({
            "mean_ood": float(np.mean(list(ood.values()))),
            "worst_ood": float(min(ood.values())),
            "worst_ood_domain": min(ood, key=ood.get),
            "relative_robustness": float(np.mean(list(ood.values())) / vals[in_domain]) if vals[in_domain] > 0 else float("nan"),
        })
    return out


def fit_baseline(id_scores, ood_scores, logit: bool = True):
    """Linear fit OOD ~ a*ID + b over baseline models. Scores in [0,1]."""
    x, y = (_logit(id_scores), _logit(ood_scores)) if logit else (np.asarray(id_scores), np.asarray(ood_scores))
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def effective_robustness(id_score, ood_score, fit, logit: bool = True) -> float:
    a, b = fit
    pred = _sigmoid(a * _logit(id_score) + b) if logit else a * id_score + b
    return float(ood_score - pred)

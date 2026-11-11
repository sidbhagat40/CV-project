"""Quantifying how diverse a pretraining pool is - the x-axis of the central claim.

* metadata_entropy: normalized Shannon entropy over metadata bins (dataset/location/weather/time-of-day).
  Cheap and interpretable, but only as good as the tags.
* vendi_score (Friedman & Dieng, 2023): effective number of distinct samples in an embedding space,
  exp(entropy of the eigenvalues of the normalized similarity matrix). Tag-free.
* frechet_distance: FID-style distance between a pool and a target domain in embedding space, useful to
  check whether gains track pool->target proximity rather than diversity per se.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg


def metadata_entropy(df: pd.DataFrame, cols: list[str], normalized: bool = True) -> float:
    p = df.groupby(cols).size().to_numpy(dtype=np.float64)
    p = p / p.sum()
    h = float(-(p * np.log(p)).sum())
    if not normalized:
        return h
    return h / np.log(len(p)) if len(p) > 1 else 0.0


def vendi_score(features: np.ndarray) -> float:
    x = features / np.linalg.norm(features, axis=1, keepdims=True).clip(1e-12)
    n = x.shape[0]
    # eigenvalues of K/n (K = x x^T) equal those of x^T x / n, which is cheaper when n > d
    gram = (x.T @ x) / n if n > x.shape[1] else (x @ x.T) / n
    eig = np.clip(np.linalg.eigvalsh(gram), 0, None)
    eig = eig[eig > 1e-12]
    return float(np.exp(-(eig * np.log(eig)).sum()))


def frechet_distance(feats_a: np.ndarray, feats_b: np.ndarray) -> float:
    mu_a, mu_b = feats_a.mean(0), feats_b.mean(0)
    cov_a, cov_b = np.cov(feats_a, rowvar=False), np.cov(feats_b, rowvar=False)
    covmean, _ = linalg.sqrtm(cov_a @ cov_b, disp=False)
    covmean = covmean.real
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(cov_a + cov_b - 2 * covmean))

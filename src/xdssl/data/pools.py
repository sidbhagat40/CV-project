"""Pretraining pools: fixed-size samples of the unified frame index.

A pool spec (configs/pools/*.yaml) says *which* frames are eligible and *how* the frame budget is split:

    name: bdd_nusc_waymo_200k
    size: 200000                       # total frames; keep equal across pools you compare (iso-size)
    seed: 0
    sources:
      - {dataset: bdd100k, weight: 1, filter: {split: [train, test]}}
      - {dataset: nuscenes, weight: 1, filter: {split: train}}
      - {dataset: waymo,    weight: 1, filter: {split: train}}
    balance_by: [timeofday, weather]   # optional: equalize strata inside each source (water-filling)
    max_per_sequence: null             # optional cap on frames per video/scene
    exclude_ids_files: []              # text files of sequence_ids to drop (evaluation-set leakage guard)

Sizes are honoured exactly or an error is raised, because silently shrinking one pool would break the
iso-size comparison the whole study relies on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.config import to_dict
from .frames import manifest_path


def load_frame_index(frames_root: str | Path, datasets) -> pd.DataFrame:
    dfs = []
    for ds in datasets:
        p = manifest_path(frames_root, ds)
        if not p.exists():
            raise FileNotFoundError(f"no manifest for dataset '{ds}' at {p}; run scripts/prepare_frames.py first")
        dfs.append(pd.read_parquet(p))
    return pd.concat(dfs, ignore_index=True)


def apply_filter(df: pd.DataFrame, include: dict | None = None, exclude: dict | None = None) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    for col, val in (include or {}).items():
        vals = val if isinstance(val, (list, tuple)) else [val]
        mask &= df[col].isin(vals).to_numpy()
    for col, val in (exclude or {}).items():
        vals = val if isinstance(val, (list, tuple)) else [val]
        mask &= ~df[col].isin(vals).to_numpy()
    return df[mask]


def waterfill(total: int, capacities: list[int], weights: list[float] | None = None) -> list[int]:
    """Split `total` proportionally to `weights`, never exceeding `capacities`; leftover from saturated bins
    is redistributed to the others. Raises if sum(capacities) < total."""
    n = len(capacities)
    weights = list(weights) if weights is not None else [1.0] * n
    if sum(capacities) < total:
        raise ValueError(f"requested {total} frames but only {sum(capacities)} available")
    alloc = [0] * n
    active = [i for i in range(n) if capacities[i] > 0 and weights[i] > 0]
    remaining = total
    while remaining > 0 and active:
        wsum = sum(weights[i] for i in active)
        shares = {i: remaining * weights[i] / wsum for i in active}
        floors = {i: int(np.floor(s)) for i, s in shares.items()}
        leftover = remaining - sum(floors.values())
        for i in sorted(active, key=lambda i: shares[i] - floors[i], reverse=True)[:leftover]:
            floors[i] += 1
        for i in active:
            give = min(floors[i], capacities[i] - alloc[i])
            alloc[i] += give
            remaining -= give
        active = [i for i in active if alloc[i] < capacities[i]]
    if remaining:
        raise ValueError("waterfill could not place all frames (all weighted bins saturated)")
    return alloc


def _cap_per_sequence(df: pd.DataFrame, cap: int | None, rng: np.random.Generator) -> pd.DataFrame:
    if not cap:
        return df
    shuffled = df.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
    return shuffled[shuffled.groupby("sequence_id").cumcount() < cap]


def _sample_source(df: pd.DataFrame, n: int, balance_by, rng: np.random.Generator) -> pd.DataFrame:
    if not balance_by:
        return df.sample(n=n, random_state=int(rng.integers(1 << 31)))
    groups = [g for _, g in df.groupby(list(balance_by))]
    alloc = waterfill(n, [len(g) for g in groups])
    parts = [g.sample(n=k, random_state=int(rng.integers(1 << 31))) for g, k in zip(groups, alloc) if k]
    return pd.concat(parts)


def read_exclude_ids(files) -> set[str]:
    ids = set()
    for f in files or []:
        if not Path(f).exists():
            print(f"WARNING: exclusion file {f} not found - evaluation images may leak into the pool. "
                  f"Create it with scripts/prepare_labeled.py before building final pools.")
            continue
        with open(f) as fh:
            ids.update(line.strip() for line in fh if line.strip())
    return ids


def build_pool(spec, frames_root: str | Path) -> pd.DataFrame:
    spec = to_dict(spec)
    rng = np.random.default_rng(spec.get("seed", 0))
    sources = spec["sources"]
    index = load_frame_index(frames_root, sorted({s["dataset"] for s in sources}))
    excluded = read_exclude_ids(spec.get("exclude_ids_files"))
    if excluded:
        index = index[~index["sequence_id"].isin(excluded)]

    eligible = []
    for s in sources:
        df = apply_filter(index[index["dataset"] == s["dataset"]], s.get("filter"), s.get("exclude"))
        df = _cap_per_sequence(df, s.get("max_per_sequence", spec.get("max_per_sequence")), rng)
        if df.empty:
            raise ValueError(f"source {s} has no eligible frames")
        eligible.append(df)

    alloc = waterfill(spec["size"], [len(d) for d in eligible], [float(s.get("weight", 1.0)) for s in sources]) \
        if spec.get("allow_reweight", False) else _strict_alloc(spec["size"], eligible, sources)
    parts = [_sample_source(df, n, spec.get("balance_by"), rng) for df, n in zip(eligible, alloc) if n]
    pool = pd.concat(parts).sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    assert len(pool) == spec["size"] and pool["frame_id"].is_unique
    return pool


def _strict_alloc(total: int, eligible, sources) -> list[int]:
    weights = np.array([float(s.get("weight", 1.0)) for s in sources])
    quotas = np.floor(total * weights / weights.sum()).astype(int)
    quotas[: total - quotas.sum()] += 1
    for q, df, s in zip(quotas, eligible, sources):
        if q > len(df):
            raise ValueError(
                f"source '{s['dataset']}' needs {q} frames but only {len(df)} are eligible. Extract more frames, "
                f"lower `size`, or set `allow_reweight: true` to let other sources fill the gap.")
    return quotas.tolist()


def summarize_pool(pool: pd.DataFrame) -> dict:
    from ..metrics.diversity import metadata_entropy

    def counts(col):
        return {str(k): int(v) for k, v in pool[col].value_counts().items()}

    return {
        "n_frames": int(len(pool)),
        "n_sequences": int(pool["sequence_id"].nunique()),
        "by_dataset": counts("dataset"),
        "by_weather": counts("weather"),
        "by_timeofday": counts("timeofday"),
        "by_location": counts("location"),
        "entropy": {
            "dataset": metadata_entropy(pool, ["dataset"]),
            "weather": metadata_entropy(pool, ["weather"]),
            "timeofday": metadata_entropy(pool, ["timeofday"]),
            "location": metadata_entropy(pool, ["location"]),
            "joint_conditions": metadata_entropy(pool, ["dataset", "location", "weather", "timeofday"]),
        },
    }


def pool_dir(pools_root: str | Path, name: str) -> Path:
    return Path(pools_root) / name

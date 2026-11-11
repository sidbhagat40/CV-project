"""Expands an experiment spec (configs/experiments/*.yaml) into an ordered list of idempotent jobs.

Run names are deterministic functions of their settings, so fine-tuning runs that appear in several experiments
(e.g. the random/ImageNet baselines) are trained once and shared.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from .utils.config import CONFIG_DIR, REPO_ROOT, load_config, load_paths, resolve_path

SCRIPTS = REPO_ROOT / "scripts"


@dataclass
class Job:
    kind: str                 # pool | pretrain | finetune
    name: str
    output: Path              # job is skipped if this exists
    cmd: list[str]
    meta: dict = field(default_factory=dict)


def _fmt_frac(f: float) -> str:
    return f"{f:g}".replace(".", "p")


def finetune_run_name(init_name: str, ft_config: str, frac: float, seed: int, tag: str | None = None) -> str:
    name = f"{init_name}__{Path(ft_config).stem}__f{_fmt_frac(frac)}__s{seed}"
    return f"{name}__{tag}" if tag else name


def expand(exp_path: str | Path, paths_file: str | None = None) -> list[Job]:
    exp = OmegaConf.load(resolve_path(exp_path))
    paths = load_paths(paths_file)
    runs_root, pools_root = resolve_path(paths.runs_root), resolve_path(paths.pools_root)
    py = sys.executable
    jobs: list[Job] = []

    pt = exp.get("pretrain") or {}
    pt_cfg = pt.get("config")
    pt_overrides = list(pt.get("overrides") or [])
    pt_tag = pt.get("tag")
    method = arch = None
    if pt_cfg:
        pcfg = load_config(pt_cfg, pt_overrides, paths_file)
        method, arch = pcfg.method, pcfg.model.arch

    inits = [(b, b, None) for b in exp.get("baselines") or []]  # (init_name, init, ckpt)
    for pool in pt.get("pools") or []:
        spec = CONFIG_DIR / "pools" / f"{pool}.yaml"
        pool_file = pools_root / pool / "pool.parquet"
        jobs.append(Job("pool", pool, pool_file, [py, str(SCRIPTS / "build_pool.py"), "--spec", str(spec)]))
        pt_name = f"{method}_{arch}_{pool}" + (f"_{pt_tag}" if pt_tag else "")
        pt_dir = runs_root / "pretrain" / pt_name
        jobs.append(Job("pretrain", pt_name, pt_dir / "final.pt",
                        [py, str(SCRIPTS / "pretrain.py"), "--config", str(resolve_path(pt_cfg)),
                         f"data.pool={pool_file.as_posix()}", f"output_dir={pt_dir.as_posix()}", f"pool_name={pool}",
                         *pt_overrides]))
        inits.append((f"{method}-{pool}" + (f"-{pt_tag}" if pt_tag else ""), "ssl", pt_dir / "final.pt"))

    ft = exp.finetune
    ft_overrides = list(ft.get("overrides") or [])
    for ft_cfg in ft.configs:
        for init_name, init, ckpt in inits:
            for frac in ft.get("label_fractions", [1.0]):
                for seed in ft.get("seeds", [0]):
                    run = finetune_run_name(init_name, ft_cfg, frac, seed, ft.get("tag"))
                    out = runs_root / "finetune" / run
                    cmd = [py, str(SCRIPTS / "finetune.py"), "--config", str(resolve_path(ft_cfg)),
                           f"model.init={init}", f"data.label_fraction={frac}", f"seed={seed}",
                           f"output_dir={out.as_posix()}", f"meta.init_name={init_name}",
                           f"meta.pool={init_name.split('-', 1)[1] if init == 'ssl' else init}",
                           f"meta.pretrain_method={method if init == 'ssl' else 'none'}", f"meta.run={run}",
                           f"meta.protocol={Path(ft_cfg).stem}"]
                    if ckpt is not None:
                        cmd.append(f"model.ckpt={Path(ckpt).as_posix()}")
                    jobs.append(Job("finetune", run, out / "results.json", cmd + ft_overrides,
                                    {"init_name": init_name, "frac": frac, "seed": seed}))
    return jobs


def run_names(exp_path: str | Path, paths_file: str | None = None) -> list[str]:
    return [j.name for j in expand(exp_path, paths_file) if j.kind == "finetune"]

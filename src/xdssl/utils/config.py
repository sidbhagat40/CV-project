"""YAML config loading with `_base_` inheritance, CLI dot-list overrides and a shared `paths` block."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"


def resolve_path(p: str | os.PathLike | None) -> Path | None:
    """Resolve a path relative to the repository root (absolute paths are returned unchanged)."""
    if p is None:
        return None
    p = Path(str(p)).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def load_paths(paths_file: str | None = None) -> DictConfig:
    candidates = [paths_file, os.environ.get("XDSSL_PATHS"), CONFIG_DIR / "paths.yaml", CONFIG_DIR / "paths.example.yaml"]
    for c in candidates:
        if c and Path(c).exists():
            return OmegaConf.load(c)
    raise FileNotFoundError("No paths file found; copy configs/paths.example.yaml to configs/paths.yaml")


def _load_with_base(path: Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    bases = cfg.pop("_base_", None)
    if bases:
        bases = [bases] if isinstance(bases, str) else list(bases)
        merged = OmegaConf.merge(*[_load_with_base((path.parent / b).resolve()) for b in bases])
        cfg = OmegaConf.merge(merged, cfg)
    return cfg


def load_config(path: str | os.PathLike, overrides: Iterable[str] = (), paths_file: str | None = None) -> DictConfig:
    """Load `path`, merge its `_base_` chain, inject `paths`, then apply `key=value` overrides."""
    cfg = _load_with_base(resolve_path(path))
    base = OmegaConf.create({"paths": load_paths(paths_file)})
    cfg = OmegaConf.merge(base, cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    return cfg


def to_dict(cfg) -> dict:
    return OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)

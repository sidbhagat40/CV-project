"""Writing downsized frames and the per-dataset `manifest.parquet` that indexes them."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
from PIL import Image
from tqdm import tqdm

from .schema import FRAME_COLUMNS


def resize_short_side(img: Image.Image, short_side: int | None) -> Image.Image:
    if not short_side:
        return img
    w, h = img.size
    s = short_side / min(w, h)
    if s >= 1.0:
        return img
    return img.resize((round(w * s), round(h * s)), Image.BICUBIC)


def save_frame(img: Image.Image, out_path: Path, short_side: int | None, quality: int = 90) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resize_short_side(img.convert("RGB"), short_side).save(out_path, quality=quality)


def manifest_path(frames_root: str | Path, dataset: str) -> Path:
    return Path(frames_root) / dataset / "manifest.parquet"


def write_manifest(rows: list[dict] | pd.DataFrame, frames_root: str | Path, dataset: str) -> pd.DataFrame:
    """Append rows to the dataset manifest (deduplicated by frame_id) so extraction can run incrementally,
    e.g. one downloaded archive at a time."""
    new = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    missing = set(FRAME_COLUMNS) - set(new.columns)
    if missing and len(new):
        raise ValueError(f"manifest rows missing columns: {missing}")
    new = new.reindex(columns=FRAME_COLUMNS)
    path = manifest_path(frames_root, dataset)
    if path.exists():
        new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
    new = new.drop_duplicates("frame_id", keep="last").reset_index(drop=True)
    new["timestamp"] = new["timestamp"].astype("int64")
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path, index=False)
    return new


def parallel_map(fn: Callable, items: Iterable, workers: int = 8, desc: str = "") -> list:
    """Thread pool map (PIL / cv2 release the GIL during decode/encode). Returns results in input order,
    dropping `None`s."""
    items = list(items)
    out = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for r in tqdm(ex.map(fn, items), total=len(items), desc=desc):
            if r is None:
                continue
            if isinstance(r, list):
                out.extend(r)
            else:
                out.append(r)
    return out

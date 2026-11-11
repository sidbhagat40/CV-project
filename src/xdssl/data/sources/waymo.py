"""Waymo Open Dataset (Perception v2.x, Parquet format): ~1150 x 20 s segments, 5 cameras at 10 Hz,
Phoenix / San Francisco / Mountain View, with day/night/dawn and sunny/rain tags in the `stats` component.

Uses only pyarrow (no TensorFlow). Each `camera_image/{segment}.parquet` holds JPEG bytes for all cameras.
Column names follow the v2 schema; if your release differs, adjust the constants below.
"""

from __future__ import annotations

import io
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from ..frames import save_frame, write_manifest
from ..schema import make_frame_id, make_frame_relpath, normalize_timeofday, normalize_weather

DATASET = "waymo"
CAMERA_NAMES = {1: "front", 2: "front_left", 3: "front_right", 4: "side_left", 5: "side_right"}
SPLIT_DIRS = {"training": "train", "validation": "val", "testing": "test"}

COL_SEGMENT = "key.segment_context_name"
COL_TS = "key.frame_timestamp_micros"
COL_CAM = "key.camera_name"
COL_IMAGE = "[CameraImageComponent].image"
COL_STATS_TOD = "[StatsComponent].time_of_day"
COL_STATS_WEATHER = "[StatsComponent].weather"
COL_STATS_LOCATION = "[StatsComponent].location"


def read_segment_stats(split_dir: Path, segment: str) -> dict:
    p = split_dir / "stats" / f"{segment}.parquet"
    if not p.exists():
        return {"weather": "unknown", "timeofday": "unknown", "location": "unknown"}
    t = pq.read_table(p, columns=[COL_STATS_TOD, COL_STATS_WEATHER, COL_STATS_LOCATION]).slice(0, 1).to_pylist()[0]
    return {
        "weather": normalize_weather(t[COL_STATS_WEATHER]),
        "timeofday": normalize_timeofday(t[COL_STATS_TOD]),
        "location": str(t[COL_STATS_LOCATION]),
    }


def select_timestamps(timestamps_us, target_hz: float) -> set[int]:
    """Greedy temporal subsampling of a sorted unique timestamp list."""
    keep, last = set(), -1e18
    gap = 1e6 / target_hz if target_hz else 0
    for ts in sorted(set(timestamps_us)):
        if ts - last >= gap:
            keep.add(ts)
            last = ts
    return keep


def iter_segment_images(parquet_path: Path, cameras, keep_ts: set[int] | None, batch_size: int = 32):
    """Yields (timestamp_us, camera_id, jpeg_bytes)."""
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=[COL_TS, COL_CAM, COL_IMAGE]):
        d = batch.to_pydict()
        for ts, cam, img in zip(d[COL_TS], d[COL_CAM], d[COL_IMAGE]):
            if cam in cameras and (keep_ts is None or ts in keep_ts):
                yield ts, cam, img


def index_waymo(waymo_root, frames_root, splits=("training",), cameras=(1, 2, 3, 4, 5), target_hz=1.0,
                short_side=256, delete_source=False):
    waymo_root, frames_root = Path(waymo_root), Path(frames_root)
    rows = []
    for split_name in splits:
        split_dir = waymo_root / split_name
        files = sorted((split_dir / "camera_image").glob("*.parquet"))
        if not files:
            print(f"[waymo] no camera_image parquet files under {split_dir}")
            continue
        for i, path in enumerate(tqdm(files, desc=f"waymo/{split_name}")):
            if i and i % 20 == 0:  # flush often: with delete_source=True the raw file is gone after this loop
                write_manifest(rows, frames_root, DATASET)
                rows = []
            seg = path.stem
            meta = read_segment_stats(split_dir, seg)
            ts_all = pq.read_table(path, columns=[COL_TS]).column(COL_TS).to_pylist()
            keep = select_timestamps(ts_all, target_hz)
            t0 = min(ts_all)
            for ts, cam, jpeg in iter_segment_images(path, set(cameras), keep):
                cam_name = CAMERA_NAMES.get(cam, f"cam{cam}")
                ts_ms = (ts - t0) // 1000
                rel = make_frame_relpath(DATASET, seg, cam_name, ts_ms)
                out = frames_root / rel
                if not out.exists():
                    save_frame(Image.open(io.BytesIO(jpeg)), out, short_side)
                rows.append({
                    "frame_id": make_frame_id(DATASET, seg, cam_name, ts_ms), "dataset": DATASET,
                    "split": SPLIT_DIRS.get(split_name, split_name), "sequence_id": seg, "camera": cam_name,
                    "timestamp": int(ts_ms), "path": rel, "scene_type": "unknown", **meta,
                })
            if delete_source:
                path.unlink()
        write_manifest(rows, frames_root, DATASET)
        rows = []
    return write_manifest(rows, frames_root, DATASET)

"""nuScenes: 1000 x 20 s scenes, 6 cameras, Boston + Singapore, with rain/night scenes noted in the text
description. Parses the raw JSON tables directly (no devkit dependency) and indexes whichever camera files are
present on disk, so partial blob downloads work.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

from ..frames import parallel_map, save_frame, write_manifest
from ..schema import make_frame_id, make_frame_relpath

DATASET = "nuscenes"
ALL_CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")


def _load(table_dir: Path, name: str) -> list[dict]:
    with open(table_dir / f"{name}.json") as f:
        return json.load(f)


def _official_split(scene_name: str, version: str) -> str | None:
    try:
        from nuscenes.utils.splits import create_splits_scenes  # optional dependency
    except ImportError:
        return None
    splits = create_splits_scenes()
    if "mini" in version:
        return "train" if scene_name in splits["mini_train"] else "val"
    return "train" if scene_name in splits["train"] else "val" if scene_name in splits["val"] else "test"


def _hash_split(scene_name: str, val_fraction: float = 0.15) -> str:
    h = int(hashlib.md5(scene_name.encode()).hexdigest(), 16) % 10_000
    return "val" if h < val_fraction * 10_000 else "train"


def _conditions(description: str) -> tuple[str, str]:
    d = description.lower()
    timeofday = "night" if "night" in d else "day"
    # nuScenes descriptions tag rain explicitly; the absence of a tag is treated as dry weather.
    weather = "rain" if "rain" in d else "clear"
    return weather, timeofday


def index_nuscenes(nusc_root, frames_root, version="v1.0-trainval", cameras=ALL_CAMERAS, target_hz=2.0,
                   short_side=256, workers=8):
    nusc_root, frames_root = Path(nusc_root), Path(frames_root)
    tdir = nusc_root / version
    scenes = {s["token"]: s for s in _load(tdir, "scene")}
    logs = {l["token"]: l for l in _load(tdir, "log")}
    samples = {s["token"]: s["scene_token"] for s in _load(tdir, "sample")}
    sensors = {s["token"]: s for s in _load(tdir, "sensor")}
    calib = {c["token"]: sensors[c["sensor_token"]] for c in _load(tdir, "calibrated_sensor")}

    per_stream = defaultdict(list)  # (scene_token, channel) -> [(timestamp_us, filename)]
    for sd in _load(tdir, "sample_data"):
        sensor = calib[sd["calibrated_sensor_token"]]
        if sensor["modality"] != "camera" or sensor["channel"] not in cameras:
            continue
        per_stream[(samples[sd["sample_token"]], sensor["channel"])].append((sd["timestamp"], sd["filename"]))

    min_gap_us = 1e6 / target_hz if target_hz else 0
    jobs = []
    for (scene_tok, channel), items in per_stream.items():
        scene = scenes[scene_tok]
        weather, timeofday = _conditions(scene.get("description", ""))
        split = _official_split(scene["name"], version) or _hash_split(scene["name"])
        location = logs[scene["log_token"]]["location"]
        items.sort()
        t0, last = items[0][0], -1e18
        for ts, fname in items:
            if ts - last < min_gap_us:
                continue
            last = ts
            jobs.append((scene["name"], channel, (ts - t0) // 1000, fname, split, weather, timeofday, location))

    def work(job):
        seq, channel, ts_ms, fname, split, weather, timeofday, location = job
        src = nusc_root / fname
        if not src.exists():
            return None
        cam = channel.lower().replace("cam_", "")
        rel = make_frame_relpath(DATASET, seq, cam, ts_ms)
        out = frames_root / rel
        if not out.exists():
            try:
                save_frame(Image.open(src), out, short_side)
            except OSError as e:
                print(f"[nuscenes] failed {src}: {e}")
                return None
        return {
            "frame_id": make_frame_id(DATASET, seq, cam, ts_ms), "dataset": DATASET, "split": split,
            "sequence_id": seq, "camera": cam, "timestamp": int(ts_ms), "path": rel, "weather": weather,
            "timeofday": timeofday, "location": location, "scene_type": "unknown",
        }

    rows = parallel_map(work, jobs, workers, desc="nuscenes")
    print(f"[nuscenes] indexed {len(rows)}/{len(jobs)} candidate frames (missing files are skipped)")
    return write_manifest(rows, frames_root, DATASET)

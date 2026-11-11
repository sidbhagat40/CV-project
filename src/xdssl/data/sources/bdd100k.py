"""BDD100K: 100K driving videos (40 s, 720p, US cities) with per-video weather / time-of-day / scene tags.

Two unlabeled sources are supported:
  * `index_bdd_images`  - the official 100K keyframes (one frame per video, ~5 GB). Cheapest way to get
                          maximal per-frame diversity; recommended starting point.
  * `index_bdd_videos`  - frames sampled from the .mov videos at `fps` (1.8 TB for all videos; download a
                          few archives at a time and run this incrementally).
Both use the video name as `sequence_id`, so frames from the same drive are grouped regardless of source.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
from PIL import Image

from ..frames import parallel_map, save_frame, write_manifest
from ..schema import make_frame_id, make_frame_relpath, normalize_timeofday, normalize_weather

DATASET = "bdd100k"
KEYFRAME_TIMESTAMP_MS = 10_000  # the 100K images are the frame at the 10th second of each video

_LABEL_CANDIDATES = [
    "labels/det_20/det_{split}.json",
    "labels/bdd100k_labels_images_{split}.json",
    "bdd100k_labels_images_{split}.json",
]


def find_root(bdd_root: str | Path) -> Path:
    """Accept either the `bdd100k/` folder itself or its parent."""
    bdd_root = Path(bdd_root)
    return bdd_root / "bdd100k" if (bdd_root / "bdd100k").is_dir() and not (bdd_root / "images").is_dir() else bdd_root


def load_attributes(bdd_root: str | Path, split: str) -> dict[str, dict]:
    """video/image stem -> {weather, timeofday, scene_type}. Empty for the unlabeled test split."""
    root = find_root(bdd_root)
    for cand in _LABEL_CANDIDATES:
        p = root / cand.format(split=split)
        if p.exists():
            with open(p) as f:
                items = json.load(f)
            out = {}
            for it in items:
                a = it.get("attributes") or {}
                out[Path(it["name"]).stem] = {
                    "weather": normalize_weather(a.get("weather")),
                    "timeofday": normalize_timeofday(a.get("timeofday")),
                    "scene_type": str(a.get("scene", "unknown")),
                }
            return out
    return {}


def _row(split, seq, ts, attrs, frames_root_rel):
    a = attrs.get(seq, {})
    return {
        "frame_id": make_frame_id(DATASET, seq, "front", ts),
        "dataset": DATASET, "split": split, "sequence_id": seq, "camera": "front", "timestamp": int(ts),
        "path": frames_root_rel,
        "weather": a.get("weather", "unknown"), "timeofday": a.get("timeofday", "unknown"),
        "location": "us", "scene_type": a.get("scene_type", "unknown"),
    }


def index_bdd_images(bdd_root, frames_root, splits=("train", "val", "test"), short_side=256, workers=8, overwrite=False):
    root, frames_root = find_root(bdd_root), Path(frames_root)
    rows = []
    for split in splits:
        img_dir = root / "images" / "100k" / split
        if not img_dir.is_dir():
            print(f"[bdd100k] skipping split '{split}': {img_dir} not found")
            continue
        attrs = load_attributes(root, split)

        def work(p: Path, split=split, attrs=attrs):
            seq = p.stem
            rel = make_frame_relpath(DATASET, seq, "front", KEYFRAME_TIMESTAMP_MS)
            out = frames_root / rel
            if overwrite or not out.exists():
                try:
                    save_frame(Image.open(p), out, short_side)
                except OSError as e:
                    print(f"[bdd100k] failed {p}: {e}")
                    return None
            return _row(split, seq, KEYFRAME_TIMESTAMP_MS, attrs, rel)

        rows += parallel_map(work, sorted(img_dir.glob("*.jpg")), workers, desc=f"bdd100k images/{split}")
    return write_manifest(rows, frames_root, DATASET)


def index_bdd_videos(bdd_root, frames_root, splits=("train", "val", "test"), fps=1.0, short_side=256, workers=4,
                     video_glob="*.mov"):
    root, frames_root = find_root(bdd_root), Path(frames_root)
    rows = []
    for split in splits:
        vid_dir = root / "videos" / split
        if not vid_dir.is_dir():
            print(f"[bdd100k] skipping videos/{split}: {vid_dir} not found")
            continue
        attrs = load_attributes(root, split)

        def work(p: Path, split=split, attrs=attrs):
            cap = cv2.VideoCapture(str(p))
            native = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step = max(1, round(native / fps))
            out_rows, idx = [], 0
            while cap.grab():
                if idx % step == 0:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    ts = round(idx / native * 1000)
                    rel = make_frame_relpath(DATASET, p.stem, "front", ts)
                    out = frames_root / rel
                    if not out.exists():
                        save_frame(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), out, short_side)
                    out_rows.append(_row(split, p.stem, ts, attrs, rel))
                idx += 1
            cap.release()
            return out_rows

        rows += parallel_map(work, sorted(vid_dir.glob(video_glob)), workers, desc=f"bdd100k videos/{split}")
    return write_manifest(rows, frames_root, DATASET)

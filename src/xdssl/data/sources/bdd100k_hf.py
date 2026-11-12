"""BDD100K 10K validation images from the Hugging Face FiftyOne mirror (`dgural/bdd100k`), used when the official
archives are unavailable. The mirror ships only the official *val* split, so it is re-split here:

* `val`   (evaluation): a fixed quota of images per condition domain, chosen by whole *video* so no drive is shared
  with the train partition,
* `train` (pretraining pools + detection fine-tuning): every other image.

Layout expected under `raw_dir`: `samples.json` and `data/<name>.jpg`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..frames import parallel_map, save_frame, write_manifest
from ..schema import make_frame_id, make_frame_relpath, normalize_timeofday, normalize_weather
from ..taxonomy import BDD_DET_MAP, det_class_index
from PIL import Image

DATASET = "bdd100k"
EVAL_QUOTA = {"clear_day": 400, "night": 500, "dawn_dusk": 150, "rain": 300, "snow": 300}


def condition_domain(weather: str, timeofday: str) -> str:
    """Mutually exclusive domain used only to allocate evaluation quotas."""
    if weather in ("rain", "snow"):
        return weather
    if timeofday in ("night", "dawn_dusk"):
        return timeofday
    return "clear_day" if weather == "clear" and timeofday == "day" else "other"


def _record(raw_dir: Path, s: dict) -> dict:
    stem = Path(s["filepath"]).stem
    w, h = s["metadata"]["width"], s["metadata"]["height"]
    boxes, labels = [], []
    for d in s["detections"]["detections"]:
        cls = BDD_DET_MAP.get(d["label"])
        if cls is None:
            continue
        x, y, bw, bh = d["bounding_box"]
        boxes.append([x * w, y * h, (x + bw) * w, (y + bh) * h])
        labels.append(det_class_index(cls))
    weather = normalize_weather(s["weather"]["label"])
    tod = normalize_timeofday(s["timeofday"]["label"])
    return {"image": str(raw_dir / s["filepath"]), "width": w, "height": h, "boxes": boxes, "labels": labels,
            "dataset": DATASET, "stem": stem,
            "attrs": {"weather": weather, "timeofday": tod, "scene_type": s["scene"]["label"], "location": "us"}}


def split_by_video(records: list[dict], quota: dict[str, int], seed: int = 0) -> dict[str, str]:
    """stem -> 'val' | 'train'. Whole videos (the stem prefix before '-') go to val until each domain quota is met."""
    videos: dict[str, list[dict]] = {}
    for r in records:
        videos.setdefault(r["stem"].split("-")[0], []).append(r)
    order = sorted(videos)
    np.random.default_rng(seed).shuffle(order)
    count = {k: 0 for k in quota}
    split = {r["stem"]: "train" for r in records}
    for v in order:
        doms = [condition_domain(r["attrs"]["weather"], r["attrs"]["timeofday"]) for r in videos[v]]
        if all(d in quota and count[d] < quota[d] for d in doms):
            for r, d in zip(videos[v], doms):
                split[r["stem"]] = "val"
                count[d] += 1
    return split


def prepare(raw_dir, frames_root, labeled_root, short_side: int = 256, quota: dict[str, int] | None = None,
            seed: int = 0, workers: int = 8) -> dict:
    from ...utils.misc import write_jsonl

    raw_dir, labeled_root = Path(raw_dir), Path(labeled_root)
    samples = json.loads((raw_dir / "samples.json").read_text())["samples"]
    records = [_record(raw_dir, s) for s in samples]
    split = split_by_video(records, quota or EVAL_QUOTA, seed)

    rows = []
    for r in records:
        seq = r["stem"].split("-")[0]
        ts = int(r["stem"].split("-")[1], 16) % (1 << 31)
        rows.append({"frame_id": make_frame_id(DATASET, seq, "front", ts), "dataset": DATASET, "split": split[r["stem"]],
                     "sequence_id": seq, "camera": "front", "timestamp": ts,
                     "path": make_frame_relpath(DATASET, seq, "front", ts), "weather": r["attrs"]["weather"],
                     "timeofday": r["attrs"]["timeofday"], "location": "us", "scene_type": r["attrs"]["scene_type"]})
    frames_root = Path(frames_root)

    def work(pair):
        rec, row = pair
        out = frames_root / row["path"]
        if not out.exists():
            save_frame(Image.open(rec["image"]), out, short_side)
        return row

    rows = parallel_map(work, list(zip(records, rows)), workers, "frames")
    write_manifest(rows, frames_root, DATASET)

    for name in ("train", "val"):
        write_jsonl([{k: v for k, v in r.items() if k != "stem"} for r in records if split[r["stem"]] == name],
                    labeled_root / f"det/bdd_det_{name}.jsonl")
    eval_videos = sorted({r["stem"].split("-")[0] for r in records if split[r["stem"]] == "val"})
    (labeled_root / "bdd_eval_ids.txt").write_text("\n".join(eval_videos) + "\n")
    return {"train": sum(v == "train" for v in split.values()), "val": sum(v == "val" for v in split.values())}

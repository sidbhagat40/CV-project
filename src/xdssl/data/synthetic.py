"""Tiny procedurally-generated "driving" data in exactly the on-disk formats the real converters produce
(frame manifests, seg/det JSONL, exclusion ids, a paths file). Used by the smoke tests and for trying the whole
pipeline before any real dataset is downloaded. Numbers obtained on it are meaningless."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..utils.misc import write_jsonl
from .frames import write_manifest
from .schema import make_frame_id, make_frame_relpath

SKY = {"day": (110, 160, 220), "night": (15, 15, 35), "dawn_dusk": (220, 140, 90)}
TINT = {"bdd100k": (1.0, 1.0, 1.0), "nuscenes": (0.85, 1.0, 1.15), "waymo": (1.15, 1.0, 0.85),
        "acdc": (1.0, 0.95, 0.9), "cityscapes": (0.95, 1.05, 0.95), "nuimages": (0.85, 1.0, 1.15)}


def render_scene(rng: np.random.Generator, timeofday="day", weather="clear", dataset="bdd100k", H=96, W=128):
    img = np.zeros((H, W, 3), np.float32)
    mask = np.full((H, W), 255, np.uint8)
    hz = int(H * rng.uniform(0.35, 0.5))
    img[:hz], mask[:hz] = SKY[timeofday], 10
    for _ in range(rng.integers(2, 5)):  # buildings
        x0, w, h = rng.integers(0, W - 10), rng.integers(10, 30), rng.integers(8, hz)
        img[hz - h:hz, x0:x0 + w], mask[hz - h:hz, x0:x0 + w] = rng.integers(60, 140), 2
    img[hz:], mask[hz:] = (90, 90, 90), 0
    side = W // 8
    img[hz:, :side], mask[hz:, :side] = (150, 140, 130), 1
    img[hz:, -side:], mask[hz:, -side:] = (150, 140, 130), 1
    boxes, labels = [], []
    for _ in range(rng.integers(1, 4)):  # vehicles
        w = int(rng.integers(14, 32)); h = int(w * rng.uniform(0.5, 0.8))
        x0, y0 = int(rng.integers(side, W - side - w)), int(rng.integers(hz, H - h))
        img[y0:y0 + h, x0:x0 + w], mask[y0:y0 + h, x0:x0 + w] = rng.integers(0, 255, 3), 13
        boxes.append([x0, y0, x0 + w, y0 + h]); labels.append(0)
    for _ in range(rng.integers(0, 3)):  # pedestrians
        w = int(rng.integers(4, 7)); h = int(w * rng.uniform(2.2, 3.0))
        x0, y0 = int(rng.integers(0, W - w)), int(rng.integers(max(0, hz - h // 2), H - h))
        img[y0:y0 + h, x0:x0 + w], mask[y0:y0 + h, x0:x0 + w] = (200, 60, 60), 11
        boxes.append([x0, y0, x0 + w, y0 + h]); labels.append(1)
    img *= np.array(TINT.get(dataset, (1, 1, 1)), np.float32)
    if timeofday == "night":
        img *= 0.35
    if weather == "fog":
        img = 0.45 * img + 0.55 * 190
    elif weather == "rain":
        for _ in range(60):
            x, y = rng.integers(0, W), rng.integers(0, H - 6)
            img[y:y + 6, x] = 200
    elif weather == "snow":
        ys, xs = rng.integers(0, H, 150), rng.integers(0, W, 150)
        img[ys, xs] = 250
    elif weather == "overcast":
        img = 0.8 * img + 0.2 * 128
    img += rng.normal(0, 6, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)), mask, boxes, labels


def _conditions(rng, clear_day_p=0.5):
    if rng.random() < clear_day_p:
        return "day", "clear"
    return str(rng.choice(["day", "night", "dawn_dusk"])), str(rng.choice(["clear", "rain", "snow", "fog", "overcast"]))


def make_synthetic(root: str | Path, seed: int = 0) -> Path:
    root = Path(root).resolve()
    rng = np.random.default_rng(seed)
    frames_root, labeled_root = root / "frames", root / "labeled"

    plan = {"bdd100k": (80, 1, "us"), "nuscenes": (10, 6, "boston-seaport"), "waymo": (10, 6, "location_sf")}
    for ds, (n_seq, per_seq, loc) in plan.items():
        rows = []
        for s in range(n_seq):
            tod, wx = _conditions(rng)
            seq = f"{ds[:4]}{s:04d}"
            for k in range(per_seq):
                ts = k * 500
                img, *_ = render_scene(rng, tod, wx, ds)
                rel = make_frame_relpath(ds, seq, "front", ts)
                (frames_root / rel).parent.mkdir(parents=True, exist_ok=True)
                img.save(frames_root / rel, quality=90)
                rows.append({"frame_id": make_frame_id(ds, seq, "front", ts), "dataset": ds,
                             "split": "train" if s < n_seq - 2 else "val", "sequence_id": seq, "camera": "front",
                             "timestamp": ts, "path": rel, "weather": wx, "timeofday": tod, "location": loc,
                             "scene_type": "city street"})
        write_manifest(rows, frames_root, ds)

    def labeled(name, n, dataset, attrs_fn, seg: bool):
        recs = []
        for i in range(n):
            attrs = attrs_fn(i)
            img, mask, boxes, labels = render_scene(rng, attrs.get("timeofday", "day"), attrs.get("weather", "clear"), dataset)
            ip = labeled_root / "synth_images" / name / f"{dataset}_{i:04d}.jpg"
            ip.parent.mkdir(parents=True, exist_ok=True)
            img.save(ip, quality=95)
            if seg:
                mp = ip.with_suffix(".png")
                Image.fromarray(mask).save(mp)
                recs.append({"image": str(ip), "mask": str(mp), "label_format": "trainid", "dataset": dataset, "attrs": attrs})
            else:
                recs.append({"image": str(ip), "width": img.width, "height": img.height, "boxes": boxes, "labels": labels,
                             "dataset": dataset, "attrs": attrs})
        return recs

    def cond_attrs(_):
        tod, wx = _conditions(rng)
        return {"timeofday": tod, "weather": wx, "location": "us"}

    acdc_cond = ["fog", "night", "rain", "snow"]
    write_jsonl(labeled("bdd_seg_train", 40, "bdd100k", cond_attrs, True), labeled_root / "seg/bdd_seg_train.jsonl")
    bdd_seg_val = labeled("bdd_seg_val", 12, "bdd100k", cond_attrs, True)
    write_jsonl(bdd_seg_val, labeled_root / "seg/bdd_seg_val.jsonl")
    write_jsonl(labeled("cityscapes_val", 8, "cityscapes", lambda i: {"timeofday": "day", "weather": "clear"}, True),
                labeled_root / "seg/cityscapes_val.jsonl")
    write_jsonl(labeled("acdc_val", 16, "acdc", lambda i: {
        "condition": acdc_cond[i % 4], "timeofday": "night" if acdc_cond[i % 4] == "night" else "day",
        "weather": acdc_cond[i % 4] if acdc_cond[i % 4] != "night" else "unknown"}, True), labeled_root / "seg/acdc_val.jsonl")
    write_jsonl(labeled("bdd_det_train", 48, "bdd100k", cond_attrs, False), labeled_root / "det/bdd_det_train.jsonl")
    bdd_det_val = labeled("bdd_det_val", 30, "bdd100k", lambda i: {
        "timeofday": ["day", "night", "dawn_dusk"][i % 3], "weather": ["clear", "rain", "snow", "fog", "clear"][i % 5]}, False)
    write_jsonl(bdd_det_val, labeled_root / "det/bdd_det_val.jsonl")
    write_jsonl(labeled("nuimages_val", 10, "nuimages", lambda i: {"region": ["boston", "singapore"][i % 2]}, False),
                labeled_root / "det/nuimages_val.jsonl")
    write_jsonl(labeled("waymo_val", 10, "waymo", lambda i: {"timeofday": ["day", "night"][i % 2]}, False),
                labeled_root / "det/waymo_val.jsonl")
    (labeled_root / "bdd_eval_ids.txt").write_text("\n".join(Path(r["image"]).stem for r in bdd_seg_val + bdd_det_val) + "\n")

    paths_file = root / "paths.yaml"
    paths_file.write_text(
        "raw: {}\n"
        f"frames_root: {frames_root.as_posix()}\n"
        f"pools_root: {(root / 'pools').as_posix()}\n"
        f"labeled_root: {labeled_root.as_posix()}\n"
        f"runs_root: {(root / 'runs').as_posix()}\n")
    return paths_file

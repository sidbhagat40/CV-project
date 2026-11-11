"""Converters from raw annotated datasets into unified JSONL records.

Segmentation record: {"image", "mask", "label_format": "trainid"|"cityscapes_id", "dataset", "attrs": {...}}
Detection record:    {"image", "width", "height", "boxes": [[x1,y1,x2,y2],...], "labels": [int,...],
                      "dataset", "attrs": {...}}   labels index into taxonomy.DET_CLASSES
`attrs` always carries weather / timeofday / location when the source provides them, so evaluation domains
(configs/eval/*.yaml) are just filters over attrs.
"""

from __future__ import annotations

import io
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from ..utils.misc import write_jsonl
from .schema import normalize_timeofday, normalize_weather
from .sources import bdd100k as bdd
from .taxonomy import BDD_DET_MAP, WAYMO_DET_MAP, det_class_index, map_nuimages_category


# ----------------------------------------------------------------------------- segmentation
def bdd_seg(bdd_root, split: str) -> list[dict]:
    root = bdd.find_root(bdd_root)
    layouts = [  # (image dir, mask dir, mask suffix)
        (root / "images/10k" / split, root / "labels/sem_seg/masks" / split, ".png"),
        (root / "seg/images" / split, root / "seg/labels" / split, "_train_id.png"),
    ]
    for img_dir, mask_dir, suffix in layouts:
        if mask_dir.is_dir():
            recs = []
            for img in sorted(img_dir.glob("*.jpg")):
                m = mask_dir / f"{img.stem}{suffix}"
                if m.exists():
                    recs.append({"image": str(img), "mask": str(m), "label_format": "trainid", "dataset": "bdd100k",
                                 "attrs": {"location": "us"}})
            return recs
    raise FileNotFoundError(f"BDD100K sem_seg masks not found under {root}")


def cityscapes_seg(cs_root, split: str) -> list[dict]:
    cs_root = Path(cs_root)
    recs = []
    for img in sorted((cs_root / "leftImg8bit" / split).glob("*/*_leftImg8bit.png")):
        city, stem = img.parent.name, img.name.replace("_leftImg8bit.png", "")
        gt = cs_root / "gtFine" / split / city
        train_ids, ids = gt / f"{stem}_gtFine_labelTrainIds.png", gt / f"{stem}_gtFine_labelIds.png"
        if train_ids.exists():
            mask, fmt = train_ids, "trainid"
        elif ids.exists():
            mask, fmt = ids, "cityscapes_id"
        else:
            continue
        recs.append({"image": str(img), "mask": str(mask), "label_format": fmt, "dataset": "cityscapes",
                     "attrs": {"location": city, "weather": "clear", "timeofday": "day"}})
    return recs


ACDC_CONDITIONS = ("fog", "night", "rain", "snow")


def acdc_seg(acdc_root, split: str = "val") -> list[dict]:
    acdc_root = Path(acdc_root)
    recs = []
    for cond in ACDC_CONDITIONS:
        for img in sorted((acdc_root / "rgb_anon" / cond / split).glob("*/*_rgb_anon.png")):
            mask = acdc_root / "gt" / cond / split / img.parent.name / img.name.replace("_rgb_anon.png", "_gt_labelTrainIds.png")
            if mask.exists():
                recs.append({"image": str(img), "mask": str(mask), "label_format": "trainid", "dataset": "acdc",
                             "attrs": {"condition": cond, "location": "switzerland",
                                       "weather": cond if cond != "night" else "unknown",
                                       "timeofday": "night" if cond == "night" else "day"}})
    return recs


# ----------------------------------------------------------------------------- detection
def bdd_det(bdd_root, split: str) -> list[dict]:
    root = bdd.find_root(bdd_root)
    label_file = next((root / c.format(split=split) for c in bdd._LABEL_CANDIDATES if (root / c.format(split=split)).exists()), None)
    if label_file is None:
        raise FileNotFoundError(f"BDD100K detection labels for '{split}' not found under {root}")
    with open(label_file) as f:
        items = json.load(f)
    recs = []
    for it in items:
        img = root / "images/100k" / split / it["name"]
        boxes, labels = [], []
        for lab in it.get("labels") or []:
            cls = BDD_DET_MAP.get(lab.get("category"))
            b = lab.get("box2d")
            if cls is None or b is None:
                continue
            boxes.append([b["x1"], b["y1"], b["x2"], b["y2"]])
            labels.append(det_class_index(cls))
        a = it.get("attributes") or {}
        recs.append({"image": str(img), "width": 1280, "height": 720, "boxes": boxes, "labels": labels,
                     "dataset": "bdd100k",
                     "attrs": {"weather": normalize_weather(a.get("weather")),
                               "timeofday": normalize_timeofday(a.get("timeofday")),
                               "scene_type": a.get("scene", "unknown"), "location": "us"}})
    return recs


def nuimages_det(nuim_root, version: str = "v1.0-val") -> list[dict]:
    root = Path(nuim_root)
    tdir = root / version

    def load(name):
        with open(tdir / f"{name}.json") as f:
            return json.load(f)

    cats = {c["token"]: c["name"] for c in load("category")}
    logs = {l["token"]: l for l in load("log")}
    samples = {s["token"]: s for s in load("sample")}
    anns = defaultdict(list)
    for a in load("object_ann"):
        anns[a["sample_data_token"]].append(a)
    recs = []
    for sd in load("sample_data"):
        if not sd["is_key_frame"] or sd["fileformat"] not in ("jpg", "jpeg"):
            continue
        boxes, labels = [], []
        for a in anns.get(sd["token"], []):
            cls = map_nuimages_category(cats[a["category_token"]])
            if cls is not None:
                boxes.append([float(v) for v in a["bbox"]])
                labels.append(det_class_index(cls))
        location = logs[samples[sd["sample_token"]]["log_token"]]["location"]
        recs.append({"image": str(root / sd["filename"]), "width": sd["width"], "height": sd["height"],
                     "boxes": boxes, "labels": labels, "dataset": "nuimages",
                     "attrs": {"location": location, "region": location.split("-")[0],
                               "camera": sd["filename"].split("/")[1] if "/" in sd["filename"] else "unknown"}})
    return recs


def waymo_det(waymo_root, out_image_dir, split: str = "validation", cameras=(1,), target_hz: float = 1.0,
              short_side: int = 720, max_segments: int | None = None) -> list[dict]:
    """Extracts annotated camera frames to JPEG (resized; boxes rescaled) and returns detection records."""
    import pyarrow.parquet as pq

    from .sources import waymo as wm

    split_dir, out_image_dir = Path(waymo_root) / split, Path(out_image_dir)
    cols = [wm.COL_TS, wm.COL_CAM, "[CameraBoxComponent].box.center.x", "[CameraBoxComponent].box.center.y",
            "[CameraBoxComponent].box.size.x", "[CameraBoxComponent].box.size.y", "[CameraBoxComponent].type"]
    box_files = sorted((split_dir / "camera_box").glob("*.parquet"))[:max_segments]
    recs = []
    for bf in tqdm(box_files, desc=f"waymo det/{split}"):
        seg = bf.stem
        meta = wm.read_segment_stats(split_dir, seg)
        t = pq.read_table(bf, columns=cols).to_pydict()
        per_frame = defaultdict(list)
        for ts, cam, cx, cy, sx, sy, typ in zip(*[t[c] for c in cols]):
            if cam in cameras and typ in WAYMO_DET_MAP:
                per_frame[(ts, cam)].append(([cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2], det_class_index(WAYMO_DET_MAP[typ])))
        img_file = split_dir / "camera_image" / f"{seg}.parquet"
        if not img_file.exists():
            continue
        all_ts = pq.read_table(img_file, columns=[wm.COL_TS]).column(wm.COL_TS).to_pylist()
        keep = wm.select_timestamps(all_ts, target_hz)
        for ts, cam, jpeg in wm.iter_segment_images(img_file, set(cameras), keep):
            img = Image.open(io.BytesIO(jpeg)).convert("RGB")
            w, h = img.size
            s = min(1.0, short_side / min(w, h))
            if s < 1.0:
                img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
            out = out_image_dir / seg / f"{wm.CAMERA_NAMES.get(cam, cam)}_{ts}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            img.save(out, quality=92)
            objs = per_frame.get((ts, cam), [])
            recs.append({"image": str(out), "width": img.size[0], "height": img.size[1],
                         "boxes": [[v * s for v in b] for b, _ in objs], "labels": [l for _, l in objs],
                         "dataset": "waymo", "attrs": {**meta, "camera": wm.CAMERA_NAMES.get(cam, str(cam))}})
    return recs


# ----------------------------------------------------------------------------- registry
def write_records(records: list[dict], out_path) -> None:
    write_jsonl(records, out_path)
    print(f"wrote {len(records)} records -> {out_path}")


def write_exclude_ids(records: list[dict], out_path) -> None:
    """Sequence ids (image stems) of evaluation images, used by pools to avoid pretraining on them."""
    ids = sorted({Path(r["image"]).stem for r in records})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(ids) + "\n")
    print(f"wrote {len(ids)} exclusion ids -> {out_path}")

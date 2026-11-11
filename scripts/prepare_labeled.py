"""Convert annotated datasets into the unified JSONL format under paths.labeled_root.

  python scripts/prepare_labeled.py bdd-seg bdd-det acdc cityscapes nuimages waymo-det
Also writes labeled_root/bdd_eval_ids.txt (ids of BDD evaluation images) which pools use to avoid leakage.
"""

import argparse
from pathlib import Path

from xdssl.data import labeled_sources as ls
from xdssl.utils.config import load_paths, resolve_path
from xdssl.utils.misc import read_jsonl

TARGETS = ["bdd-seg", "bdd-det", "cityscapes", "acdc", "nuimages", "waymo-det"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+", choices=TARGETS + ["all"])
    ap.add_argument("--waymo-max-segments", type=int, default=None)
    ap.add_argument("--waymo-hz", type=float, default=1.0)
    ap.add_argument("--paths", default=None)
    a = ap.parse_args()
    p = load_paths(a.paths)
    out = resolve_path(p.labeled_root)
    targets = TARGETS if "all" in a.targets else a.targets

    for t in targets:
        try:
            if t == "bdd-seg":
                ls.write_records(ls.bdd_seg(p.raw.bdd100k, "train"), out / "seg/bdd_seg_train.jsonl")
                ls.write_records(ls.bdd_seg(p.raw.bdd100k, "val"), out / "seg/bdd_seg_val.jsonl")
            elif t == "bdd-det":
                ls.write_records(ls.bdd_det(p.raw.bdd100k, "train"), out / "det/bdd_det_train.jsonl")
                ls.write_records(ls.bdd_det(p.raw.bdd100k, "val"), out / "det/bdd_det_val.jsonl")
            elif t == "cityscapes":
                ls.write_records(ls.cityscapes_seg(p.raw.cityscapes, "val"), out / "seg/cityscapes_val.jsonl")
            elif t == "acdc":
                ls.write_records(ls.acdc_seg(p.raw.acdc, "val"), out / "seg/acdc_val.jsonl")
            elif t == "nuimages":
                ls.write_records(ls.nuimages_det(p.raw.nuimages, "v1.0-val"), out / "det/nuimages_val.jsonl")
            elif t == "waymo-det":
                ls.write_records(ls.waymo_det(p.raw.waymo_v2, out / "waymo_images", "validation", target_hz=a.waymo_hz,
                                              max_segments=a.waymo_max_segments), out / "det/waymo_val.jsonl")
        except FileNotFoundError as e:
            print(f"[{t}] skipped: {e}")

    eval_files = [out / "seg/bdd_seg_val.jsonl", out / "det/bdd_det_val.jsonl"]
    recs = [r for f in eval_files if Path(f).exists() for r in read_jsonl(f)]
    if recs:
        ls.write_exclude_ids(recs, out / "bdd_eval_ids.txt")


if __name__ == "__main__":
    main()

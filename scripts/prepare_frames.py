"""Extract downsized unlabeled frames + manifest for one source dataset.

Examples:
  python scripts/prepare_frames.py bdd100k-images
  python scripts/prepare_frames.py bdd100k-videos --fps 1
  python scripts/prepare_frames.py nuscenes --version v1.0-trainval --hz 2
  python scripts/prepare_frames.py waymo --splits training --hz 1 --delete-source
Raw locations come from configs/paths.yaml (raw.*); output goes to paths.frames_root/<dataset>/.
Runs are incremental: re-running after downloading more archives only processes new frames.
"""

import argparse

from xdssl.data.sources import bdd100k, nuscenes, waymo
from xdssl.utils.config import load_paths, resolve_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["bdd100k-images", "bdd100k-videos", "nuscenes", "waymo"])
    ap.add_argument("--short-side", type=int, default=256, help="frames are stored with this short side (px)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--fps", type=float, default=1.0, help="BDD video sampling rate")
    ap.add_argument("--hz", type=float, default=None, help="nuScenes / Waymo sampling rate")
    ap.add_argument("--version", default="v1.0-trainval", help="nuScenes version (v1.0-mini for testing)")
    ap.add_argument("--cameras", nargs="*", default=None)
    ap.add_argument("--delete-source", action="store_true", help="Waymo: delete each parquet after extraction")
    ap.add_argument("--paths", default=None)
    a = ap.parse_args()

    p = load_paths(a.paths)
    frames_root = resolve_path(p.frames_root)
    if a.source == "bdd100k-images":
        df = bdd100k.index_bdd_images(p.raw.bdd100k, frames_root, a.splits or ("train", "val", "test"), a.short_side, a.workers)
    elif a.source == "bdd100k-videos":
        df = bdd100k.index_bdd_videos(p.raw.bdd100k, frames_root, a.splits or ("train",), a.fps, a.short_side, a.workers)
    elif a.source == "nuscenes":
        df = nuscenes.index_nuscenes(p.raw.nuscenes, frames_root, a.version, tuple(a.cameras or nuscenes.ALL_CAMERAS),
                                     a.hz or 2.0, a.short_side, a.workers)
    else:
        df = waymo.index_waymo(p.raw.waymo_v2, frames_root, a.splits or ("training",),
                               tuple(int(c) for c in a.cameras) if a.cameras else (1, 2, 3, 4, 5), a.hz or 1.0,
                               a.short_side, a.delete_source)
    print(f"manifest now has {len(df)} frames")
    if len(df):
        print(df.groupby(["split", "timeofday", "weather"]).size().to_string())


if __name__ == "__main__":
    main()

"""Label taxonomies shared across datasets so that cross-dataset evaluation is apples-to-apples."""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------- segmentation
# Cityscapes 19-class "trainId" taxonomy. BDD100K sem_seg and ACDC ship labels in exactly this format.
SEG_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation",
    "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]
SEG_IGNORE = 255

SEG_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
    [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32],
], dtype=np.uint8)

# Cityscapes raw labelId -> trainId (for gtFine_labelIds.png files)
_CS_ID_TO_TRAIN = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                   26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
CITYSCAPES_ID_TO_TRAINID = np.full(256, SEG_IGNORE, dtype=np.uint8)
for _k, _v in _CS_ID_TO_TRAIN.items():
    CITYSCAPES_ID_TO_TRAINID[_k] = _v

LABEL_LUTS = {"trainid": None, "cityscapes_id": CITYSCAPES_ID_TO_TRAINID}

# ----------------------------------------------------------------------------- detection
# Harmonized 2-class set: the only categories with compatible box definitions across BDD100K, nuImages and
# Waymo. (Cyclists are excluded because BDD boxes the rider only while Waymo boxes rider+bike.)
DET_CLASSES = ["vehicle", "pedestrian"]

BDD_DET_MAP = {"car": "vehicle", "truck": "vehicle", "bus": "vehicle", "pedestrian": "pedestrian", "person": "pedestrian"}

NUIMAGES_DET_PREFIX_MAP = [
    ("vehicle.car", "vehicle"), ("vehicle.truck", "vehicle"), ("vehicle.bus", "vehicle"),
    ("vehicle.emergency", "vehicle"), ("vehicle.construction", "vehicle"), ("vehicle.trailer", "vehicle"),
    ("human.pedestrian", "pedestrian"),
]

WAYMO_DET_MAP = {1: "vehicle", 2: "pedestrian"}  # waymo box types: 1 vehicle, 2 pedestrian, 3 sign, 4 cyclist


def det_class_index(name: str) -> int:
    """0-based index into DET_CLASSES (the detector adds +1 for background internally)."""
    return DET_CLASSES.index(name)


def map_nuimages_category(category: str) -> str | None:
    for prefix, target in NUIMAGES_DET_PREFIX_MAP:
        if category.startswith(prefix):
            return target
    return None

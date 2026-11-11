"""Shared vocabulary for the unified frame index.

Every unlabeled source dataset is converted into a `manifest.parquet` with FRAME_COLUMNS, so pools can be
defined as filters/samples over one table regardless of where the frames came from.
"""

from __future__ import annotations

FRAME_COLUMNS = [
    "frame_id",      # globally unique: "{dataset}/{sequence_id}/{camera}/{timestamp}"
    "dataset",       # bdd100k | nuscenes | waymo | ...
    "split",         # train | val | test  (source dataset's own split)
    "sequence_id",   # video / scene / segment id; frames sharing it are temporally correlated
    "camera",        # front | front_left | ...
    "timestamp",     # int, milliseconds from sequence start (or absolute; only ordering within a sequence matters)
    "path",          # relative to paths.frames_root
    "weather",       # see WEATHER
    "timeofday",     # see TIMEOFDAY
    "location",      # free-form city / region string, e.g. "us", "boston-seaport", "location_sf"
    "scene_type",    # free-form, e.g. "city street", "highway", or "unknown"
]

WEATHER = ("clear", "overcast", "partly_cloudy", "rain", "snow", "fog", "unknown")
TIMEOFDAY = ("day", "night", "dawn_dusk", "unknown")

_WEATHER_SYNONYMS = {
    "clear": "clear", "sunny": "clear",
    "overcast": "overcast", "cloudy": "overcast",
    "partly cloudy": "partly_cloudy", "partly_cloudy": "partly_cloudy",
    "rain": "rain", "rainy": "rain",
    "snow": "snow", "snowy": "snow",
    "fog": "fog", "foggy": "fog",
}
_TIMEOFDAY_SYNONYMS = {
    "day": "day", "daytime": "day",
    "night": "night",
    "dawn/dusk": "dawn_dusk", "dawn": "dawn_dusk", "dusk": "dawn_dusk", "dawn_dusk": "dawn_dusk",
}


def normalize_weather(value) -> str:
    return _WEATHER_SYNONYMS.get(str(value).strip().lower(), "unknown") if value else "unknown"


def normalize_timeofday(value) -> str:
    return _TIMEOFDAY_SYNONYMS.get(str(value).strip().lower(), "unknown") if value else "unknown"


def make_frame_id(dataset: str, sequence_id: str, camera: str, timestamp: int) -> str:
    return f"{dataset}/{sequence_id}/{camera}/{int(timestamp)}"


def make_frame_relpath(dataset: str, sequence_id: str, camera: str, timestamp: int) -> str:
    return f"{dataset}/{sequence_id}/{camera}_{int(timestamp)}.jpg"

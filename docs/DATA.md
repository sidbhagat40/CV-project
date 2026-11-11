# Data preparation

All datasets need a free registration and acceptance of their licenses (non-commercial research use). Raw
downloads go wherever you like; point `configs/paths.yaml` (`raw.*`) at them. The pipeline writes three
derived trees, all small compared with the raw data:

- `frames_root/<dataset>/...jpg` + `manifest.parquet`: unlabeled frames resized to a 256 px short side
  (~15-25 KB each; 100K frames is about 2 GB).
- `labeled_root/{seg,det}/*.jsonl`: unified annotation records pointing at the raw labeled images.
- `pools_root/<pool>/pool.parquet`: sampled pretraining pools.

There is about 200 GB free on D:, which is not enough for complete nuScenes (~300 GB with all sensors) or Waymo
(well over 1 TB). Both indexers therefore work **incrementally**: download a chunk, extract frames, delete the
chunk, and repeat. Re-running only adds new frames to the manifest.

## Minimum viable setup (start here)

| what | download | used for |
|------|----------|----------|
| BDD100K 100K images + det labels | `100k_images_{train,val,test}.zip`, `bdd100k_det_20_labels_trainval.zip` | pretraining frames, detection fine-tuning + BDD condition slices |
| BDD100K 10K images + sem_seg labels | `10k_images_*.zip`, `bdd100k_sem_seg_labels_trainval.zip` | segmentation fine-tuning + val |
| ACDC | `rgb_anon_trainvaltest.zip`, `gt_trainval.zip` | fog/night/rain/snow segmentation targets |

That is enough for the BDD-only parts of E1 (`bdd_clearday` vs `bdd_mixed`) with segmentation on ACDC and
detection on BDD night/rain/snow.

```powershell
python scripts/prepare_frames.py bdd100k-images          # ~100K frames
python scripts/prepare_labeled.py bdd-seg bdd-det acdc   # also writes labeled_root/bdd_eval_ids.txt
python scripts/build_pool.py --spec configs/pools/bdd_clearday.yaml
python scripts/build_pool.py --spec configs/pools/bdd_mixed.yaml
```

## Per-dataset notes

### BDD100K (https://dl.cv.ethz.ch/bdd100k/data/ ; formerly bdd-data.berkeley.edu)
Expected layout: `bdd100k/images/100k/{train,val,test}`, `bdd100k/images/10k/{train,val}`,
`bdd100k/labels/det_20/det_{train,val}.json`, `bdd100k/labels/sem_seg/masks/{train,val}/*.png`.
The legacy `bdd100k_labels_images_{split}.json` and `seg/labels/*_train_id.png` layouts are also detected.
Weather, time of day and scene attributes come from the detection label files, so download those even if you only
pretrain. Test-split images have no attributes (`unknown`); they can still go into pools whose filters allow it.
Videos (`videos/{train,val}/*.mov`, ~1.8 TB in total) are optional: `prepare_frames.py bdd100k-videos --fps 1`
gives 40 frames per video and processes whatever subset of archives is present.

### nuScenes (https://www.nuscenes.org/nuscenes#download)
Needs `v1.0-trainval_meta` plus any number of the `v1.0-trainval{01..10}_blobs` (each blob holds ~85 scenes'
sensor files). Only `samples/CAM_*` and `sweeps/CAM_*` are read, so you can delete lidar/radar folders after
unpacking a blob. Try `--version v1.0-mini` first (4 GB). If `nuscenes-devkit` is installed the official
train/val split is used; otherwise a deterministic hash split by scene name is used. That is fine, because
nuScenes itself is never an evaluation target. Rain and night come from the scene description text.

### nuImages (same site, "nuImages")
Needs `nuimages-v1.0-all-metadata` and the val samples (`samples/CAM_*`). Only key frames have boxes.
Categories are mapped to vehicle and pedestrian (see `taxonomy.NUIMAGES_DET_PREFIX_MAP`).

### Waymo Open Dataset, Perception v2 (https://waymo.com/open/download/)
Uses the **v2 Parquet** release, so no TensorFlow is needed. With `gcloud` authenticated, download per-segment
files, for example:
```powershell
gsutil -m cp "gs://waymo_open_dataset_v_2_0_1/training/camera_image/*.parquet" D:/datasets/waymo_v2/training/camera_image/
gsutil -m cp "gs://waymo_open_dataset_v_2_0_1/training/stats/*.parquet"        D:/datasets/waymo_v2/training/stats/
```
(`stats` is tiny; `camera_image` is large, so copy a few hundred segments at a time and run
`prepare_frames.py waymo --delete-source`.) For evaluation, `validation/camera_image`, `validation/camera_box` and
`validation/stats` are needed. Check the version folder name in the bucket. Column names are constants at the
top of `src/xdssl/data/sources/waymo.py`; verify them on one file:
`python -c "import pyarrow.parquet as pq; print(pq.read_schema('<file>.parquet'))"`.

### Cityscapes (https://www.cityscapes-dataset.com/downloads/)
`leftImg8bit_trainvaltest.zip` (11 GB) + `gtFine_trainvaltest.zip`. Raw `*_labelIds.png` are mapped to trainIds
on the fly; `*_labelTrainIds.png` from cityscapesscripts are used if present.

### ACDC (https://acdc.vision.ee.ethz.ch/download)
`rgb_anon_trainvaltest.zip` + `gt_trainval.zip`. The val split (~400 images, ~100 per condition) is used as the
target, because test labels are withheld.

## Evaluation domains

Domains are defined in `configs/eval/{seg,det}_targets.yaml` as `(jsonl file, attrs filter)`. Adding a new
target means writing a converter that emits the unified record format (see `labeled_sources.py`) and adding one
line to the YAML. Missing files are skipped with a warning.

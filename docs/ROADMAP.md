# Roadmap

Each item is sized for one focused session. "Done when" is the acceptance criterion. Run `python -m pytest -q`
after every change, and add a test for any new logic. Read `AGENTS.md` first.

Status legend: `[ ]` todo, `[x]` done.

## Done (v0.1 scaffold)
- [x] Unified frame index + incremental indexers for BDD100K (images, videos), nuScenes, Waymo v2
- [x] Pool builder (iso-size, weights, filters, condition balancing, per-sequence cap, leakage exclusion)
- [x] ViT-S/16 with timm-compatible weights; MAE; MoCo v3 with temporal positives
- [x] Segmentation (Semantic FPN, linear probe) and detection (Faster R-CNN) fine-tuning with layer-wise LR decay
- [x] Multi-domain evaluation (mIoU + bootstrap CI, COCO AP per class), robustness summaries, effective robustness
- [x] Pool diversity (metadata entropy, Vendi, Fréchet distance to targets)
- [x] Experiment runner (E1-E5), analysis tables and figures, qualitative demo
- [x] Synthetic data + unit and end-to-end tests; GPU benchmark

## P0: validate on real data (blocks all experiments)
1. [ ] **BDD100K end-to-end.** Run `prepare_frames.py bdd100k-images` and `prepare_labeled.py bdd-seg bdd-det`
   on the real files. Done when manifests and JSONL exist, the attribute counts printed by `prepare_frames` look
   right (about 53% clear, 53% daytime), and `bdd_eval_ids.txt` is non-empty.
2. [ ] **ACDC + Cityscapes converters.** Done when `acdc_val.jsonl` has ~400 records (about 100 per condition)
   and a mask spot-check shows values in {0..18, 255}.
3. [ ] **nuScenes indexer on v1.0-mini.** Done when the manifest has frames from 10 scenes and 6 cameras, and
   night/rain tags appear for the scenes whose descriptions mention them.
4. [ ] **Waymo v2 schema check.** On one downloaded segment, print `pq.read_schema` for `camera_image`,
   `camera_box` and `stats`, then fix the column constants in `sources/waymo.py` if they differ. Done when
   `prepare_frames.py waymo` and `prepare_labeled.py waymo-det --waymo-max-segments 2` both succeed and drawn
   boxes line up with objects (add a `scripts/visualize_records.py` for this; it is generally useful).
5. [ ] **Data-loader throughput.** Pretrain for 200 steps on a real pool and compare step time in `log.jsonl`
   with `scripts/benchmark.py` (~300 ms/step). If loading is >20% slower, try more `num_workers`, check that
   frames are 256 px, or pre-decode the pool into a uint8 `.npy` memmap of fixed-size 256x256 center crops
   (add `data.format: memmap`). Done when loader overhead is <20%.
6. [ ] **First real signal.** Build `bdd_clearday` and `bdd_mixed`, pretrain both with default settings, and run
   `seg_bdd_linear` for random / imagenet / both pools. Done when `scripts/analyze.py` produces the tables.

## P1: quality and rigor
7. [ ] **MAE reconstruction visualizer** (`scripts/visualize_mae.py`): grid of input / masked / reconstruction
   from a `final.pt`. Sanity check for pretraining, and makes a good report figure.
8. [ ] **Periodic ID-only validation** during fine-tuning (loss curve + ID metric every N steps, logged only;
   no checkpoint selection, and never on OOD targets).
9. [ ] **Detection confidence intervals:** per-image bootstrap of AP (cache matched detections from COCOeval once,
   then resample images). Done when `results.json` has `AP_ci95` for detection domains.
10. [ ] **Segmentation sliding-window evaluation** at full resolution as an option (`eval.mode: slide`).
11. [ ] **TensorBoard / W&B logging** behind a config flag (`logging.backend`).
12. [ ] **Paired significance test** between two inits over seeds and domains (paired bootstrap or Wilcoxon) in
    `analysis.py`, reported in the summary tables.
13. [ ] **Pool-construction report:** `scripts/describe_pools.py` renders a bar chart of each pool's
    dataset / weather / time-of-day composition from `summary.json` (report figure).

## P2: research extensions
14. [ ] **E6 continued pretraining:** experiment YAML with `pretrain.overrides: [model.init_from=imagenet]` and a tag.
15. [ ] **More targets:** Dark Zurich val (night segmentation), Foggy Cityscapes (synthetic fog levels, which
    gives a robustness curve vs fog density), Waymo 2D panoptic mapped to Cityscapes-19 classes.
16. [ ] **Exchange-rate computation** in `analysis.py`: interpolate each baseline's OOD-vs-label-fraction curve
    (log x-axis) and report the label multiple needed to match each SSL init.
17. [ ] **Diversity-controlled sampling:** pool builder option that greedily maximizes the Vendi score (or
    k-center coverage) in embedding space at fixed size. This separates "diversity" from "which datasets".
18. [ ] **Temporal SSL:** VideoMAE-style tube masking on short clips from nuScenes/Waymo/BDD videos.
19. [ ] **Interactive demo:** Gradio app that loads two `model.pt` files and shows side-by-side predictions on
    uploaded images or on random ACDC fog/night samples.
20. [ ] **ConvNet backbone** (ResNet-50 via torchvision) to check whether conclusions hold beyond ViTs.

## P3: write-up
21. [ ] `report/` LaTeX skeleton (Intro, Related Work, Method, Setup, Results, Analysis, Limitations), with
    figures pulled from `outputs/`.
22. [ ] Related-work notes: MAE, MoCo v3, DINOv2 (data curation for diversity), Taori et al. 2020 (effective
    robustness), ACDC, BDD100K, domain generalization for segmentation (e.g. RobustNet, DAFormer), driving SSL.

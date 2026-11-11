# Research plan

## 1. Claim under test

> At a **fixed unlabeled frame budget and fixed pretraining compute**, pretraining on a pool that is more
> diverse in geography, weather and time of day yields driving-perception models that generalize better to unseen
> conditions and datasets. The gain is larger than what you get from adding more labeled data from a single domain.

The proposal mentions "tens of millions of frames". On one laptop GPU (RTX 4060, 8 GB) each pretraining run can
*see* ~8M samples (30k steps x 256), but it cannot use millions of *unique* frames for every pool in the grid.
So the study is framed as **diversity at a controlled budget**, which is also the scientifically cleaner question:
without the control, "more diverse" is confounded with "more data". The pipeline scales unchanged (raise
`size` in `configs/pools/*.yaml`, extract BDD video frames) if more compute becomes available.

## 2. Hypotheses

| id | hypothesis | experiment |
|----|------------|------------|
| H1 | Same size, more diverse pool -> higher mean/worst OOD score and positive effective robustness | E1 |
| H2 | H1 still holds when the target dataset was **never** in the pretraining pool | E2 (leave-one-dataset-out) |
| H3 | A diverse 18K pool beats a 3x larger single-source pool OOD | E3 |
| H4 | Diverse pretraining is worth k x more single-domain labels OOD ("exchange rate") | E4 |
| H5 | The effect is not specific to MAE (also appears with MoCo v3 + temporal positives) | E5 |

## 3. Variables

**Independent:** pretraining-pool composition (configs/pools/*.yaml).

**Held fixed across compared runs:**
- number of frames in the pool, `size` (enforced exactly, otherwise `build_pool` raises an error)
- pretraining steps, batch size, LR schedule, augmentations, architecture (ViT-S/16), input size (224)
- frame preprocessing (all sources resized to a 256 px short side)
- fine-tuning data: same JSONL, same `train_filter`, same `label_fraction` subset (`subset_seed` fixed), same steps
- evaluation protocol: same target files, same resolution, final checkpoint (no target-based model selection)

**Dependent (per target domain):** mIoU (segmentation) / COCO AP (detection), and from those:
- *ID*: score on the source domain's own validation set
- *mean OOD* and *worst OOD*: mean and minimum over shifted domains
- *relative robustness*: mean OOD / ID
- *effective robustness* (Taori et al. 2020): OOD minus the OOD predicted from ID by a logit-linear fit over the
  baseline models (random/ImageNet init at several label fractions). This is the key metric: it separates
  "more robust" from "simply better".

**Pool diversity measures (x-axis of plots):** normalized metadata entropy (dataset, location, weather,
time of day) and the Vendi score of frame embeddings from a fixed ImageNet ViT-S. We also compute the Fréchet distance
from each pool to each target, to check whether gains track *proximity to the target* rather than diversity.

## 4. Data roles

| dataset | unlabeled pretraining | labeled fine-tuning | evaluation target |
|---------|----------------------|---------------------|-------------------|
| BDD100K (US; all weathers, day/night) | 100K keyframes (+ optional video frames) | sem_seg 7K; det 70K (clear-day subset ~20K) | BDD val, sliced by weather / time of day |
| nuScenes (Boston, Singapore; rain/night tags) | 6 cameras @ 2 Hz | - | via nuImages |
| nuImages | - | - | 2D boxes, val, by city |
| Waymo v2 (Phoenix, SF, MTV; day/night/dawn) | 5 cameras @ 1 Hz | - | 2D boxes, val, by time of day |
| Cityscapes (Germany, clear day) | - | - | seg val (new city) |
| ACDC (Switzerland; fog/night/rain/snow) | - | - | seg val per condition |

Taxonomies are harmonized:
- **Segmentation** uses the Cityscapes 19-class trainIds, native in BDD and ACDC.
- **Detection** uses `vehicle` (car/truck/bus) and `pedestrian`, the only classes with compatible box definitions
  across BDD, nuImages and Waymo. Cyclists are excluded because BDD boxes the rider while Waymo boxes rider and bike.

## 5. Experiments

All pools contain 18K frames unless the name says otherwise. "seeds" refers to fine-tuning seeds. Pretraining uses one
seed per pool; if variance between pools turns out to be small, add pretraining seeds for the top pools.

| exp | pools / inits | protocols | runs |
|-----|---------------|-----------|------|
| E0 baselines | random, ImageNet-supervised ViT-S | all | (inside E1/E4) |
| E1 diversity | bdd_clearday (narrow), bdd_mixed (condition-diverse), bdd_nusc_waymo (multi-source) | seg linear probe, seg full FT, det full FT | 3 pretrains, 45 fine-tunes |
| E2 leave-one-out | bdd_nusc, bdd_waymo, bdd_nusc_waymo | det | 3 pretrains, 9 fine-tunes |
| E3 scale vs diversity | all3 @ 6K/18K/54K, bdd_mixed @ 18K/54K | seg linear, det | 5 pretrains, 20 fine-tunes |
| E4 label scaling | random, ImageNet, narrow, diverse x {5,10,25,50,100}% labels | det (10k steps) | 40 fine-tunes |
| E5 method | MoCo v3 (temporal positives) on narrow vs diverse | seg linear, det | 2 pretrains, 8 fine-tunes |

**Recommended order:**
1. Smoke test.
2. BDD only: build bdd_clearday and bdd_mixed, run linear probes. This is the cheapest first signal for H1.
3. Add nuScenes and Waymo, then run the full E1.
4. E4, for the headline exchange-rate figure.
5. E2, E3 and E5 as time permits.

## 6. Confounds and how they are handled

| confound | handling |
|----------|----------|
| More data vs more diverse data | iso-size pools + iso-step pretraining; E3 varies size explicitly |
| Redundant frames from video (nuScenes/Waymo have many frames per scene, BDD keyframes 1 per video) | `summary.json` reports `n_sequences`; Vendi score is embedding-based and reflects redundancy; optional `max_per_sequence` cap |
| Target data exposure: pooling nuScenes means nuImages targets are "seen unlabeled" | E2 leave-one-dataset-out pools; report seen and unseen targets separately |
| Leakage of evaluation images into pretraining | pools use only `train` splits; `bdd_eval_ids.txt` removes BDD val images (the 10K seg set overlaps the 100K videos) |
| Resolution / camera / aspect ratio differences between sources | every frame resized to a 256 px short side; random-resized crops to 224 |
| Diversity gains being just better ID accuracy | effective robustness relative to the baseline fit |
| Fine-tuning noise | 3 seeds (E1); bootstrap 95% CIs over images for mIoU |
| Tiny target slices (BDD val fog has only a few dozen images) | report n per domain; treat as qualitative; rely on ACDC fog for quantitative fog claims |
| Hyper-parameter selection on targets | none: fixed recipes, final checkpoint, ID-only tuning if any |
| Augmentation as "free diversity" (MoCo colour jitter mimics lighting change) | identical augmentations for all pools within a method; E5 checks method dependence |

## 7. Compute budget (measured on this machine, RTX 4060 Laptop 8 GB, bf16)

| job | step time | peak mem | default length | wall-clock (compute only) |
|-----|-----------|----------|----------------|---------------------------|
| MAE ViT-S 224, batch 256 | 300 ms | 5.3 GB | 30k steps | ~2.5 h |
| Seg FPN ViT-S 512 crop, batch 8 | 288 ms | 3.4 GB | 16k steps | ~1.3 h |
| Faster R-CNN ViT-S 576x1024, batch 4 | 406 ms | 3.4 GB | 20k steps | ~2.3 h |
| Seg linear probe | - | - | 4k steps | ~0.3 h |

Data loading on Windows may be the real bottleneck for MAE (it needs ~850 decoded JPEGs/s). Check `log.jsonl`
step times on real data, and see ROADMAP item P0-5. Re-measure with `python scripts/benchmark.py`.

Approximate GPU-hours: E1 ~70 (one seed: ~25), E2 ~30, E3 ~35, E4 ~50, E5 ~15.

## 8. Expected outcomes and risks

- With only ~18K unique frames, SSL from scratch will probably trail ImageNet-supervised init in *absolute* numbers.
  That does not affect the claim, which is about the *ordering of pools* and effective robustness. For a practical
  variant, set `model.init_from=imagenet` in the pretraining config (continued pretraining). Report it as a second
  regime, not mixed into the main grid.
- If differences between pools fall within seed noise at 18K, the first lever is a larger `size` (BDD video frames
  make 54K+ easy). The second lever is longer pretraining.
- The strongest result would be: positive effective robustness for bdd_nusc_waymo that persists in leave-one-out
  (H2), plus an exchange rate of 2x or more in E4.

## 9. Deliverables

- Table: per-domain scores (mean +- std) for each init and protocol (`outputs/<exp>/*_summary.csv`)
- Fig. 1: ID vs mean-OOD scatter with the baseline fit (effective robustness)
- Fig. 2: mean OOD vs pool diversity (Vendi / entropy)
- Fig. 3: label-scaling curves, ID and OOD (exchange rate)
- Fig. 4: qualitative side-by-side under fog/night (`scripts/demo.py`)

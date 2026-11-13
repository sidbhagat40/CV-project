# Viva preparation: Cross-Domain Self-Supervised Pretraining for Robust Driving Perception

Team: Siddharth Bhagat (B23MT1040), Kunal Jangid (B23MT1026), Abhishek Yadav (B23MT1003)

---

## 1. The project in 60 seconds

**Question.** Driving models fail on conditions they were not trained on (night, rain, snow). If we pretrain on unlabeled
driving images *without labels*, does it help more when those images are **diverse** (many weathers and times of day) than
when they are all similar (clear daytime)?

**What we did.** We took BDD100K driving images and built two pretraining sets of the same size (1,200 images):
one *narrow* (clear daytime only) and one *diverse* (balanced across weather and time of day). We pretrained a Vision Transformer
(ViT-S/16) on each with **MAE** (masked autoencoder, a self-supervised method), for exactly the same number of steps. Then we
fine-tuned each model on **labeled clear-daytime images only** for vehicle and pedestrian detection, and tested on clear day (the
"in-domain" case) and on **held-out night, dawn/dusk, rain and snow** images (the "out-of-domain" case). We also compared with random and ImageNet-supervised starting points.

**What we found.** The diverse pool was slightly but consistently better (mean out-of-domain AP 6.3 vs 6.1; better in 9 of 10 comparisons).
The gain is small and only suggestive, because each pool was pretrained once. ImageNet-supervised initialization was far better than both
(13.2 AP), which is expected with only 1,200 pretraining images.

---

## 2. Why we did it (motivation)

- Self-driving systems meet conditions that no labeled dataset covers fully. Labels are costly; unlabeled video is cheap.
- Self-supervised learning (SSL) can use unlabeled data. But **what kind of unlabeled data is best** is an open question.
- Comparing a big diverse dataset with a small narrow one mixes up "more diverse" with "more data". So we **fixed the size and the compute** and changed
  only the diversity. That is the key idea of the experimental design.

---

## 3. Basic concepts (know these cold)

| Term | One-line explanation |
|---|---|
| **Self-supervised learning** | Learn useful features from unlabeled data by solving a task made from the data itself (e.g., predict hidden parts of an image). |
| **Pretraining vs fine-tuning** | Pretraining learns general features (no labels). Fine-tuning adapts the model to a task with labeled data. |
| **Vision Transformer (ViT)** | Splits an image into patches (here 16x16 pixels), treats them as tokens, and processes them with transformer attention. ViT-S = small version (about 22M parameters). |
| **MAE (Masked Autoencoder)** | Hide 75% of patches, feed only the visible 25% to the encoder, and train a small decoder to reconstruct the hidden pixels. Cheap because the encoder sees few tokens. |
| **MoCo v3** | A contrastive SSL method (pull two views of the same image together, push others apart). In the codebase as an ablation; **not run** in our final experiment. |
| **Distribution shift / domain shift** | Test data differs from training data (here: weather and lighting). |
| **In-domain (ID) vs out-of-domain (OOD)** | ID: same conditions as the training labels (clear day). OOD: shifted (night, rain, snow, dawn/dusk). |
| **Robustness** | How well performance holds under shift. |
| **Effective robustness** (Taori et al. 2020) | OOD performance *beyond what ID accuracy would predict*. Separates "more robust" from "simply more accurate". |
| **Object detection** | Predict boxes and classes. We used Faster R-CNN with the ViT as the backbone. |
| **AP (Average Precision)** | Standard COCO detection metric: area under the precision-recall curve, averaged over IoU thresholds 0.5 to 0.95. Higher is better. |
| **IoU** | Overlap of a predicted and a true box: intersection / union. |
| **Layer-wise learning-rate decay** | Earlier layers get smaller learning rates during fine-tuning, so pretrained features are not destroyed. |
| **Wilcoxon signed-rank test** | Non-parametric paired test, used to check whether the diverse-minus-narrow differences are consistently positive. |
| **Bootstrap CI** | Resample the paired differences many times to get a confidence interval for the mean difference. |
| **Vendi score / entropy** | Measures of diversity of a dataset (the code supports them; we reported metadata entropy of the pools). |
| **Fréchet distance** | Distance between feature distributions of two image sets; used to check closeness of a pool to the test data. The code has it, but we did **not** run it. |

---

## 4. The setup in detail

### Data
- **Source:** BDD100K, the 10,000 official *validation* images, from a Hugging Face mirror (`dgural/bdd100k`). Each image has weather, time of day, scene type and
  box labels.
- **Why this source:** the official download site (`dl.cv.ethz.ch`) was unreachable from our machine, the full dataset needs registration, and our disk had only 23 GB free.
- **Split by video, not by image.** Images from the same drive look alike. If one drive appeared in both train and test, the results would be inflated (leakage).
  So whole videos go either to train or to evaluation.
  - **Evaluation partition: 1,650 images** with fixed quotas: 400 clear-day, 500 night, 150 dawn/dusk, 300 rain, 300 snow.
  - **Train partition: 8,350 images** for pools and fine-tuning.

### Pools (pretraining sets, 1,200 frames each)
- `hf_clearday`: clear weather and daytime only. Metadata entropy 0 (no diversity).
- `hf_mixed`: time of day x weather strata equalized (water-filling). Weather entropy 0.91, time-of-day entropy 0.98.

### Pretraining (MAE)
- ViT-S/16, input 224 px, mask ratio 75%, normalized-pixel loss.
- **8,000 optimizer steps, effective batch 128** (batch 64 x 2 gradient accumulation), identical for both pools. Length is in *steps*, not epochs, so compute is equal.
- About 850 passes over each pool. Final loss 0.444 (narrow) and 0.454 (diverse).

### Fine-tuning (detection)
- Faster R-CNN on a ViT feature pyramid. Classes: **vehicle** (car, truck, bus) and **pedestrian** only, since those are consistent across datasets.
- Trained on **1,364 clear-daytime images** only, 3,000 steps, effective batch 4, 2 seeds. Final checkpoint used, with **no selection using OOD data**.

### Starting points compared
random init, ImageNet-supervised ViT-S, MAE on narrow pool, MAE on diverse pool. Each with 2 fine-tuning seeds = 8 fine-tuned models.

### Hardware
Laptop RTX 3050, 4 GB. We used bf16 and gradient accumulation. A 15-hour total time budget.

---

## 5. Results (memorize these numbers)

Detection AP (%), mean of 2 seeds:

| Init | Clear day (ID) | Dawn/dusk | Night | Rain | Snow | Mean OOD | Worst OOD |
|---|---|---|---|---|---|---|---|
| Random | 7.6 | 5.8 | 1.2 | 4.8 | 4.4 | 4.0 | 1.2 |
| ImageNet (supervised) | 17.8 | 16.1 | 7.8 | 14.4 | 14.6 | 13.2 | 7.8 |
| MAE narrow | 10.2 | 8.5 | 2.3 | 6.8 | 6.8 | 6.1 | 2.3 |
| MAE diverse | 10.3 | 8.6 | 2.4 | 6.9 | 7.2 | 6.3 | 2.4 |

Key statements:
1. Any MAE pretraining beats random init (6.1 to 6.3 vs 4.0 mean OOD).
2. **Diverse vs narrow: +0.20 AP out of domain** (bootstrap 95% CI [0.08, 0.30], Wilcoxon p = 0.023, 8 paired comparisons). Also +0.16 in-domain. Diverse wins in 9 of 10 (domain, seed) pairs; the exception is rain, seed 1 (-0.12).
3. The gain is about 3% relative: **small**.
4. **ImageNet beats all self-supervised models by a wide margin** (13.2 vs 6.3).
5. Night is the hardest domain for every model.
6. Effective-robustness residuals are within +/-0.5 AP points and the baseline fit has only two points, so **we do not claim effective robustness**.

---

## 6. Honest limitations (examiners love this; say them yourself)

1. **One pretraining run per pool.** The two compared models each come from a single pretraining run. The p-value only reflects fine-tuning noise,
   so we cannot separate "diverse pool helps" from "this pretraining run was slightly lucky".
2. **Tiny pools** (1.2K frames); small gap (about 3% relative).
3. **Possible confound: target proximity.** The diverse pool contains night, rain and snow images like the test slices. The gain may come from being *closer to the targets*, not from diversity in general.
   We did not run the Fréchet-distance check.
4. **BDD only.** No cross-dataset shift (nuScenes, Waymo, Cityscapes, ACDC) tested. Detection only (no segmentation results).
5. The 8 OOD pairs are not fully independent (domains share the same models).
6. Hypotheses H2 to H5 from the plan (leave-one-dataset-out, scale vs diversity, label "exchange rate", MoCo) were **not tested**.

---

## 7. What the code does (pipeline)

```
HF BDD-val (images + labels)
  -> scripts/prepare_bdd_hf.py     split by video, resize frames to 256 px short side, write manifest + detection JSONL + eval-id list
  -> scripts/build_pool.py         sample exact-size pools (error if too few frames)
  -> scripts/pretrain.py           MAE pretraining, step-based -> final.pt
  -> scripts/finetune.py           Faster R-CNN fine-tuning on clear-day labels -> evaluate all domains -> results.json
  -> scripts/analyze.py            tables, effective robustness, paired test, figures
  -> scripts/visualize_mae.py      input / masked / reconstruction figure
  -> scripts/describe_pools.py     pool composition chart
```
`scripts/run_experiment.py --exp configs/experiments/hf_det.yaml` runs the whole grid (idempotent and resumable).

**Code we added on top of the original repo:** MAE visualizer; paired significance test (`analysis.paired_comparison`); pool-composition plot;
Hugging Face data converter with video-level split (`src/xdssl/data/sources/bdd100k_hf.py`); small-GPU configs; the report.

**Research invariants the code enforces:** iso-size pools (exact size or error), iso-compute pretraining (in steps), identical labeled subsets across inits,
no model selection on OOD data, no leakage of evaluation images into pools, harmonized classes.

---

## 8. Likely viva questions and model answers

### Concept questions
**Q1. What is self-supervised learning and why use it here?**
It learns features from unlabeled data by solving a pretext task made from the data (e.g., reconstruct masked patches). Driving video is plentiful but labels are expensive, so SSL lets us use cheap data.

**Q2. How does MAE work? Why mask 75%?**
Split the image into 16x16 patches, hide 75% at random, encode only the visible patches, then a light decoder predicts the pixels of the hidden ones. A high mask ratio makes the task hard enough to learn semantics
(neighbouring pixels are too predictable otherwise) and makes the encoder cheap because it processes only 25% of tokens.

**Q3. Why a Vision Transformer?**
MAE is natural for ViTs because patches can be dropped. The same backbone serves detection and segmentation, and ViTs pretrained with SSL scale well.

**Q4. What is the difference between pretraining and fine-tuning?** See section 3.

**Q5. What is distribution shift? Give examples from this project.**
A mismatch between training and test conditions. We train on clear daytime and test on night, rain, snow and dawn/dusk.

**Q6. What is effective robustness and why not just compare OOD accuracy?**
A model with higher in-domain accuracy usually also has higher OOD accuracy. Effective robustness measures OOD performance *beyond* what a baseline line (OOD vs ID) predicts, so it separates real robustness from general accuracy. We computed it but did not claim it, since our baseline fit had only two points.

**Q7. What is AP? What is IoU?** See section 3.

### Design questions
**Q8. Why fix the pool size and the number of steps?**
Otherwise "diverse" is confounded with "more data" or "more compute". Fixing size and steps isolates diversity.

**Q9. Why steps instead of epochs?**
Pools may differ in size; the same number of epochs would mean different compute. Equal steps gives equal compute.

**Q10. Why split by video?**
Frames from one drive are near duplicates. If they were on both sides of the split, evaluation would measure memorization, not generalization.

**Q11. Why train the detector only on clear-day images?**
To create a real distribution shift: the model never sees labeled night/rain/snow, so OOD scores reflect how well the *pretrained features* generalize.

**Q12. Why only two detection classes?**
Vehicle and pedestrian have compatible definitions across BDD, nuImages and Waymo. Other classes (e.g., rider/bicycle) are boxed differently across datasets.

**Q13. Why did you not choose checkpoints using OOD data?**
That would leak target information and inflate OOD scores. We use the final checkpoint only.

**Q14. Why is ImageNet init so much better?**
It was trained supervised on 1.28M labeled images, versus our 1,200 unlabeled frames for 8K steps. SSL with so little data cannot match it. The claim we test is the *ordering of pools*, not beating ImageNet.

**Q15. Why did you use the Hugging Face BDD-val copy?**
The official download host was unreachable from our network, the full dataset needs registration, and disk space was limited (23 GB). The mirror holds the official 10K validation images with attributes and boxes, about 0.7 GB.

### Results questions
**Q16. Is your result significant?**
The paired test gives p = 0.023 and the bootstrap CI excludes zero, but it only covers fine-tuning noise, since each pool was pretrained once. The effect (+0.2 AP, about 3%) is small. We call it suggestive, not conclusive.

**Q17. Why might diverse pretraining help?**
The encoder sees night, rain and snow appearance during pretraining, so its features are less specialized to clear daytime. This is a hypothesis; we did not isolate the mechanism.

**Q18. Could the gain come from something other than diversity?**
Yes: proximity to the test conditions (the diverse pool contains the same conditions as the test slices), and pretraining-seed variance. Fréchet-distance analysis and more seeds would help.

**Q19. The diverse model is also better in-domain. Does that undermine "robustness"?**
It means part of the gain is general accuracy rather than robustness specifically. That is why effective robustness exists, but with our tiny setup we cannot confirm it.

**Q20. Why is night the hardest domain?**
Low light, glare and noise are far from the clear-day training images, and BDD night boxes are harder (small, dark objects).

**Q21. What do the MAE reconstructions show?**
The model fills in plausible colours and layout even at night and dusk, showing pretraining learned scene structure. Reconstructions are blurry, as expected for MAE at this scale.

### Critical and extension questions
**Q22. What would you do with more time?**
More pretraining seeds per pool; the Fréchet-distance check; larger pools (BDD video frames, nuScenes, Waymo); cross-dataset targets (ACDC, Cityscapes, nuImages); segmentation; continued pretraining from ImageNet weights; MoCo v3 as a second method.

**Q23. Does this generalize to other methods or datasets?**
Not shown. We tested only MAE, only BDD and only detection.

**Q24. What was the biggest practical difficulty?**
Data access (official host unreachable), a 4 GB GPU (small batches, gradient accumulation), and one random-init evaluation that ran out of memory (re-evaluated with batch size 1).

**Q25. How do you guarantee no leakage?**
Video-level split; evaluation videos are listed in `bdd_eval_ids.txt` and removed from pools; pools use only the `train` partition.

### Team and code questions
**Q26. Which code is yours vs. the original repo?**
The repo came with the pipeline scaffold (pools, MAE, detection, evaluation). We added the data converter for the Hugging Face mirror with a video-level split, the MAE visualizer, the paired significance test, the pool composition plot, small-GPU configs, the experiment, and the report.
*(Each team member should be able to explain at least the pipeline in section 7.)*

**Q27. How would you run it again?**
`python scripts/prepare_bdd_hf.py --raw <raw dir>`, then `python scripts/run_experiment.py --exp configs/experiments/hf_det.yaml`, then `python scripts/analyze.py --exp configs/experiments/hf_det.yaml --compare mae-hf_mixed mae-hf_clearday`.
Set `configs/paths.yaml` first (copy from `configs/paths.example.yaml`).

---

## 9. Quick facts cheat sheet

- Backbone: ViT-S/16 (about 22M parameters), input 224 px
- MAE: mask 75%, decoder 4 blocks, dim 256
- Pools: 1,200 frames each; narrow = clear day; diverse = balanced over time of day x weather
- Pretraining: 8,000 steps, effective batch 128, same for both pools
- Fine-tuning: Faster R-CNN, 1,364 clear-day images, 3,000 steps, 2 seeds
- Data: 10,000 BDD100K val images -> 8,350 train + 1,650 eval (split by video)
- Eval quotas: 400 clear-day, 500 night, 150 dawn/dusk, 300 rain, 300 snow
- Headline: diverse vs narrow +0.20 AP OOD (CI [0.08, 0.30], p = 0.023); ImageNet 13.2 vs MAE about 6.2 mean OOD
- Hardware: RTX 3050 4 GB; each fine-tune about 35 min, each pretrain about 45 min
- Repository branch with all work: `data-free-prep` on github.com/sidbhagat40/CV-project

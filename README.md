# Cross-Domain Self-Supervised Pretraining for Robust Driving Perception

Does pretraining on **more diverse** unlabeled driving footage (several cities, weathers and times of day,
pooled from BDD100K, nuScenes and Waymo) make perception models more robust to **unseen conditions**? And is it
worth more than extra labeled data from a single domain?

This repo pretrains a ViT with self-supervised learning (MAE; MoCo v3 as an ablation) on pools of driving frames
that have **identical size and compute but different diversity**. It then fine-tunes each model on one labeled
source domain (BDD100K) and measures performance on shifted targets:

- BDD night / rain / snow slices
- nuImages (Boston, Singapore)
- Waymo (day, night, dawn)
- Cityscapes
- ACDC fog / night / rain / snow

Robustness is reported both as raw out-of-domain scores and as **effective robustness**: OOD performance beyond
what in-domain accuracy predicts.

- Research design, hypotheses, confounds, compute budget: [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md)
- Dataset download and preparation: [docs/DATA.md](docs/DATA.md)
- Code structure and extension points: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Open tasks: [docs/ROADMAP.md](docs/ROADMAP.md)

## Quickstart

```powershell
pip install -r requirements.txt
pip install -e .
python -m pytest -q                                   # unit + end-to-end tests on synthetic data

# whole pipeline on synthetic data (~3 min): pools -> MAE -> seg/det fine-tune -> eval -> tables/figures
python scripts/make_synthetic_data.py --out data_synth
python scripts/run_experiment.py --exp configs/smoke/experiment.yaml --paths data_synth/paths.yaml
python scripts/analyze.py        --exp configs/smoke/experiment.yaml --paths data_synth/paths.yaml
```

## Real data workflow

```powershell
copy configs\paths.example.yaml configs\paths.yaml    # then edit dataset locations

# 1. unlabeled frames + unified annotations
python scripts/prepare_frames.py bdd100k-images
python scripts/prepare_frames.py nuscenes --hz 2
python scripts/prepare_frames.py waymo --hz 1 --delete-source
python scripts/prepare_labeled.py bdd-seg bdd-det acdc cityscapes nuimages waymo-det

# 2. an experiment grid (idempotent and resumable; --dry-run prints the commands)
python scripts/run_experiment.py --exp configs/experiments/e1_diversity.yaml --dry-run
python scripts/run_experiment.py --exp configs/experiments/e1_diversity.yaml

# 3. diversity measurements, tables, figures, demo
python scripts/pool_diversity.py --pools bdd_clearday bdd_mixed bdd_nusc_waymo --targets configs/eval/seg_targets.yaml
python scripts/analyze.py --exp configs/experiments/e1_diversity.yaml
python scripts/demo.py --models runs/finetune/<a>/model.pt runs/finetune/<b>/model.pt `
    --jsonl <labeled_root>/seg/acdc_val.jsonl --filter condition=fog --out outputs/demo_fog.png
```

Any config value can be overridden from the command line, e.g.
`python scripts/pretrain.py --config configs/pretrain/mae_vit_small.yaml data.pool=... optim.total_steps=5000`.

## Layout

```
configs/      paths, pools, pretrain/finetune recipes, evaluation suites, experiment grids, smoke configs
src/xdssl/    data (indexers, pools, datasets), models (ViT, MAE, MoCo, heads), engine, metrics, analysis
scripts/      thin CLIs: prepare_*, build_pool, pretrain, finetune, evaluate, run_experiment, analyze, demo, benchmark
tests/        unit tests + end-to-end pipeline test on synthetic data
docs/         research plan, data guide, architecture, roadmap
```

## Measured cost (RTX 4060 Laptop, 8 GB, bf16)

| job | time |
|-----|------|
| MAE ViT-S pretraining (30k steps x 256) | ~2.5 h |
| Segmentation fine-tune (16k steps) | ~1.3 h |
| Detection fine-tune (20k steps) | ~2.3 h |
| Segmentation linear probe | ~0.3 h |

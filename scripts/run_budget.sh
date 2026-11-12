#!/usr/bin/env bash
# One-shot ~11 h pipeline for a 4 GB GPU. Prerequisite: configs/paths.yaml filled in and raw data extracted.
# Resumable: every step skips outputs that already exist.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
python3 scripts/prepare_frames.py bdd100k-images
python3 scripts/prepare_labeled.py bdd-seg acdc
python3 scripts/run_experiment.py --exp configs/experiments/budget_linear.yaml   # pools, 2 pretrains, 4 probes
python3 scripts/run_experiment.py --exp configs/experiments/budget_fpn.yaml      # 4 full fine-tunes
python3 scripts/pool_diversity.py --pools bdd_clearday bdd_mixed --targets configs/eval/seg_targets.yaml
python3 scripts/describe_pools.py --pools bdd_clearday bdd_mixed --out outputs/pools.png
python3 scripts/visualize_mae.py --ckpt runs/pretrain/mae_vit_small_bdd_mixed/final.pt --pool "$(python3 -c "from xdssl.utils.config import load_paths,resolve_path;print(resolve_path(load_paths().pools_root)/'bdd_mixed'/'pool.parquet')")"
python3 scripts/analyze.py --exp configs/experiments/budget_linear.yaml --compare mae-bdd_mixed mae-bdd_clearday
python3 scripts/analyze.py --exp configs/experiments/budget_fpn.yaml --compare mae-bdd_mixed mae-bdd_clearday

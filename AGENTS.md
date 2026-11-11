# Guidance for AI agents working in this repo

Read `docs/RESEARCH_PLAN.md` (what we are testing and why) and `docs/ARCHITECTURE.md` (where things live)
before changing code. Open tasks are in `docs/ROADMAP.md`; tick items off there when you finish them.

## Commands
- Install: `pip install -e .` (PyTorch with CUDA is already installed system-wide on the dev machine)
- Tests (must pass before you finish): `python -m pytest -q`; takes ~30 s, uses synthetic data
- Full smoke grid: `python scripts/make_synthetic_data.py --out data_synth` then
  `python scripts/run_experiment.py --exp configs/smoke/experiment.yaml --paths data_synth/paths.yaml`
- GPU sizing: `python scripts/benchmark.py`

## Research invariants (do not break these)
1. **Iso-size pools.** `build_pool` returns exactly `size` frames or raises an error. Never make it silently return
   fewer frames.
2. **Iso-compute pretraining.** Pretraining length is in steps, identical across compared pools. Do not switch it
   to epochs.
3. **Identical labeled subsets.** `load_records` subsets depend only on (file, filter, fraction, subset_seed).
   Never make them depend on the model or the run seed.
4. **No target-domain selection.** Never pick checkpoints, hyper-parameters or early-stopping points using OOD
   target domains. Final checkpoint only; in-domain validation at most.
5. **No leakage.** Pools use source `train` splits and `bdd_eval_ids.txt`. Evaluation images must never enter a
   pool.
6. **Harmonized taxonomies.** Segmentation uses Cityscapes-19 trainIds (255 = ignore). Detection uses `DET_CLASSES`
   (0-based in JSONL; +1 inside the torchvision detector). Map new datasets onto these; do not extend them
   casually, because that changes every result.
7. **Results schema.** `results.json` (`meta`, `domains`, `summary`) is consumed by `analysis.py`. Extend it, but
   don't rename fields.

## Conventions
- Configs are the source of truth. New behaviour goes behind a config key with a default that preserves current
  results. Paths always come from `${paths.*}`; never hard-code machine paths in code.
- Windows dev machine: every script needs `if __name__ == "__main__":` (DataLoader workers spawn processes), and
  paths should go through `pathlib` / `resolve_path`.
- Keep scripts thin (argument parsing only) and put logic in `src/xdssl`.
- Match the existing style: type hints on public functions, short docstrings that state intent or constraints,
  no comments narrating the code.
- New logic needs a test in `tests/` that runs on CPU in seconds (use `xdssl.data.synthetic`).

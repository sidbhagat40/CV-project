"""End-to-end: synthetic data -> pools -> MAE/MoCo pretraining -> seg + det fine-tuning -> evaluation -> analysis."""

import json

import pandas as pd
import pytest

from xdssl import analysis as A
from xdssl.data.pools import build_pool, summarize_pool
from xdssl.data.synthetic import make_synthetic
from xdssl.engine.finetune import load_finetuned, run_finetune
from xdssl.engine.pretrain import run_pretrain
from xdssl.utils.config import load_config


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return str(make_synthetic(tmp_path_factory.mktemp("synth")))


def test_full_pipeline(synth, tmp_path):
    spec = load_config("configs/pools/smoke_diverse.yaml", paths_file=synth)
    pool = build_pool({k: v for k, v in spec.items() if k != "paths"}, spec.paths.frames_root)
    assert len(pool) == 24 and summarize_pool(pool)["by_dataset"] == {"bdd100k": 8, "nuscenes": 8, "waymo": 8}
    pool_file = tmp_path / "pool.parquet"
    pool.to_parquet(pool_file)

    ckpt = run_pretrain(load_config("configs/smoke/pretrain_mae.yaml",
                                    [f"data.pool={pool_file.as_posix()}", f"output_dir={(tmp_path / 'mae').as_posix()}"], synth))
    assert ckpt.exists()
    moco = run_pretrain(load_config("configs/smoke/pretrain_moco.yaml",
                                    [f"data.pool={pool_file.as_posix()}", f"output_dir={(tmp_path / 'moco').as_posix()}"], synth))
    assert moco.exists()

    # synthetic Waymo has no dawn/dusk frames, so that det domain is skipped (10 of 11)
    for cfg_path, metric, n_domains in [("configs/smoke/seg.yaml", "mIoU", 6), ("configs/smoke/det.yaml", "AP", 10)]:
        out = tmp_path / "runs" / "finetune" / f"x__{metric}"
        res = run_finetune(load_config(cfg_path, ["model.init=ssl", f"model.ckpt={ckpt.as_posix()}",
                                                   f"output_dir={out.as_posix()}", "meta.init_name=x"], synth))
        assert len(res["domains"]) == n_domains
        assert all(metric in r for r in res["domains"].values())
        assert "mean_ood" in res["summary"]
        model, cfg = load_finetuned(out / "model.pt")
        assert cfg.task in ("seg", "det")

    df = A.collect(tmp_path / "runs")
    assert set(df.task) == {"seg", "det"} and df.value.notna().all()
    assert not A.summary_table(df).empty


def test_experiment_expansion(synth):
    from xdssl.experiments import expand
    jobs = expand("configs/smoke/experiment.yaml", synth)
    kinds = [j.kind for j in jobs]
    assert kinds.count("pool") == 2 and kinds.count("pretrain") == 2
    assert kinds.count("finetune") == 3 * 2 * 2  # inits x configs x label fractions
    assert len({j.name for j in jobs if j.kind == "finetune"}) == 12


def test_mae_visualizer_roundtrip(synth, tmp_path):
    import importlib.util
    import torch
    spec = load_config("configs/pools/smoke_diverse.yaml", paths_file=synth)
    pool = build_pool({k: v for k, v in spec.items() if k != "paths"}, spec.paths.frames_root)
    pool.to_parquet(tmp_path / "pool.parquet")
    ckpt = run_pretrain(load_config("configs/smoke/pretrain_mae.yaml",
                                    [f"data.pool={(tmp_path / 'pool.parquet').as_posix()}",
                                     f"output_dir={(tmp_path / 'mae').as_posix()}"], synth))
    sp = importlib.util.spec_from_file_location("visualize_mae", "scripts/visualize_mae.py")
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    model, cfg = mod.load_mae(ckpt, torch.device("cpu"))
    x = torch.randn(2, 3, *cfg.data.input_size)
    masked, recon, paste = model.reconstruct(x)
    assert masked.shape == recon.shape == paste.shape == x.shape
    assert (masked == 0).float().mean() > 0.5  # ~75% of patches are masked

import numpy as np
import pandas as pd
import pytest
import torch

from xdssl.data.pools import apply_filter, build_pool, waterfill
from xdssl.engine.optim import param_groups
from xdssl.metrics.diversity import metadata_entropy, vendi_score
from xdssl.metrics.robustness import effective_robustness, fit_baseline, summarize_domains
from xdssl.metrics.segmentation import confusion_matrix, miou
from xdssl.models.mae import MAE
from xdssl.models.segmentation import SegmentationModel
from xdssl.models.vit import create_vit


def test_waterfill_respects_caps_and_total():
    assert waterfill(10, [100, 100]) == [5, 5]
    assert waterfill(10, [2, 100, 100]) == [2, 4, 4]
    assert sum(waterfill(7, [3, 3, 3])) == 7
    with pytest.raises(ValueError):
        waterfill(10, [2, 3])


def test_apply_filter_include_exclude():
    df = pd.DataFrame({"a": ["x", "y", "z"], "b": [1, 2, 3]})
    assert apply_filter(df, {"a": ["x", "y"]}, {"b": 2})["a"].tolist() == ["x"]


def _index(tmp_path, n_per=50):
    from xdssl.data.frames import write_manifest
    for ds in ["d1", "d2"]:
        rows = [{"frame_id": f"{ds}/{i}", "dataset": ds, "split": "train", "sequence_id": f"{ds}s{i // 5}",
                 "camera": "front", "timestamp": i, "path": f"{ds}/{i}.jpg",
                 "weather": ["clear", "rain"][i % 2], "timeofday": "day" if i < 40 else "night",
                 "location": ds, "scene_type": "x"} for i in range(n_per)]
        write_manifest(rows, tmp_path, ds)


def test_build_pool_is_exact_size_and_balanced(tmp_path):
    _index(tmp_path)
    spec = {"name": "p", "size": 40, "seed": 0, "balance_by": ["timeofday"],
            "sources": [{"dataset": "d1", "weight": 1}, {"dataset": "d2", "weight": 1}]}
    pool = build_pool(spec, tmp_path)
    assert len(pool) == 40 and pool.frame_id.is_unique
    assert pool.dataset.value_counts().to_dict() == {"d1": 20, "d2": 20}
    assert (pool.timeofday == "night").sum() == 20  # 10 night frames per source, all taken by water-filling
    # determinism
    assert build_pool(spec, tmp_path).frame_id.tolist() == pool.frame_id.tolist()


def test_build_pool_raises_when_source_too_small(tmp_path):
    _index(tmp_path)
    spec = {"name": "p", "size": 90, "sources": [{"dataset": "d1", "weight": 2}, {"dataset": "d2", "weight": 1}]}
    with pytest.raises(ValueError):
        build_pool(spec, tmp_path)


def test_miou_perfect_and_ignore():
    t = torch.tensor([[0, 1], [2, 255]])
    cm = confusion_matrix(t.clamp(max=2), t, 3)
    assert miou(cm) == 1.0


def test_vendi_bounds():
    assert vendi_score(np.ones((50, 8))) == pytest.approx(1.0, abs=1e-6)
    assert vendi_score(np.eye(8)) == pytest.approx(8.0, rel=1e-6)


def test_metadata_entropy():
    df = pd.DataFrame({"a": ["x"] * 5 + ["y"] * 5})
    assert metadata_entropy(df, ["a"]) == pytest.approx(1.0)


def test_effective_robustness_zero_on_fit_line():
    ids, oods = [0.3, 0.5, 0.7], [0.2, 0.35, 0.5]
    fit = fit_baseline(ids, oods)
    assert abs(effective_robustness(0.5, 0.35, fit)) < 0.02
    assert effective_robustness(0.5, 0.45, fit) > 0.05


def test_summarize_domains():
    s = summarize_domains({"a": {"mIoU": 0.6}, "b": {"mIoU": 0.3}, "c": {"mIoU": 0.4}}, "mIoU", "a")
    assert s["worst_ood_domain"] == "b" and s["mean_ood"] == pytest.approx(0.35)


def test_mae_forward_and_vit_resolution_change():
    vit = create_vit("vit_tiny", img_size=(64, 64))
    mae = MAE(vit, decoder_dim=32, decoder_depth=1, decoder_heads=2)
    losses, out = mae(torch.randn(2, 3, 64, 64))
    assert losses["loss"].ndim == 0 and out["mask"].sum() == 2 * 12
    feats = vit.forward_features(torch.randn(1, 3, 96, 160), [3, 11])
    assert feats[0].shape == (1, 192, 6, 10)


def test_segmentation_model_output_shape():
    model = SegmentationModel(create_vit("vit_tiny", img_size=(64, 64)), 19)
    assert model(torch.randn(2, 3, 64, 96)).shape == (2, 19, 64, 96)


def test_layer_decay_groups():
    vit = create_vit("vit_tiny")
    model = SegmentationModel(vit, 19)
    groups = param_groups(model, 1e-3, 0.05, backbone=vit, layer_decay=0.5)
    lrs = sorted({g["lr"] for g in groups})
    assert lrs[0] == pytest.approx(1e-3 * 0.5 ** 13) and lrs[-1] == pytest.approx(1e-3)


def test_paired_comparison_detects_consistent_gain():
    import pandas as pd
    from xdssl.analysis import paired_comparison
    rows = []
    for seed in range(3):
        for dom, base in [("fog", 0.30), ("night", 0.25), ("rain", 0.35)]:
            for init, gain in [("a", 0.05), ("b", 0.0)]:
                rows.append(dict(task="seg", label_fraction=1.0, seed=seed, domain=dom, init_name=init, is_id=False,
                                 value=base + gain + 0.001 * seed))
    out = paired_comparison(pd.DataFrame(rows), "a", "b")
    assert out.n_pairs.iloc[0] == 9 and abs(out.mean_diff.iloc[0] - 0.05) < 1e-9
    assert out.ci95_lo.iloc[0] > 0 and out.wilcoxon_p.iloc[0] < 0.01


def test_plot_pool_composition(tmp_path):
    import json
    from xdssl.analysis import plot_pool_composition
    for name, ds in [("p1", {"bdd100k": 10}), ("p2", {"bdd100k": 5, "waymo": 5})]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary.json").write_text(json.dumps(
            {"n_frames": 10, "by_dataset": ds, "by_weather": {"clear": 10}, "by_timeofday": {"day": 6, "night": 4}}))
    plot_pool_composition(tmp_path, ["p1", "p2"], tmp_path / "o.png")
    assert (tmp_path / "o.png").stat().st_size > 0


def test_bdd_hf_split_is_by_video_and_meets_quota(tmp_path):
    from xdssl.data.sources.bdd100k_hf import _record, split_by_video
    from pathlib import Path
    samples = []
    for v in range(40):
        for i in range(3):
            samples.append({"filepath": f"data/v{v:04d}-{i:04x}.jpg", "metadata": {"width": 1280, "height": 720},
                            "weather": {"label": "rainy" if v % 2 else "clear"},
                            "timeofday": {"label": "daytime"}, "scene": {"label": "city street"},
                            "detections": {"detections": [
                                {"label": "car", "bounding_box": [0.1, 0.2, 0.1, 0.1]},
                                {"label": "traffic sign", "bounding_box": [0, 0, 0.1, 0.1]}]}})
    recs = [_record(Path("raw"), s) for s in samples]
    assert len(recs[0]["boxes"]) == 1 and recs[0]["boxes"][0] == pytest.approx([128.0, 144.0, 256.0, 216.0])
    split = split_by_video(recs, {"clear_day": 6, "rain": 9}, seed=0)
    for v in range(40):
        assert len({split[f"v{v:04d}-{i:04x}"] for i in range(3)}) == 1
    assert sum(s == "val" for s in split.values()) == 15

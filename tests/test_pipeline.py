"""End-to-end smoke test on synthetic audio: train -> save -> predict, no dataset needed."""

import json

import numpy as np
import pytest
import soundfile as sf

from raga.config import Config, EvalConfig, FeatureConfig
from raga.evaluate import bootstrap_ci
from raga.predict import RagaPredictor
from raga.split import save_split
from raga.train import run

SR = 22050
# Three "ragas", each a distinct tone so a linear model separates them.
TOY = {"Thodi_sad": 220.0, "Saveri_compassion": 330.0, "Bilahari_tranquility": 495.0}


def _clip(freq, k, rng):
    t = np.arange(SR) / SR
    y = 0.3 * np.sin(2 * np.pi * freq * (1 + 0.01 * k) * t) + 0.01 * rng.standard_normal(SR)
    return y.astype(np.float32)


@pytest.fixture
def toy_dataset(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "data"
    (root / "Studio_Audio_split").mkdir(parents=True)
    train, test = [], []
    for stem, freq in TOY.items():
        for k in range(1, 9):
            name = f"{stem}_{k:02d}.wav"
            sf.write(root / "Studio_Audio_split" / name, _clip(freq, k, rng), SR)
            (test if k > 6 else train).append(name)
    split = tmp_path / "split.json"
    save_split(split, train, test)
    return root, split


@pytest.mark.parametrize("strategy", ["stratified", "blocked"])
def test_train_and_predict_end_to_end(toy_dataset, tmp_path, strategy):
    root, split = toy_dataset
    cfg = Config(
        name="toy",
        split=str(split),
        artifacts_dir=str(tmp_path / "artifacts"),
        features=FeatureConfig(duration=None),
        eval=EvalConfig(cv_folds=3, cv_strategy=strategy, n_bootstrap=200),
    )
    metrics = run(cfg, root, evaluate_test=True)

    assert metrics["test"]["accuracy"] == 1.0
    assert metrics["cv"]["accuracy_mean"] > metrics["cv_chance"]["accuracy_mean"]
    run_dir = tmp_path / "artifacts" / "runs" / "toy"
    assert json.loads((run_dir / "metrics.json").read_text())["n_train"] == 18
    looks = (tmp_path / "artifacts" / "test_evaluations.jsonl").read_text().splitlines()
    assert len(looks) == 1

    predictor = RagaPredictor(run_dir / "model.joblib")
    out = predictor.predict_file(root / "Studio_Audio_split" / "Saveri_compassion_08.wav")
    assert out["raga"] == "saveri" and out["emotion"] == "compassion"
    assert abs(sum(p["prob"] for p in out["top_k"]) - 1.0) < 1e-6  # 3 classes, top_k=3


def test_train_without_flag_does_not_touch_test(toy_dataset, tmp_path):
    root, split = toy_dataset
    cfg = Config(
        name="toy",
        split=str(split),
        artifacts_dir=str(tmp_path / "artifacts"),
        features=FeatureConfig(duration=None),
        eval=EvalConfig(cv_folds=3),
    )
    metrics = run(cfg, root)
    assert "test" not in metrics
    assert not (tmp_path / "artifacts" / "test_evaluations.jsonl").exists()


def test_bootstrap_ci_brackets_point_estimate():
    y = np.array(["a"] * 30 + ["b"] * 3)
    pred = y.copy()
    pred[:3] = "b"
    ci = bootstrap_ci(y, pred, n_boot=500, seed=0)
    lo, hi = ci["accuracy"]
    assert lo < (pred == y).mean() < hi

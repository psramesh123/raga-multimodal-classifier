"""End-to-end training run: manifest -> split -> features -> CV -> fit -> (test) -> save."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib

from raga import __version__
from raga.config import Config, ModelConfig
from raga.data import RAGA_EMOTION, build_manifest, originals
from raga.evaluate import bootstrap_ci, cross_validate, plot_confusion, score
from raga.features import (
    FEATURE_SCHEMA_VERSION,
    build_feature_table,
    feature_columns,
    select_groups,
)
from raga.models import build_model
from raga.split import apply_split, cv_folds, load_split


def _git_sha() -> str | None:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        out = subprocess.run(cmd, capture_output=True, text=True)
        return out.stdout.strip() or None
    except OSError:
        return None


def run(cfg: Config, root: Path, evaluate_test: bool = False) -> dict:
    run_dir = Path(cfg.artifacts_dir) / "runs" / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = originals(build_manifest(root))
    train_rec, test_rec = apply_split(manifest, load_split(cfg.split))
    print(f"[{cfg.name}] train={len(train_rec)} test={len(test_rec)} (test sealed)")

    # Test features are extracted only when the test set is actually evaluated.
    cache_dir = Path(cfg.artifacts_dir) / "cache"
    train = build_feature_table(train_rec, root, cfg.features, cache_dir)
    cols = select_groups(feature_columns(train), cfg.features.groups)
    X, y = train[cols], train["raga"]
    labels = sorted(y.unique())

    def folds():
        return cv_folds(train, cfg.eval.cv_strategy, cfg.eval.cv_folds, cfg.eval.seed)

    seed = cfg.eval.seed
    chance = cross_validate(build_model(ModelConfig(name="dummy"), seed), X, y, folds())
    model = build_model(cfg.model, seed)
    cv = cross_validate(model, X, y, folds())
    oof = cv.pop("oof_pred")
    chance.pop("oof_pred")
    print(
        f"[{cfg.name}] CV ({cfg.eval.cv_strategy}, {len(cols)} features): "
        f"acc {cv['accuracy_mean']:.3f} ± {cv['accuracy_std']:.3f}  "
        f"macro-F1 {cv['macro_f1_mean']:.3f} ± {cv['macro_f1_std']:.3f}  "
        f"(chance acc {chance['accuracy_mean']:.3f})"
    )
    plot_confusion(
        y,
        oof,
        labels,
        f"{cfg.name}: {cfg.eval.cv_strategy} CV (training originals)",
        run_dir / "confusion_cv.png",
    )

    model.fit(X, y)
    metrics = {
        "run": cfg.name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "package_version": __version__,
        "config": dataclasses.asdict(cfg),
        "n_train": len(train),
        "n_features": len(cols),
        "cv": cv,
        "cv_chance": chance,
        "cv_oof": score(y, oof, labels),
    }

    if evaluate_test:
        test = build_feature_table(test_rec, root, cfg.features, cache_dir)
        pred = model.predict(test[cols])
        result = score(test["raga"], pred, labels)
        result["ci95"] = bootstrap_ci(test["raga"], pred, cfg.eval.n_bootstrap, seed)
        metrics["test"] = result
        lo, hi = result["ci95"]["accuracy"]
        print(f"[{cfg.name}] TEST acc {result['accuracy']:.3f} (95% CI {lo:.2f}-{hi:.2f})")
        plot_confusion(
            test["raga"], pred, labels, f"{cfg.name}: held-out test", run_dir / "confusion_test.png"
        )
        _log_test_look(cfg, metrics)

    bundle = {
        "model": model,
        "classes": list(model.classes_),
        "feature_columns": cols,
        "feature_config": dataclasses.asdict(cfg.features),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "raga_emotion": RAGA_EMOTION,
        "run": cfg.name,
        "git_sha": metrics["git_sha"],
    }
    joblib.dump(bundle, run_dir / "model.joblib")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[{cfg.name}] saved {run_dir}/model.joblib and metrics.json")
    return metrics


def _log_test_look(cfg: Config, metrics: dict) -> None:
    """Append-only record of every look at the test set, so repeated peeking is visible."""
    log = Path(cfg.artifacts_dir) / "test_evaluations.jsonl"
    entry = {
        "at": metrics["created_at"],
        "run": cfg.name,
        "git_sha": metrics["git_sha"],
        "split": cfg.split,
        "accuracy": metrics["test"]["accuracy"],
        "macro_f1": metrics["test"]["macro_f1"],
    }
    with log.open("a") as f:
        f.write(json.dumps(entry) + "\n")

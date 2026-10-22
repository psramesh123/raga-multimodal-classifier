"""Cross-validation, held-out evaluation, and bootstrap confidence intervals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, classification_report, f1_score


def cross_validate(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, folds) -> dict:
    """Per-fold scores plus out-of-fold predictions for every training row."""
    oof = np.empty(len(y), dtype=object)
    accs, f1s = [], []
    for tr, va in folds:
        fitted = clone(model).fit(X.iloc[tr], y.iloc[tr])
        pred = fitted.predict(X.iloc[va])
        oof[va] = pred
        accs.append(accuracy_score(y.iloc[va], pred))
        f1s.append(f1_score(y.iloc[va], pred, average="macro", zero_division=0))
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "fold_accuracy": [float(a) for a in accs],
        "oof_pred": oof,
    }


def bootstrap_ci(
    y_true, y_pred, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> dict[str, list[float]]:
    """Percentile bootstrap CI over examples for accuracy and macro-F1.

    With ~33 test clips the interval is wide; report it next to every point estimate.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, f1s = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        accs[b] = accuracy_score(y_true[idx], y_pred[idx])
        f1s[b] = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "accuracy": [float(np.percentile(accs, lo)), float(np.percentile(accs, hi))],
        "macro_f1": [float(np.percentile(f1s, lo)), float(np.percentile(f1s, hi))],
    }


def score(y_true, y_pred, labels: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
    }


def plot_confusion(y_true, y_pred, labels: list[str], title: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

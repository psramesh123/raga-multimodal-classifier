"""Model registry: config name -> unfitted scikit-learn estimator."""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from raga.config import ModelConfig


def build_model(cfg: ModelConfig, seed: int) -> BaseEstimator:
    p = dict(cfg.params)
    if cfg.name == "logreg":
        clf = LogisticRegression(max_iter=p.pop("max_iter", 2000), **p)
        return Pipeline([("scale", StandardScaler()), ("clf", clf)])
    if cfg.name == "random_forest":
        p.setdefault("n_estimators", 300)
        return RandomForestClassifier(random_state=seed, **p)
    if cfg.name == "hist_gb":
        return HistGradientBoostingClassifier(random_state=seed, **p)
    if cfg.name == "dummy":
        return DummyClassifier(strategy=p.get("strategy", "most_frequent"))
    raise ValueError(f"Unknown model {cfg.name!r}")

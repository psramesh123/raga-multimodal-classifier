"""Inference on a single recording with a saved model bundle."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from raga.config import FeatureConfig
from raga.features import FEATURE_SCHEMA_VERSION, extract_features, load_audio


class RagaPredictor:
    def __init__(self, bundle_path: str | Path):
        bundle = joblib.load(bundle_path)
        if bundle["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Model was trained on feature schema v{bundle['feature_schema_version']}, "
                f"this code extracts v{FEATURE_SCHEMA_VERSION}; retrain the model"
            )
        self.model = bundle["model"]
        self.classes: list[str] = bundle["classes"]
        self.columns: list[str] = bundle["feature_columns"]
        self.feature_cfg = FeatureConfig(**bundle["feature_config"])
        self.raga_emotion: dict[str, str] = bundle["raga_emotion"]
        self.run = bundle.get("run")

    def predict_array(self, y: np.ndarray, top_k: int = 3) -> dict:
        """``y`` must already be mono at ``self.feature_cfg.sr``."""
        feats = extract_features(y, self.feature_cfg)
        X = pd.DataFrame([feats])[self.columns]
        probs = self.model.predict_proba(X)[0]
        order = np.argsort(probs)[::-1][:top_k]
        top = [{"raga": self.model.classes_[i], "prob": float(probs[i])} for i in order]
        return {
            "raga": top[0]["raga"],
            "confidence": top[0]["prob"],
            "emotion": self.raga_emotion.get(top[0]["raga"]),
            "top_k": top,
            "model_run": self.run,
        }

    def predict_file(self, path: str | Path, top_k: int = 3) -> dict:
        return self.predict_array(load_audio(path, self.feature_cfg), top_k)

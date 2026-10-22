"""Per-recording aggregated audio descriptors (from notebooks/02_features_baseline).

Extraction always produces every feature family; ``select_groups`` narrows them at
model time, so ablations reuse one cached feature table.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from raga.config import FeatureConfig

# Bump when the meaning or order of feature columns changes; saved models record it.
FEATURE_SCHEMA_VERSION = 1

FEATURE_GROUPS = ("mfcc", "chroma", "spectral")


def load_audio(path: str | Path, cfg: FeatureConfig) -> np.ndarray:
    y, _ = librosa.load(path, sr=cfg.sr, duration=cfg.duration)
    return y


def extract_features(y: np.ndarray, cfg: FeatureConfig) -> dict[str, float]:
    """Timbral (MFCC), tonal (chroma) and spectral-shape summaries of one recording."""
    if y.size == 0:
        raise ValueError("Empty audio")
    sr = cfg.sr
    feats: dict[str, float] = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cfg.n_mfcc)
    for i, row in enumerate(mfcc, start=1):
        feats[f"mfcc{i}_mean"] = float(row.mean())
        feats[f"mfcc{i}_std"] = float(row.std())

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i, row in enumerate(chroma, start=1):
        feats[f"chroma{i}_mean"] = float(row.mean())
        feats[f"chroma{i}_std"] = float(row.std())

    feats["spec_centroid_mean"] = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    feats["spec_bandwidth_mean"] = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
    feats["spec_rolloff_mean"] = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
    feats["zcr_mean"] = float(librosa.feature.zero_crossing_rate(y).mean())
    feats["rms_mean"] = float(librosa.feature.rms(y=y).mean())
    return feats


def feature_group(column: str) -> str:
    if column.startswith("mfcc"):
        return "mfcc"
    if column.startswith("chroma"):
        return "chroma"
    return "spectral"


def select_groups(columns: list[str], groups: list[str]) -> list[str]:
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown feature groups {sorted(unknown)}; choose from {FEATURE_GROUPS}")
    return [c for c in columns if feature_group(c) in groups]


def _cache_key(cfg: FeatureConfig) -> str:
    # groups only filter columns, so they are not part of the extraction identity
    ident = {"v": FEATURE_SCHEMA_VERSION, "sr": cfg.sr, "dur": cfg.duration, "mfcc": cfg.n_mfcc}
    return hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:10]


def build_feature_table(
    recordings: pd.DataFrame, root: Path, cfg: FeatureConfig, cache_dir: Path | None = None
) -> pd.DataFrame:
    """Features for every row of ``recordings`` (needs ``filename``, ``path``, ``raga``).

    Results are cached per extraction config; only files missing from the cache are read.
    """
    cache = None
    cached = pd.DataFrame()
    if cache_dir is not None:
        cache = Path(cache_dir) / f"features_{_cache_key(cfg)}.csv"
        if cache.exists():
            cached = pd.read_csv(cache)

    done = set(cached["filename"]) if not cached.empty else set()
    todo = recordings[~recordings["filename"].isin(done)]
    rows = []
    for i, rec in enumerate(todo.itertuples(index=False), start=1):
        feats = extract_features(load_audio(root / rec.path, cfg), cfg)
        rows.append({"filename": rec.filename, **feats})
        if i % 20 == 0 or i == len(todo):
            print(f"  features: {i}/{len(todo)}", flush=True)

    table = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
    if cache is not None and rows:
        cache.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(cache, index=False)

    out = recordings[["filename", "raga", "take"]].merge(table, on="filename", how="left")
    if out.drop(columns=["take"]).isna().any().any():
        raise RuntimeError("Feature extraction left missing values")
    return out


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c not in ("filename", "raga", "take")]

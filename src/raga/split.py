"""Train/test splits stored as filename lists, and cross-validation folds.

Splits are files, not code: the sealed test set must not change when the code does.
``splits/legacy_stratified.json`` freezes the split used for the Phase 1 results.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split


def load_split(path: str | Path) -> dict:
    split = json.loads(Path(path).read_text())
    overlap = set(split["train"]) & set(split["test"])
    if overlap:
        raise ValueError(f"Split {path} has {len(overlap)} files in both train and test")
    return split


def save_split(path: str | Path, train: list[str], test: list[str], **meta) -> None:
    payload = {**meta, "train": sorted(train), "test": sorted(test)}
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def make_stratified_split(originals: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Random split stratified by raga. Kept for new datasets; the legacy split is frozen."""
    train, test = train_test_split(
        originals, test_size=test_size, stratify=originals["raga"], random_state=seed
    )
    return train["filename"].tolist(), test["filename"].tolist()


def apply_split(df: pd.DataFrame, split: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = (set(split["train"]) | set(split["test"])) - set(df["filename"])
    if missing:
        raise KeyError(f"{len(missing)} split files not found, e.g. {sorted(missing)[:3]}")
    train = df[df["filename"].isin(split["train"])].reset_index(drop=True)
    test = df[df["filename"].isin(split["test"])].reset_index(drop=True)
    return train, test


def cv_folds(train: pd.DataFrame, strategy: str, n_folds: int, seed: int):
    """Yield (train_idx, val_idx) over ``train``.

    ``stratified``: shuffled folds stratified by raga (Phase 1 protocol).
    ``blocked``: each raga's clips are consecutive 20 s cuts of one performance, so
    neighbouring clips share far more than the raga. Blocked folds hold out a contiguous
    range of clip indices per raga, so validation clips are never adjacent-in-time to
    training clips from the same block.
    """
    y = train["raga"].to_numpy()
    if strategy == "stratified":
        yield from StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(train, y)
    elif strategy == "blocked":
        groups = take_blocks(train["take"], n_folds)
        yield from GroupKFold(n_folds).split(train, y, groups)
    else:
        raise ValueError(f"Unknown cv_strategy {strategy!r}")


def take_blocks(takes: pd.Series, n_blocks: int) -> np.ndarray:
    if takes.isna().any():
        raise ValueError("Blocked CV needs a clip index ('take') for every recording")
    takes = takes.astype(int).to_numpy()
    width = math.ceil(takes.max() / n_blocks)
    return np.minimum((takes - 1) // width, n_blocks - 1)

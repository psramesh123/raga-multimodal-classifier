from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raga.data import RAGAS
from raga.split import apply_split, cv_folds, load_split, save_split, take_blocks

LEGACY = Path(__file__).parents[1] / "splits" / "legacy_stratified.json"


def test_legacy_split_is_sealed_and_balanced():
    split = load_split(LEGACY)
    assert len(split["train"]) == 132 and len(split["test"]) == 33
    test_ragas = pd.Series(split["test"]).str.split("_").str[0].str.lower().value_counts()
    assert (test_ragas == 3).all() and len(test_ragas) == 11


def test_load_split_rejects_overlap(tmp_path):
    path = tmp_path / "bad.json"
    save_split(path, ["a.wav", "b.wav"], ["b.wav"])
    with pytest.raises(ValueError, match="both train and test"):
        load_split(path)


def test_apply_split_rejects_missing_files():
    df = pd.DataFrame({"filename": ["a.wav"]})
    with pytest.raises(KeyError):
        apply_split(df, {"train": ["a.wav"], "test": ["missing.wav"]})


def _toy_train():
    rows = [
        {"filename": f"{r}_{t:02d}.wav", "raga": r, "take": t} for r in RAGAS for t in range(1, 16)
    ]
    return pd.DataFrame(rows)


def test_take_blocks_are_contiguous():
    blocks = take_blocks(pd.Series(range(1, 16)), 5)
    assert blocks.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]


def test_blocked_cv_never_mixes_a_block_across_train_and_val():
    train = _toy_train()
    blocks = take_blocks(train["take"], 5)
    seen_val = []
    for tr, va in cv_folds(train, "blocked", 5, seed=0):
        assert set(blocks[tr]).isdisjoint(blocks[va])
        assert set(train.loc[va, "raga"]) == set(RAGAS)  # every raga validated each fold
        seen_val.extend(va)
    assert sorted(seen_val) == list(range(len(train)))


def test_stratified_cv_is_deterministic():
    train = _toy_train()
    a = [va.tolist() for _, va in cv_folds(train, "stratified", 5, seed=42)]
    b = [va.tolist() for _, va in cv_folds(train, "stratified", 5, seed=42)]
    assert a == b


def test_unknown_strategy():
    with pytest.raises(ValueError):
        list(cv_folds(_toy_train(), "random", 5, 0))


def test_take_blocks_requires_takes():
    with pytest.raises(ValueError):
        take_blocks(pd.Series([1, np.nan], dtype="Int64"), 2)

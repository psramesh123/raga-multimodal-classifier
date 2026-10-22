import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from raga.config import FeatureConfig
from raga.features import (
    FEATURE_GROUPS,
    build_feature_table,
    extract_features,
    feature_columns,
    select_groups,
)

SR = 22050


def tone(freq: float, seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_feature_vector_shape_matches_phase1():
    feats = extract_features(tone(440.0), FeatureConfig())
    assert len(feats) == 69  # 20 MFCC x2 + 12 chroma x2 + 5 spectral
    assert all(np.isfinite(v) for v in feats.values())


def test_chroma_finds_the_pitch_class():
    # A4 = 440 Hz; chroma bin 10 (1-indexed, C=1) is A
    feats = extract_features(tone(440.0), FeatureConfig())
    means = [feats[f"chroma{i}_mean"] for i in range(1, 13)]
    assert int(np.argmax(means)) + 1 == 10


def test_empty_audio_rejected():
    with pytest.raises(ValueError):
        extract_features(np.zeros(0, dtype=np.float32), FeatureConfig())


@pytest.mark.parametrize("groups, n", [(["mfcc"], 40), (["chroma"], 24), (["spectral"], 5)])
def test_select_groups(groups, n):
    cols = list(extract_features(tone(330.0), FeatureConfig()))
    assert len(select_groups(cols, groups)) == n
    assert len(select_groups(cols, list(FEATURE_GROUPS))) == 69


def test_select_groups_rejects_unknown():
    with pytest.raises(ValueError):
        select_groups(["mfcc1_mean"], ["tempo"])


def test_feature_table_caches(tmp_path):
    root = tmp_path / "audio"
    root.mkdir()
    sf.write(root / "a.wav", tone(440.0), SR)
    recs = pd.DataFrame({"filename": ["a.wav"], "path": ["a.wav"], "raga": ["thodi"], "take": [1]})
    cfg = FeatureConfig(duration=None)

    first = build_feature_table(recs, root, cfg, cache_dir=tmp_path / "cache")
    (root / "a.wav").unlink()  # a second call must not need the audio
    second = build_feature_table(recs, root, cfg, cache_dir=tmp_path / "cache")
    pd.testing.assert_frame_equal(first, second)
    assert len(feature_columns(first)) == 69

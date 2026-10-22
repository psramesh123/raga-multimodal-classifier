from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from raga.data import RAGA_EMOTION, RAGAS, build_manifest, parse_filename

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "fname, raga, aug_type, aug_param, take",
    [
        ("Thodi_sad_08.wav", "thodi", "original", None, 8),
        ("Poorvikalyani_12.wav", "poorvikalyani", "original", None, 12),
        ("Anandabhairavi_blissfulness_01.wav", "anandabhairavi", "original", None, 1),
        ("Bhairavi_devotion_14.wav", "bhairavi", "original", None, 14),
        ("Bowli_melanchony_03.wav", "bhauli", "original", None, 3),
        ("Nilambariraaga_pitch_shift_2.wav", "neelambari", "pitch_shift", 2.0, None),
        ("hamsadwaniraaga2_happy_pitch_shift_-2.wav", "hamsadhwani", "pitch_shift", -2.0, 2),
        ("F_saveri_compassion_time_stretch_1.2.wav", "saveri", "time_stretch", 1.2, None),
        ("F_flute_hamsadwani_happy_time_stretch_1.0.wav", "hamsadhwani", "time_stretch", 1.0, None),
        ("Fpoorvikalyani_serenity_noise_0.01.wav", "poorvikalyani", "noise", 0.01, None),
        ("saveri5_compassion_noise_0.005.wav", "saveri", "noise", 0.005, 5),
        ("Gaanamurtheraaga_devotion_noise_0.01.wav", "gaanamurthe", "noise", 0.01, None),
    ],
)
def test_parse_filename(fname, raga, aug_type, aug_param, take):
    p = parse_filename(fname)
    assert (p.raga, p.aug_type, p.aug_param, p.take) == (raga, aug_type, aug_param, take)


def test_anandabhairavi_not_parsed_as_bhairavi():
    assert parse_filename("Anandabhairavi_blissfulness_02.wav").raga == "anandabhairavi"


def test_unknown_raga_is_none():
    assert parse_filename("Kalyani_01.wav").raga is None


def test_matches_notebook_parser_on_every_dataset_filename():
    """Regression against the labels notebook 01 produced for all 613 files."""
    expected = pd.read_csv(FIXTURES / "notebook_parse_labels.csv")
    for row in expected.itertuples(index=False):
        p = parse_filename(row.filename)
        assert p.raga == row.raga, row.filename
        assert p.aug_type == row.aug_type, row.filename
        if pd.isna(row.aug_param):
            assert p.aug_param is None, row.filename
        else:
            assert p.aug_param == pytest.approx(row.aug_param), row.filename


def test_emotion_map_covers_every_raga():
    assert sorted(RAGA_EMOTION) == RAGAS


def _write_wav(path: Path, sr: int = 8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(sr, dtype=np.float32), sr)


def test_build_manifest_uses_relative_paths(tmp_path):
    _write_wav(tmp_path / "Studio_Audio_split" / "Thodi_sad_01.wav")
    _write_wav(tmp_path / "Data_Augmentation" / "F_saveri_compassion_noise_0.01.wav")
    df = build_manifest(tmp_path)
    assert set(df["path"]) == {
        "Studio_Audio_split/Thodi_sad_01.wav",
        "Data_Augmentation/F_saveri_compassion_noise_0.01.wav",
    }
    assert dict(zip(df["raga"], df["emotion"], strict=True)) == {
        "thodi": "sad",
        "saveri": "compassion",
    }


def test_build_manifest_fails_loudly_on_unparsed_file(tmp_path):
    _write_wav(tmp_path / "Kalyani_01.wav")
    with pytest.raises(ValueError, match="did not parse"):
        build_manifest(tmp_path)

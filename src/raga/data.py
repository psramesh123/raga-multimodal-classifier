"""Filename parsing and the dataset manifest (from notebooks/01_data_inventory)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

# Canonical raga -> spellings seen in the dataset filenames (lowercase).
RAGA_ALIASES: dict[str, list[str]] = {
    "anandabhairavi": ["anandabhairavi"],
    "poorvikalyani": ["poorvikalyani"],
    "hamsadhwani": ["hamsadhwani", "hamsadwani"],
    "gaanamurthe": ["gaanamurthe", "ganamurthe"],
    "dwijavanthi": ["dwijavanthi", "dhwijavanthi"],
    "neelambari": ["neelambari", "nilambari"],
    "bhairavi": ["bhairavi"],
    "bilahari": ["bilahari"],
    "saveri": ["saveri"],
    "bhauli": ["bhauli", "bowli", "bhowli"],
    "thodi": ["thodi"],
}
RAGAS: list[str] = sorted(RAGA_ALIASES)

# Emotion is a fixed function of raga in this dataset, not an independent label.
RAGA_EMOTION: dict[str, str] = {
    "anandabhairavi": "blissfulness",
    "bhairavi": "devotion",
    "bhauli": "melancholy",
    "bilahari": "tranquility",
    "dwijavanthi": "calm",
    "gaanamurthe": "devotion",
    "hamsadhwani": "happy",
    "neelambari": "relaxation",
    "poorvikalyani": "serenity",
    "saveri": "compassion",
    "thodi": "sad",
}

AUG_TYPES = ("pitch_shift", "time_stretch", "noise")

# Longest alias first so "anandabhairavi" wins over "bhairavi".
_ALIAS_MAP = sorted(
    ((alias, canon) for canon, aliases in RAGA_ALIASES.items() for alias in aliases),
    key=lambda pair: len(pair[0]),
    reverse=True,
)


@dataclass(frozen=True)
class ParsedName:
    raga: str | None
    aug_type: str
    aug_param: float | None
    take: int | None  # clip index for originals (e.g. Thodi_sad_08 -> 8); source take for augs


def parse_filename(fname: str) -> ParsedName:
    """Recover raga, augmentation, and clip index from a RaagaDhvani filename.

    Handles the spelling variants (Bowli/Bhauli, Nilambari/Neelambari, ...) and the
    inconsistent flute prefixes (``F_``, ``F``, ``F_flute_``) found in the dataset.
    """
    s = Path(fname).stem.lower().strip()
    s = re.sub(r"^f_?(?:flute_?)?", "", s)

    raga = None
    for alias, canon in _ALIAS_MAP:
        if s.startswith(alias):
            raga = canon
            s = s[len(alias) :]
            break
    s = re.sub(r"^(raaga|raga)_?", "", s)

    aug_type, aug_param = "original", None
    for a in AUG_TYPES:
        m = re.search(a + r"_(-?\d+(?:\.\d+)?)", s)
        if m:
            aug_type, aug_param = a, float(m.group(1))
            s = s[: m.start()].rstrip("_")
            break

    # Originals end in a clip index ("_08"); some augmentations carry a take number
    # straight after the raga name ("saveri5_compassion").
    m = re.search(r"_?(\d+)$", s) or re.match(r"^(\d+)", s)
    take = int(m.group(1)) if m else None

    return ParsedName(raga=raga, aug_type=aug_type, aug_param=aug_param, take=take)


def build_manifest(root: Path) -> pd.DataFrame:
    """One row per .wav under ``root`` with parsed labels.

    ``path`` is relative to ``root`` so the manifest is portable across machines.
    Raises if any file fails to parse, rather than silently dropping it.
    """
    rows = []
    for wav in sorted(root.rglob("*.wav")):
        parsed = parse_filename(wav.name)
        rows.append(
            {
                "filename": wav.name,
                "folder": wav.parent.name,
                "path": str(wav.relative_to(root)),
                **asdict(parsed),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No .wav files under {root}")

    unparsed = df.loc[df["raga"].isna(), "filename"].tolist()
    if unparsed:
        raise ValueError(f"{len(unparsed)} filenames did not parse, e.g. {unparsed[:5]}")
    if df["filename"].duplicated().any():
        dupes = df.loc[df["filename"].duplicated(), "filename"].tolist()
        raise ValueError(f"Duplicate filenames break split bookkeeping: {dupes[:5]}")

    df["emotion"] = df["raga"].map(RAGA_EMOTION)
    df["take"] = df["take"].astype("Int64")
    return df


def originals(manifest: pd.DataFrame) -> pd.DataFrame:
    return manifest[manifest["aug_type"] == "original"].reset_index(drop=True)

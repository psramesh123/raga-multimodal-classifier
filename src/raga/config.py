"""Run configuration: one YAML file per experiment, plus the data location."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DATA_DIR_ENV = "RAGA_DATA_DIR"


@dataclass
class FeatureConfig:
    sr: int = 22050
    duration: float | None = 60.0  # seconds loaded per recording; None = whole file
    n_mfcc: int = 20
    # Feature families to keep: any of "mfcc", "chroma", "spectral".
    # Restricting this is how the timbre-vs-tonal ablation is run.
    groups: list[str] = field(default_factory=lambda: ["mfcc", "chroma", "spectral"])


@dataclass
class ModelConfig:
    name: str = "logreg"  # logreg | random_forest | hist_gb | dummy
    params: dict = field(default_factory=dict)


@dataclass
class EvalConfig:
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # stratified | blocked
    seed: int = 42
    n_bootstrap: int = 2000


@dataclass
class Config:
    name: str = "baseline"
    split: str = "splits/legacy_stratified.json"
    artifacts_dir: str = "artifacts"
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            name=raw.get("name", Path(path).stem),
            split=raw.get("split", cls.split),
            artifacts_dir=raw.get("artifacts_dir", cls.artifacts_dir),
            features=FeatureConfig(**raw.get("features", {})),
            model=ModelConfig(**raw.get("model", {})),
            eval=EvalConfig(**raw.get("eval", {})),
        )


def data_dir(override: str | Path | None = None) -> Path:
    """Dataset root (the folder that contains Readme.txt and the audio subfolders)."""
    root = override or os.environ.get(DATA_DIR_ENV)
    if not root:
        raise RuntimeError(
            f"Set {DATA_DIR_ENV} to the RaagaDhvani dataset root, e.g.\n"
            f'  export {DATA_DIR_ENV}="data/RaagaDhvani-A novel curated Augmented Carnatic mus/'
            'RaagaDhvani_Dataset_VidushiArchana"'
        )
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"{DATA_DIR_ENV} points at {root}, which is not a directory")
    return root

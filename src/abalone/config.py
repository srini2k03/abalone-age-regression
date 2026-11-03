"""Typed experiment configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"


class DataConfig(BaseModel):
    source: Literal["uci", "kaggle"] = "uci"
    cache_dir: Path = Path("data/raw")
    urls: list[str] = Field(default_factory=lambda: [UCI_URL])
    expected_fingerprint: str | None = None
    kaggle_train_path: Path = Path("data/kaggle/train.csv")
    kaggle_test_path: Path = Path("data/kaggle/test.csv")
    holdout_size: float = Field(0.2, ge=0.0, lt=1.0)


class FeatureConfig(BaseModel):
    clip_outliers: bool = True
    clip_quantiles: tuple[float, float] = (0.001, 0.999)
    add_ratios: bool = True
    add_volume: bool = True
    add_log_weights: bool = True

    @field_validator("clip_quantiles")
    @classmethod
    def _check_quantiles(cls, v: tuple[float, float]) -> tuple[float, float]:
        lo, hi = v
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError("clip_quantiles must satisfy 0 <= low < high <= 1")
        return v


class CVConfig(BaseModel):
    n_splits: int = Field(5, ge=2)
    n_repeats: int = Field(2, ge=1)
    n_jobs: int = -1


class ModelsConfig(BaseModel):
    enabled: list[str] = Field(
        default_factory=lambda: ["baseline", "linear", "ridge", "random_forest", "xgboost", "catboost"]
    )
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    ensemble_members: list[str] = Field(
        default_factory=lambda: ["ridge", "random_forest", "xgboost", "catboost"]
    )


class TrackingConfig(BaseModel):
    backend: Literal["json", "mlflow"] = "json"
    dir: Path = Path("runs")


class ArtifactsConfig(BaseModel):
    dir: Path = Path("models")
    model_filename: str = "model.joblib"

    @property
    def model_path(self) -> Path:
        return self.dir / self.model_filename


class Config(BaseModel):
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load a :class:`Config` from YAML (falls back to defaults if no file exists)."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if path.exists():
        with path.open() as fh:
            raw = yaml.safe_load(fh) or {}
    elif path != DEFAULT_CONFIG_PATH:
        raise FileNotFoundError(f"Config file not found: {path}")
    if overrides:
        raw = _deep_merge(raw, overrides)
    return Config.model_validate(raw)


def _deep_merge(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out

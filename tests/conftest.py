from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abalone.config import Config, load_config


def make_synthetic(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Synthetic abalone-like table with a realistic size -> rings relationship."""
    rng = np.random.default_rng(seed)
    sex = rng.choice(["M", "F", "I"], size=n, p=[0.37, 0.31, 0.32])
    length = rng.uniform(0.1, 0.8, n)
    diameter = length * rng.uniform(0.75, 0.82, n)
    height = np.clip(length * rng.uniform(0.25, 0.4, n), 0.0, None)
    whole = 1.8 * length**3 * rng.uniform(0.85, 1.15, n) * 4
    shucked = whole * rng.uniform(0.35, 0.5, n)
    viscera = whole * rng.uniform(0.18, 0.25, n)
    shell = whole * rng.uniform(0.25, 0.33, n)
    log_rings = 1.4 + 1.6 * shell + 0.8 * length + (sex == "I") * -0.2 + rng.normal(0, 0.12, n)
    rings = np.clip(np.round(np.expm1(log_rings)), 1, 29).astype(int)
    return pd.DataFrame(
        {
            "sex": sex,
            "length": length,
            "diameter": diameter,
            "height": height,
            "whole_weight": whole,
            "shucked_weight": shucked,
            "viscera_weight": viscera,
            "shell_weight": shell,
            "rings": rings,
        }
    )


@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    return make_synthetic()


@pytest.fixture()
def small_cfg(tmp_path) -> Config:
    """Fast config: 3-fold x 1 repeat, tiny boosters, everything written under tmp_path."""
    return load_config(
        overrides={
            "cv": {"n_splits": 3, "n_repeats": 1, "n_jobs": 1},
            "data": {"cache_dir": str(tmp_path / "raw"), "expected_fingerprint": None},
            "tracking": {"dir": str(tmp_path / "runs")},
            "artifacts": {"dir": str(tmp_path / "models")},
            "models": {
                "params": {
                    "random_forest": {"n_estimators": 30, "min_samples_leaf": 4, "max_features": 0.5},
                    "xgboost": {"n_estimators": 60, "learning_rate": 0.1, "max_depth": 3},
                    "lightgbm": {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 7},
                    "catboost": {"iterations": 60, "learning_rate": 0.1, "depth": 4},
                },
                "ensemble_members": ["ridge", "xgboost", "catboost"],
            },
        }
    )


@pytest.fixture()
def synthetic_csv(tmp_path, synthetic_df) -> str:
    path = tmp_path / "train.csv"
    synthetic_df.to_csv(path, index=False)
    return str(path)

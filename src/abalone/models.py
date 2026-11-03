"""Model zoo. Every model is trained on ``log1p(rings)`` so that minimising
squared error in model space directly minimises RMSLE in ring space."""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from abalone.config import Config

logger = logging.getLogger(__name__)

BASE_MODELS = ["baseline", "linear", "ridge", "random_forest", "xgboost", "lightgbm", "catboost"]
ENSEMBLES = ["stacking", "blend"]
ALL_MODELS = BASE_MODELS + ENSEMBLES

_OPTIONAL_DEPS = {"xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost"}
_SCALED = {"linear", "ridge"}


def is_available(name: str) -> bool:
    module = _OPTIONAL_DEPS.get(name)
    return module is None or importlib.util.find_spec(module) is not None


def resolve_models(names: list[str]) -> list[str]:
    """Validate names and drop models whose optional dependency is missing."""
    unknown = sorted(set(names) - set(ALL_MODELS))
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}; choose from {ALL_MODELS}")
    out = []
    for name in names:
        if is_available(name):
            out.append(name)
        else:
            logger.warning("Skipping %s: optional dependency %r not installed", name, _OPTIONAL_DEPS[name])
    return out


def _regressor(name: str, params: dict[str, Any], seed: int, n_jobs: int) -> RegressorMixin:
    if name == "linear":
        return LinearRegression(**params)
    if name == "ridge":
        return Ridge(**{"alpha": 1.0, **params})
    if name == "random_forest":
        return RandomForestRegressor(**{"n_estimators": 300, **params}, random_state=seed, n_jobs=n_jobs)
    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            **{"n_estimators": 600, "learning_rate": 0.03, "max_depth": 4, **params},
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=n_jobs,
        )
    if name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            **{"n_estimators": 600, "learning_rate": 0.03, **params},
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=-1,
        )
    if name == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            **{"iterations": 800, "learning_rate": 0.05, "depth": 6, **params},
            loss_function="RMSE",
            random_seed=seed,
            thread_count=n_jobs,
            verbose=0,
            allow_writing_files=False,
        )
    raise ValueError(f"No base regressor named {name!r}")


def build_log_space_pipeline(
    name: str, cfg: Config, params: dict[str, Any] | None = None, n_jobs: int = 1
) -> Pipeline | RegressorMixin:
    """Preprocessing + regressor operating on the log1p target (no target transform)."""
    from abalone.features import build_preprocessor

    if name == "baseline":
        return DummyRegressor(strategy="mean")
    params = {**cfg.models.params.get(name, {}), **(params or {})}
    return Pipeline(
        [
            ("preprocess", build_preprocessor(cfg.features, scale=name in _SCALED)),
            ("model", _regressor(name, params, cfg.seed, n_jobs)),
        ]
    )


def build_model(
    name: str, cfg: Config, params: dict[str, Any] | None = None, n_jobs: int = 1
) -> TransformedTargetRegressor:
    """Return an estimator that takes raw features and predicts rings."""
    if name in ENSEMBLES:
        members = [m for m in resolve_models(cfg.models.ensemble_members) if m not in ENSEMBLES]
        if len(members) < 2:
            raise ValueError("Ensembles need at least two available members")
        estimators = [(m, build_log_space_pipeline(m, cfg, n_jobs=n_jobs)) for m in members]
        if name == "stacking":
            inner: RegressorMixin = StackingRegressor(
                estimators=estimators,
                final_estimator=LinearRegression(positive=True),
                cv=5,
                n_jobs=1,
            )
        else:
            inner = VotingRegressor(estimators=estimators, n_jobs=1)
    else:
        inner = build_log_space_pipeline(name, cfg, params=params, n_jobs=n_jobs)
    return TransformedTargetRegressor(
        regressor=inner, func=np.log1p, inverse_func=np.expm1, check_inverse=False
    )

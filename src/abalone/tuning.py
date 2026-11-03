"""Optuna hyperparameter search (optional dependency: ``pip install optuna``)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

from abalone.config import Config
from abalone.metrics import rmsle_scorer
from abalone.models import build_model

logger = logging.getLogger(__name__)

SearchSpace = Callable[[Any], dict[str, Any]]

SEARCH_SPACES: dict[str, SearchSpace] = {
    "ridge": lambda t: {"alpha": t.suggest_float("alpha", 1e-3, 100.0, log=True)},
    "random_forest": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 200, 600, step=100),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 20),
        "max_features": t.suggest_float("max_features", 0.3, 1.0),
    },
    "xgboost": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 200, 1200, step=100),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": t.suggest_int("max_depth", 2, 8),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": t.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    },
    "lightgbm": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 200, 1200, step=100),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": t.suggest_int("num_leaves", 4, 64, log=True),
        "min_child_samples": t.suggest_int("min_child_samples", 5, 60),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    },
    "catboost": lambda t: {
        "iterations": t.suggest_int("iterations", 300, 1500, step=100),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": t.suggest_int("depth", 3, 8),
        "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
    },
}


def tune(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Config,
    n_trials: int = 30,
    timeout: float | None = None,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Minimise K-fold CV RMSLE for ``name``; returns best params and score."""
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Tuning requires optuna: pip install 'abalone-age-regression[tune]'") from exc
    if name not in SEARCH_SPACES:
        raise ValueError(f"No search space for {name!r}; choose from {sorted(SEARCH_SPACES)}")

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)

    def objective(trial: optuna.Trial) -> float:
        params = SEARCH_SPACES[name](trial)
        model = build_model(name, cfg, params=params, n_jobs=1)
        scores = cross_val_score(model, X, y, cv=cv, scoring=rmsle_scorer, n_jobs=cfg.cv.n_jobs)
        return float(-np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        study_name=f"abalone-{name}",
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    best = {k: v for k, v in study.best_params.items()}
    if name == "lightgbm":
        best["subsample_freq"] = 1
    logger.info("Best %s RMSLE %.4f with %s", name, study.best_value, best)
    return {
        "model": name,
        "best_rmsle": float(study.best_value),
        "best_params": best,
        "n_trials": len(study.trials),
    }

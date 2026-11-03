"""Repeated K-fold cross-validation for model comparison."""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_validate

from abalone.config import Config
from abalone.metrics import SCORING
from abalone.models import build_model

logger = logging.getLogger(__name__)


def make_cv(cfg: Config) -> RepeatedKFold:
    return RepeatedKFold(n_splits=cfg.cv.n_splits, n_repeats=cfg.cv.n_repeats, random_state=cfg.seed)


def cross_validate_model(name: str, X: pd.DataFrame, y: pd.Series, cfg: Config, params=None) -> dict:
    """Cross-validate one model; returns per-fold and aggregated metrics."""
    start = time.perf_counter()
    scores = cross_validate(
        build_model(name, cfg, params=params, n_jobs=1),
        X,
        y,
        cv=make_cv(cfg),
        scoring=SCORING,
        n_jobs=cfg.cv.n_jobs,
        error_score="raise",
    )
    folds = {
        "rmsle": -scores["test_rmsle"],
        "rmse": -scores["test_rmse"],
        "mae": -scores["test_mae"],
        "r2": scores["test_r2"],
    }
    result = {"model": name, "n_folds": len(folds["rmsle"])}
    for metric, values in folds.items():
        result[f"{metric}_mean"] = float(np.mean(values))
        result[f"{metric}_std"] = float(np.std(values))
    result["fit_time_s"] = float(np.mean(scores["fit_time"]))
    result["wall_time_s"] = float(time.perf_counter() - start)
    result["fold_rmsle"] = [float(v) for v in folds["rmsle"]]
    logger.info("%-14s RMSLE %.4f +/- %.4f", name, result["rmsle_mean"], result["rmsle_std"])
    return result


def compare_models(names: list[str], X: pd.DataFrame, y: pd.Series, cfg: Config) -> pd.DataFrame:
    """Cross-validate several models and return a leaderboard sorted by RMSLE."""
    rows = [cross_validate_model(name, X, y, cfg) for name in names]
    board = pd.DataFrame(rows).sort_values("rmsle_mean").reset_index(drop=True)
    if "baseline" in set(board["model"]):
        base = float(board.loc[board["model"] == "baseline", "rmsle_mean"].iloc[0])
        board["improvement_vs_baseline_pct"] = 100.0 * (base - board["rmsle_mean"]) / base
    if "linear" in set(board["model"]):
        lin = float(board.loc[board["model"] == "linear", "rmsle_mean"].iloc[0])
        board["improvement_vs_linear_pct"] = 100.0 * (lin - board["rmsle_mean"]) / lin
    return board

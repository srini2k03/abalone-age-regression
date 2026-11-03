"""Evaluation metrics (RMSLE is the competition metric)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score


def rmsle(y_true, y_pred) -> float:
    """Root mean squared logarithmic error; negative predictions are clipped to 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    if np.any(y_true < 0):
        raise ValueError("RMSLE is undefined for negative targets")
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def regression_report(y_true, y_pred) -> dict[str, float]:
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    return {
        "rmsle": rmsle(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

SCORING = {
    "rmsle": rmsle_scorer,
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}

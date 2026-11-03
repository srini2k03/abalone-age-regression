"""End-to-end orchestration used by the CLI: train / evaluate / predict."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from abalone import RING_TO_AGE_OFFSET
from abalone.config import Config
from abalone.data import (
    FEATURES,
    TARGET,
    UCI_COLUMNS,
    frame_fingerprint,
    holdout_split,
    load_dataset,
    split_xy,
)
from abalone.evaluation import compare_models
from abalone.metrics import regression_report
from abalone.models import build_model, resolve_models
from abalone.persistence import load_model, save_model
from abalone.tracking import make_tracker
from abalone.validation import validate

logger = logging.getLogger(__name__)


def _with_params(cfg: Config, params_file: str | Path | None) -> Config:
    if params_file is None:
        return cfg
    tuned = json.loads(Path(params_file).read_text())
    if "model" in tuned and "best_params" in tuned:  # output of `abalone tune`
        tuned = {tuned["model"]: tuned["best_params"]}
    merged = {**cfg.models.params}
    for name, params in tuned.items():
        merged[name] = {**merged.get(name, {}), **params}
    return cfg.model_copy(update={"models": cfg.models.model_copy(update={"params": merged})})


def prepare_data(cfg: Config, data_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = load_dataset(cfg.data, data_path)
    report = validate(df, require_target=True)
    if any(report.warnings.values()):
        logger.info("Data anomalies (handled downstream): %s", report.warnings)
    info: dict[str, Any] = {
        "n_rows": len(df),
        "anomalies": report.warnings,
        "source": str(data_path) if data_path else cfg.data.source,
    }
    if set(UCI_COLUMNS).issubset(df.columns):
        info["fingerprint"] = frame_fingerprint(df)
    train_df, test_df = holdout_split(df, cfg.data.holdout_size, cfg.seed)
    info.update(n_train=len(train_df), n_holdout=len(test_df))
    return train_df, test_df, info


def train(
    cfg: Config,
    data_path: str | Path | None = None,
    models: list[str] | None = None,
    params_file: str | Path | None = None,
    select: str | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Cross-validate candidate models, refit the best one and evaluate on the hold-out."""
    cfg = _with_params(cfg, params_file)
    train_df, test_df, data_info = prepare_data(cfg, data_path)
    X, y = split_xy(train_df)
    names = resolve_models(models or cfg.models.enabled)

    tracker = make_tracker(cfg.tracking.backend, cfg.tracking.dir, run_name="train")
    tracker.log_params(
        {
            "seed": cfg.seed,
            "models": names,
            "cv": cfg.cv.model_dump(),
            "features": cfg.features.model_dump(),
            "model_params": cfg.models.params,
            "data": data_info,
        }
    )

    board = compare_models(names, X, y, cfg)
    tracker.log_table("leaderboard", board.drop(columns=["fold_rmsle"]))

    candidates = board[board["model"] != "baseline"]
    best_name = select or str((candidates if len(candidates) else board).iloc[0]["model"])
    best_row = board[board["model"] == best_name].iloc[0]
    logger.info("Refitting %s on %d rows", best_name, len(X))
    final = build_model(best_name, cfg, n_jobs=-1).fit(X, y)

    holdout: dict[str, Any] = {}
    if len(test_df):
        Xt, yt = split_xy(test_df)
        holdout = regression_report(yt, final.predict(Xt))
        base = build_model("baseline", cfg).fit(X, y)
        base_rmsle = regression_report(yt, base.predict(Xt))["rmsle"]
        holdout["baseline_rmsle"] = base_rmsle
        holdout["improvement_vs_baseline_pct"] = 100.0 * (base_rmsle - holdout["rmsle"]) / base_rmsle

    metadata = {
        "model_name": best_name,
        "cv": {
            k: float(best_row[k]) for k in ("rmsle_mean", "rmsle_std", "rmse_mean", "mae_mean", "r2_mean")
        },
        "cv_scheme": f"RepeatedKFold({cfg.cv.n_splits}x{cfg.cv.n_repeats}, seed={cfg.seed})",
        "holdout": holdout,
        "data": data_info,
        "run_id": tracker.run_id,
    }
    model_path = Path(output) if output else cfg.artifacts.model_path
    save_model(final, model_path, metadata)
    tracker.log_metrics(
        {
            "best_model": best_name,
            **{f"cv_{k}": v for k, v in metadata["cv"].items()},
            **{f"holdout_{k}": v for k, v in holdout.items()},
        }
    )
    tracker.end()
    return {
        "leaderboard": board,
        "best_model": best_name,
        "holdout": holdout,
        "model_path": str(model_path),
        "run_dir": str(tracker.dir),
        "data": data_info,
    }


def evaluate(
    cfg: Config, model_path: str | Path, data_path: str | Path | None = None, full: bool = False
) -> dict[str, Any]:
    """Score a saved model on the configured hold-out split (or on a whole labelled CSV)."""
    model, meta = load_model(model_path)
    if data_path is not None and full:
        df = load_dataset(cfg.data, data_path)
        validate(df, require_target=True)
    else:
        _, df, _ = prepare_data(cfg, data_path)
    X, y = split_xy(df)
    report = regression_report(y, model.predict(X))
    report.update(n_rows=len(df), model_name=meta.get("model_name"))
    return report


def predict_frame(model, df: pd.DataFrame) -> pd.DataFrame:
    validate(df, require_target=False)
    rings = np.clip(model.predict(df[FEATURES]), 1.0, None)
    out = pd.DataFrame({"predicted_rings": rings, "predicted_age_years": rings + RING_TO_AGE_OFFSET})
    if "id" in df.columns:
        out.insert(0, "id", df["id"].to_numpy())
    return out


def predict_file(
    model_path: str | Path, input_path: str | Path, output_path: str | Path, kaggle_format: bool = False
) -> pd.DataFrame:
    from abalone.data import load_csv, parse_uci

    model, _ = load_model(model_path)
    p = Path(input_path)
    df = parse_uci(p) if p.suffix == ".data" else load_csv(p)
    preds = predict_frame(model, df.drop(columns=[TARGET], errors="ignore"))
    if kaggle_format:  # Kaggle S4E4 submission: id,Rings
        if "id" not in preds.columns:
            raise ValueError("--kaggle output needs an 'id' column in the input file")
        preds = pd.DataFrame({"id": preds["id"], "Rings": preds["predicted_rings"]})
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output_path, index=False)
    return preds

"""Train the compact web model and export everything the static site needs.

The model is the project's own XGBoost pipeline (QuantileClipper -> MorphometricFeatures ->
one-hot sex -> XGBRegressor on log1p(rings)) with the Optuna-tuned hyperparameters reported in
the README (300 trees of depth 4). It is fitted on the 80% training split only, so the reported
hold-out metrics are on rows the model never saw.

Outputs (all under ``web/data/``):

* ``model.json``     - clip bounds, feature order and the 300 trees as compact arrays,
                       evaluated in the browser by ``web/js/model.js``.
* ``insights.json``  - hold-out / CV metrics, SHAP importance, presets (real UCI rows),
                       slider ranges and the ring histogram.
* ``parity-fixture.json`` - ~200 rows with Python predictions; ``scripts/check_web_parity.mjs``
                       checks the JavaScript evaluator reproduces them.

Usage::

    python scripts/export_web_model.py            # train, export, run the JS parity check
    python scripts/export_web_model.py --no-cv    # skip the 5x2 cross-validation
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from abalone import RING_TO_AGE_OFFSET
from abalone.config import load_config
from abalone.data import (
    CATEGORICAL,
    NUMERIC,
    TARGET,
    frame_fingerprint,
    holdout_split,
    load_dataset,
    split_xy,
)
from abalone.evaluation import cross_validate_model
from abalone.features import EPS
from abalone.metrics import regression_report
from abalone.models import build_model

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "web" / "data"

# Result of `abalone tune --model xgboost --n-trials 10` (see README "Tuning").
WEB_MODEL_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.023993242906812727,
    "max_depth": 4,
    "subsample": 0.728034992108518,
    "colsample_bytree": 0.8711055768358081,
    "min_child_weight": 1.8187859051288213,
    "reg_lambda": 0.11400863701127326,
}

logger = logging.getLogger("export_web_model")


def _parse_base_score(raw: str | float) -> float:
    """XGBoost >= 3 stores base_score as a vector string such as ``"[2.2618E0]"``."""
    if isinstance(raw, (int, float)):
        return float(raw)
    return float(str(raw).strip("[]").split(",")[0])


def export_trees(booster, feature_names: list[str]) -> dict:
    """Compact JSON representation of an XGBoost regression booster."""
    dump = json.loads(booster.save_raw("json"))
    learner = dump["learner"]
    objective = learner["objective"]["name"]
    if objective != "reg:squarederror":
        raise ValueError(f"Unsupported objective {objective!r}")
    if learner["feature_names"] and list(learner["feature_names"]) != list(feature_names):
        raise ValueError("Booster feature order differs from the preprocessor output")
    trees = []
    for tree in learner["gradient_booster"]["model"]["trees"]:
        left = tree["left_children"]
        trees.append(
            {
                # split feature index, or -1 for a leaf
                "f": [-1 if left[i] == -1 else int(s) for i, s in enumerate(tree["split_indices"])],
                # split threshold (go left when x < t), or the leaf value
                "t": [float(v) for v in tree["split_conditions"]],
                "l": [int(v) for v in left],
                "r": [int(v) for v in tree["right_children"]],
                # missing-value direction (inputs are never missing in the UI, kept for completeness)
                "d": [int(v) for v in tree["default_left"]],
            }
        )
    return {
        "base_score": _parse_base_score(learner["learner_model_param"]["base_score"]),
        "trees": trees,
    }


def _round(x: float, nd: int = 6) -> float:
    return float(round(float(x), nd))


def _pick_presets(holdout: pd.DataFrame, preds: np.ndarray) -> list[dict]:
    """A handful of real, unseen UCI rows spanning infants to old adults."""
    df = holdout.assign(pred=preds)
    wanted = [
        ("Juvenile (infant)", (df["sex"] == "I") & df[TARGET].between(5, 6)),
        ("Young male", (df["sex"] == "M") & (df[TARGET] == 9)),
        ("Adult female", (df["sex"] == "F") & (df[TARGET] == 11)),
        ("Mature male", (df["sex"] == "M") & (df[TARGET] == 14)),
        ("Old female", (df["sex"] == "F") & (df[TARGET] >= 18)),
    ]
    presets = []
    for label, mask in wanted:
        cand = df[mask]
        if cand.empty:
            continue
        # first match in the (shuffled) hold-out order: no selection on prediction quality
        row = cand.iloc[0]
        presets.append(
            {
                "label": label,
                "uci_row": int(row["uci_row"]),
                "sex": row["sex"],
                **{c: float(row[c]) for c in NUMERIC},
                "rings": int(row[TARGET]),
                "python_pred_rings": _round(row["pred"]),
            }
        )
    return presets


def _parity_rows(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """~200 rows: random real rows, every anomalous row, and a few synthetic out-of-range ones."""
    rng = np.random.default_rng(seed)
    anomalous = df[
        (df["height"] <= 0)
        | (df["shucked_weight"] > df["whole_weight"])
        | (df["shell_weight"] > df["whole_weight"])
    ]
    rest = df.drop(index=anomalous.index)
    random_rows = rest.iloc[rng.choice(len(rest), size=185, replace=False)]
    synthetic = pd.DataFrame(
        [
            # beyond the clip bounds on both sides, zero weights, extreme ratios
            {"sex": "M", "length": 0.9, "diameter": 0.75, "height": 1.2, "whole_weight": 3.5,
             "shucked_weight": 1.8, "viscera_weight": 0.9, "shell_weight": 1.2},
            {"sex": "I", "length": 0.05, "diameter": 0.04, "height": 0.0, "whole_weight": 0.0,
             "shucked_weight": 0.0, "viscera_weight": 0.0, "shell_weight": 0.0},
            {"sex": "F", "length": 0.5, "diameter": 0.4, "height": 0.13, "whole_weight": 0.2,
             "shucked_weight": 0.5, "viscera_weight": 0.2, "shell_weight": 0.3},
            {"sex": "I", "length": 0.3, "diameter": 0.22, "height": 0.07, "whole_weight": 0.15,
             "shucked_weight": 0.06, "viscera_weight": 0.03, "shell_weight": 0.0},
            {"sex": "M", "length": 0.62, "diameter": 0.49, "height": 0.17, "whole_weight": 1.2,
             "shucked_weight": 0.5, "viscera_weight": 0.26, "shell_weight": 0.36},
        ]
    )  # fmt: skip
    synthetic["uci_row"] = -1
    synthetic[TARGET] = np.nan
    return pd.concat([random_rows, anomalous, synthetic], ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-c", "--config", default=None, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--no-cv", action="store_true", help="Skip the 5x2 repeated K-fold CV")
    parser.add_argument("--no-parity", action="store_true", help="Skip the Node.js parity check")
    parser.add_argument("--out", default=str(OUT_DIR), help="Output directory (default: web/data)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(cfg.data)
    df.insert(0, "uci_row", np.arange(1, len(df) + 1))  # 1-based line number in abalone.data
    train_df, test_df = holdout_split(df, cfg.data.holdout_size, cfg.seed)
    X, y = split_xy(train_df)
    Xt, yt = split_xy(test_df)

    cv = None
    if not args.no_cv:
        logger.info("Cross-validating the web model (%s)", cfg.cv)
        res = cross_validate_model("xgboost", X, y, cfg, params=WEB_MODEL_PARAMS)
        cv = {k: _round(res[k]) for k in ("rmsle_mean", "rmsle_std", "rmse_mean", "mae_mean", "r2_mean")}
        cv["scheme"] = f"RepeatedKFold({cfg.cv.n_splits}x{cfg.cv.n_repeats}, seed={cfg.seed})"

    model = build_model("xgboost", cfg, params=WEB_MODEL_PARAMS, n_jobs=-1).fit(X, y)
    pipe = model.regressor_
    pre = pipe.named_steps["preprocess"]
    xgb = pipe.named_steps["model"]
    feature_names = [str(c) for c in pre.get_feature_names_out()]

    clipper = pre.named_steps["clip"]
    encoder = pre.named_steps["encode"].named_transformers_["sex"]
    morph = pre.named_steps["features"]
    if not (morph.add_ratios and morph.add_volume and morph.add_log_weights):
        raise SystemExit("The JS feature code implements the default (all features on) configuration")

    test_pred = np.clip(model.predict(Xt), 1.0, None)
    holdout = {k: _round(v) for k, v in regression_report(yt, test_pred).items()}
    base_pred = build_model("baseline", cfg).fit(X, y).predict(Xt)
    holdout["baseline_rmsle"] = _round(regression_report(yt, base_pred)["rmsle"])
    holdout["n_rows"] = len(test_df)
    logger.info("Hold-out RMSLE %.4f (baseline %.4f)", holdout["rmsle"], holdout["baseline_rmsle"])

    spec = {
        "format": "abalone-xgb-trees/1",
        "description": "XGBoost regressor on log1p(rings); prediction = expm1(base_score + sum(leaves))",
        "input_columns": CATEGORICAL + NUMERIC,
        "clip": {
            "columns": list(clipper.columns_),
            "lower": [float(clipper.lower_[c]) for c in clipper.columns_],
            "upper": [float(clipper.upper_[c]) for c in clipper.columns_],
        },
        "eps": EPS,
        "sex_categories": [str(c) for c in encoder.categories_[0]],
        "feature_names": feature_names,
        "min_rings": 1.0,
        "age_offset": RING_TO_AGE_OFFSET,
        **export_trees(xgb.get_booster(), feature_names),
    }
    model_path = out_dir / "model.json"
    model_path.write_text(json.dumps(spec, separators=(",", ":")))
    logger.info(
        "Wrote %s (%.0f kB, %d trees)", model_path, model_path.stat().st_size / 1024, len(spec["trees"])
    )

    from abalone.explain import shap_importance

    shap_table, _ = shap_importance(model, X, max_samples=1000, seed=cfg.seed)
    ring_counts = df[TARGET].value_counts().sort_index()
    insights = {
        "model": {
            "name": "XGBoost (tuned)",
            "params": WEB_MODEL_PARAMS,
            "n_trees": len(spec["trees"]),
            "n_features": len(feature_names),
            "trained_on_rows": len(train_df),
            "model_json_bytes": model_path.stat().st_size,
        },
        "data": {
            "n_rows": len(df),
            "fingerprint": frame_fingerprint(df),
            "n_train": len(train_df),
            "n_holdout": len(test_df),
        },
        "holdout": holdout,
        "cv": cv,
        "shap": {
            "sample_rows": int(min(1000, len(X))),
            "space": "log1p(rings)",
            "values": [
                {"feature": r.feature, "mean_abs_shap": _round(r.mean_abs_shap)}
                for r in shap_table.itertuples()
            ],
        },
        "ranges": {
            c: {"min": float(df[c].min()), "max": float(df[c].max()), "median": float(df[c].median())}
            for c in NUMERIC
        },
        "ring_histogram": {str(int(k)): int(v) for k, v in ring_counts.items()},
        "presets": _pick_presets(test_df, test_pred),
    }
    (out_dir / "insights.json").write_text(json.dumps(insights, indent=1))

    parity = _parity_rows(df, cfg.seed)
    margin = model.regressor_.predict(parity[CATEGORICAL + NUMERIC])
    rings = np.clip(model.predict(parity[CATEGORICAL + NUMERIC]), 1.0, None)
    fixture = {
        "tolerance": 1e-4,
        "rows": [
            {
                "sex": r.sex,
                **{c: float(getattr(r, c)) for c in NUMERIC},
                "python_margin": float(m),
                "python_rings": float(p),
            }
            for r, m, p in zip(parity.itertuples(), margin, rings, strict=True)
        ],
    }
    (out_dir / "parity-fixture.json").write_text(json.dumps(fixture, separators=(",", ":")))
    logger.info("Wrote %d parity rows", len(fixture["rows"]))

    if args.no_parity:
        return 0
    node = shutil.which("node")
    if node is None:
        logger.warning("node not found; run `node scripts/check_web_parity.mjs` to verify the JS model")
        return 0
    return subprocess.call([node, str(ROOT / "scripts" / "check_web_parity.mjs"), str(out_dir)])


if __name__ == "__main__":
    sys.exit(main())

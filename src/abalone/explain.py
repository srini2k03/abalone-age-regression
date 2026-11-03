"""SHAP-based feature importance for fitted pipelines (optional: ``pip install shap``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

TREE_MODELS = ("xgboost", "lightgbm", "catboost", "random_forest")


def _pick_pipeline(model, member: str | None = None) -> tuple[str, Pipeline]:
    """Find a preprocess+model pipeline inside a (possibly ensembled) fitted model."""
    inner = model.regressor_ if isinstance(model, TransformedTargetRegressor) else model
    if isinstance(inner, Pipeline):
        return type(inner.named_steps["model"]).__name__, inner
    named = getattr(inner, "named_estimators_", None)
    if named is None:
        raise ValueError(f"Cannot explain model of type {type(inner).__name__}")
    order = [member] if member else list(TREE_MODELS)
    for name in order:
        if name in named:
            return name, named[name]
    name = next(iter(named))
    return name, named[name]


def shap_importance(
    model, X: pd.DataFrame, member: str | None = None, max_samples: int = 1000, seed: int = 0
) -> tuple[pd.DataFrame, str]:
    """Mean |SHAP| per engineered feature, in log1p(rings) space."""
    import shap

    name, pipe = _pick_pipeline(model, member)
    sample = X.sample(min(max_samples, len(X)), random_state=seed)
    Xt = pipe.named_steps["preprocess"].transform(sample)
    estimator = pipe.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(Xt)
    else:
        explainer = shap.LinearExplainer(estimator, Xt)
        values = explainer.shap_values(Xt)
    values = np.asarray(values)
    table = (
        pd.DataFrame({"feature": Xt.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return table, name


def plot_importance(table: pd.DataFrame, path: str | Path, title: str, top: int = 15) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_df = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(top_df) + 1.2))
    ax.barh(top_df["feature"], top_df["mean_abs_shap"], color="#2a6f97")
    ax.set_xlabel("mean |SHAP value|  (log1p rings)")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path

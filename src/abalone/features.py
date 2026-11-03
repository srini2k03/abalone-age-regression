"""Feature engineering as scikit-learn transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from abalone.config import FeatureConfig
from abalone.data import CATEGORICAL, NUMERIC

EPS = 1e-4
WEIGHT_COLUMNS = ["whole_weight", "shucked_weight", "viscera_weight", "shell_weight"]


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorize numeric columns to quantile bounds learned on the training fold.

    Bounds are learned in ``fit`` only, so there is no leakage across CV folds.
    Columns not listed are passed through untouched.
    """

    def __init__(self, columns: list[str] | None = None, quantiles: tuple[float, float] = (0.001, 0.999)):
        self.columns = columns
        self.quantiles = quantiles

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.columns if self.columns is not None else list(X.select_dtypes("number").columns)
        lo, hi = self.quantiles
        self.lower_ = X[cols].quantile(lo)
        self.upper_ = X[cols].quantile(hi)
        self.columns_ = list(cols)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "columns_")
        X = X.copy()
        X[self.columns_] = X[self.columns_].clip(lower=self.lower_, upper=self.upper_, axis=1)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


class MorphometricFeatures(BaseEstimator, TransformerMixin):
    """Domain features derived from the raw morphometric measurements.

    * ``*_ratio``: each sub-weight divided by whole weight (body composition).
    * ``volume``: ``length * diameter * height`` (ellipsoid-ish size proxy) and
      ``density`` = whole weight / volume.
    * ``log_*``: ``log1p`` of the heavy-tailed weight columns.
    """

    def __init__(self, add_ratios: bool = True, add_volume: bool = True, add_log_weights: bool = True):
        self.add_ratios = add_ratios
        self.add_volume = add_volume
        self.add_log_weights = add_log_weights

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        X = X.copy()
        whole = X["whole_weight"].clip(lower=EPS)
        if self.add_ratios:
            X["shucked_ratio"] = X["shucked_weight"] / whole
            X["viscera_ratio"] = X["viscera_weight"] / whole
            X["shell_ratio"] = X["shell_weight"] / whole
            X["meat_to_shell"] = X["shucked_weight"] / X["shell_weight"].clip(lower=EPS)
        if self.add_volume:
            volume = X["length"] * X["diameter"] * X["height"].clip(lower=EPS)
            X["volume"] = volume
            X["density"] = X["whole_weight"] / volume.clip(lower=EPS)
            X["aspect_ratio"] = X["diameter"] / X["length"].clip(lower=EPS)
        if self.add_log_weights:
            for col in WEIGHT_COLUMNS:
                X[f"log_{col}"] = np.log1p(X[col].clip(lower=0))
        return X

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_)
        if self.add_ratios:
            names += ["shucked_ratio", "viscera_ratio", "shell_ratio", "meat_to_shell"]
        if self.add_volume:
            names += ["volume", "density", "aspect_ratio"]
        if self.add_log_weights:
            names += [f"log_{c}" for c in WEIGHT_COLUMNS]
        return np.asarray(names, dtype=object)


def engineered_numeric_columns(cfg: FeatureConfig) -> list[str]:
    """Names of all numeric columns produced by :class:`MorphometricFeatures`."""
    names = (
        MorphometricFeatures(cfg.add_ratios, cfg.add_volume, cfg.add_log_weights)
        .fit(pd.DataFrame(columns=CATEGORICAL + NUMERIC))
        .get_feature_names_out()
    )
    return [n for n in names if n not in CATEGORICAL]


def build_preprocessor(cfg: FeatureConfig | None = None, scale: bool = False) -> Pipeline:
    """Clip outliers -> engineer features -> one-hot ``sex`` (+ optional scaling).

    Linear models get ``scale=True``; tree ensembles are scale-invariant.
    """
    cfg = cfg or FeatureConfig()
    numeric_cols = engineered_numeric_columns(cfg)
    steps: list[tuple[str, object]] = []
    if cfg.clip_outliers:
        steps.append(("clip", QuantileClipper(columns=list(NUMERIC), quantiles=tuple(cfg.clip_quantiles))))
    steps.append(("features", MorphometricFeatures(cfg.add_ratios, cfg.add_volume, cfg.add_log_weights)))
    encoder = ColumnTransformer(
        [
            ("sex", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", StandardScaler() if scale else "passthrough", numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    steps.append(("encode", encoder))
    pipe = Pipeline(steps)
    pipe.set_output(transform="pandas")
    return pipe

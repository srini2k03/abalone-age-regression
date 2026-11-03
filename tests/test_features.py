import numpy as np
import pandas as pd

from abalone.config import FeatureConfig
from abalone.data import FEATURES, NUMERIC
from abalone.features import MorphometricFeatures, QuantileClipper, build_preprocessor


def test_clipper_learns_bounds_on_fit_data_only(synthetic_df):
    train, test = synthetic_df.iloc[:300], synthetic_df.iloc[300:].copy()
    test.loc[test.index[0], "length"] = 99.0  # extreme outlier in unseen data
    clip = QuantileClipper(columns=NUMERIC, quantiles=(0.01, 0.99)).fit(train[FEATURES])
    out = clip.transform(test[FEATURES])
    assert out["length"].max() <= train["length"].quantile(0.99) + 1e-12
    assert list(out.columns) == FEATURES  # categorical passthrough preserved
    assert (out["sex"] == test["sex"]).all()


def test_morphometric_features_columns_and_names(synthetic_df):
    tr = MorphometricFeatures().fit(synthetic_df[FEATURES])
    out = tr.transform(synthetic_df[FEATURES])
    assert list(out.columns) == list(tr.get_feature_names_out())
    for col in ("shell_ratio", "volume", "density", "log_whole_weight"):
        assert col in out
    np.testing.assert_allclose(
        out["shell_ratio"], synthetic_df["shell_weight"] / synthetic_df["whole_weight"]
    )


def test_morphometric_features_handle_zero_height_and_weight(synthetic_df):
    df = synthetic_df[FEATURES].copy()
    df.loc[0, "height"] = 0.0
    df.loc[1, "whole_weight"] = 0.0
    df.loc[2, "shell_weight"] = 0.0
    out = MorphometricFeatures().fit_transform(df)
    assert np.isfinite(out.select_dtypes("number").to_numpy()).all()


def test_feature_flags_toggle_outputs(synthetic_df):
    out = MorphometricFeatures(add_ratios=False, add_volume=False, add_log_weights=False).fit_transform(
        synthetic_df[FEATURES]
    )
    assert list(out.columns) == FEATURES


def test_preprocessor_one_hot_and_unknown_sex(synthetic_df):
    pre = build_preprocessor(FeatureConfig(), scale=True).fit(synthetic_df[FEATURES])
    unseen = synthetic_df[FEATURES].head(3).copy()
    unseen["sex"] = ["M", "F", "X"]
    out = pre.transform(unseen)
    assert isinstance(out, pd.DataFrame)
    assert {"sex_M", "sex_F", "sex_I"} <= set(out.columns)
    assert out.loc[2, ["sex_M", "sex_F", "sex_I"]].sum() == 0  # unknown -> all zeros
    assert not out.isna().any().any()


def test_preprocessor_scaling(synthetic_df):
    out = build_preprocessor(FeatureConfig(), scale=True).fit_transform(synthetic_df[FEATURES])
    assert abs(out["length"].mean()) < 1e-6
    raw = build_preprocessor(FeatureConfig(clip_outliers=False), scale=False).fit_transform(
        synthetic_df[FEATURES]
    )
    np.testing.assert_allclose(raw["length"], synthetic_df["length"])

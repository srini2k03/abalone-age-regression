import numpy as np
import pytest

from abalone.data import split_xy
from abalone.evaluation import compare_models, cross_validate_model
from abalone.models import ALL_MODELS, build_model, is_available, resolve_models


@pytest.mark.parametrize("name", ALL_MODELS)
def test_every_model_fits_and_predicts_positive_rings(name, synthetic_df, small_cfg):
    if not is_available(name):
        pytest.skip(f"{name} not installed")
    X, y = split_xy(synthetic_df)
    model = build_model(name, small_cfg).fit(X, y)
    pred = model.predict(X.head(20))
    assert pred.shape == (20,)
    assert np.all(pred > 0)


def test_models_learn_in_log_space(synthetic_df, small_cfg):
    """TransformedTargetRegressor must fit the inner model on log1p(y)."""
    X, y = split_xy(synthetic_df)
    model = build_model("ridge", small_cfg).fit(X, y)
    inner_pred = model.regressor_.predict(X)
    np.testing.assert_allclose(np.expm1(inner_pred), model.predict(X))
    assert inner_pred.max() < np.log1p(y.max()) + 1


def test_resolve_models_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_models(["nope"])


def test_cross_validate_model_shapes(synthetic_df, small_cfg):
    X, y = split_xy(synthetic_df)
    res = cross_validate_model("ridge", X, y, small_cfg)
    assert res["n_folds"] == 3 and len(res["fold_rmsle"]) == 3
    assert 0 < res["rmsle_mean"] < 1


def test_repeated_cv_fold_count(synthetic_df, small_cfg):
    cfg = small_cfg.model_copy(update={"cv": small_cfg.cv.model_copy(update={"n_repeats": 2})})
    X, y = split_xy(synthetic_df)
    assert cross_validate_model("linear", X, y, cfg)["n_folds"] == 6


def test_compare_models_beats_baseline(synthetic_df, small_cfg):
    X, y = split_xy(synthetic_df)
    board = compare_models(["baseline", "linear", "ridge"], X, y, small_cfg)
    assert list(board["rmsle_mean"]) == sorted(board["rmsle_mean"])
    assert board.iloc[-1]["model"] == "baseline"
    assert (board.loc[board.model != "baseline", "improvement_vs_baseline_pct"] > 0).all()

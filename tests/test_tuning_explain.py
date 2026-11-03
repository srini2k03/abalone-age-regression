import pytest

from abalone.data import split_xy
from abalone.models import build_model


def test_optuna_tuning_returns_params(synthetic_df, small_cfg):
    pytest.importorskip("optuna")
    from abalone.tuning import tune

    X, y = split_xy(synthetic_df)
    res = tune("ridge", X, y, small_cfg, n_trials=3, n_splits=3)
    assert res["n_trials"] == 3 and "alpha" in res["best_params"]
    assert 0 < res["best_rmsle"] < 1


@pytest.mark.parametrize("name", ["xgboost", "ridge", "blend"])
def test_shap_importance(name, synthetic_df, small_cfg):
    pytest.importorskip("shap")
    from abalone.explain import shap_importance

    X, y = split_xy(synthetic_df)
    model = build_model(name, small_cfg).fit(X, y)
    table, _explained = shap_importance(model, X, max_samples=100)
    assert table["mean_abs_shap"].is_monotonic_decreasing
    assert "shell_weight" in set(table["feature"])
    assert (table["mean_abs_shap"] >= 0).all()

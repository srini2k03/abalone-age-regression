import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from abalone.metrics import regression_report, rmsle, rmsle_scorer


def test_rmsle_zero_for_perfect_predictions():
    y = np.array([1, 5, 10, 29])
    assert rmsle(y, y) == pytest.approx(0.0)


def test_rmsle_matches_manual_formula():
    y, p = np.array([3.0, 10.0]), np.array([4.0, 8.0])
    expected = np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2))
    assert rmsle(y, p) == pytest.approx(expected)


def test_rmsle_equals_rmse_in_log_space():
    rng = np.random.default_rng(0)
    y = rng.integers(1, 30, 100).astype(float)
    p = y * rng.uniform(0.7, 1.3, 100)
    rmse_log = np.sqrt(np.mean((np.log1p(y) - np.log1p(p)) ** 2))
    assert rmsle(y, p) == pytest.approx(rmse_log)


def test_negative_predictions_are_clipped():
    assert rmsle([0.0], [-5.0]) == pytest.approx(0.0)


def test_negative_targets_raise():
    with pytest.raises(ValueError):
        rmsle([-1.0], [1.0])


def test_scorer_is_negated_for_sklearn():
    class Const(RegressorMixin, BaseEstimator):
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.full(len(X), 2.0)

    X, y = np.zeros((2, 1)), np.array([1.0, 3.0])
    assert rmsle_scorer(Const().fit(X, y), X, y) == pytest.approx(-rmsle(y, [2.0, 2.0]))


def test_regression_report_keys():
    rep = regression_report([1, 2, 3], [1, 2, 4])
    assert set(rep) == {"rmsle", "rmse", "mae", "r2"}

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from abalone.cli import app
from abalone.persistence import load_model
from abalone.pipeline import evaluate, predict_file, train

runner = CliRunner()


def test_train_evaluate_predict_roundtrip(tmp_path, small_cfg, synthetic_csv):
    result = train(small_cfg, data_path=synthetic_csv, models=["baseline", "linear", "ridge"])
    assert result["best_model"] in {"linear", "ridge"}
    assert result["holdout"]["rmsle"] < result["holdout"]["baseline_rmsle"]

    _model, meta = load_model(result["model_path"])
    assert meta["model_name"] == result["best_model"]
    assert meta["target_transform"] == "log1p"
    run_dir = tmp_path / "runs"
    metrics = json.loads(next(run_dir.glob("*/metrics.json")).read_text())
    assert "cv_rmsle_mean" in metrics

    rep = evaluate(small_cfg, result["model_path"], synthetic_csv)
    assert rep["rmsle"] == pytest.approx(result["holdout"]["rmsle"])

    test_df = pd.read_csv(synthetic_csv).drop(columns=["rings"]).head(10)
    test_df.insert(0, "id", range(100, 110))
    test_csv = tmp_path / "test.csv"
    test_df.to_csv(test_csv, index=False)
    sub = predict_file(result["model_path"], test_csv, tmp_path / "sub.csv", kaggle_format=True)
    assert list(sub.columns) == ["id", "Rings"] and sub["id"].tolist() == list(range(100, 110))


def test_train_with_tuned_params_file(tmp_path, small_cfg, synthetic_csv):
    params = tmp_path / "best.json"
    params.write_text(json.dumps({"model": "ridge", "best_params": {"alpha": 3.0}, "best_rmsle": 0.1}))
    result = train(small_cfg, data_path=synthetic_csv, models=["ridge"], params_file=params)
    model, _ = load_model(result["model_path"])
    assert model.regressor_.named_steps["model"].alpha == 3.0


def test_cli_train_and_predict(tmp_path, synthetic_csv):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"cv: {{n_splits: 3, n_repeats: 1, n_jobs: 1}}\n"
        f"tracking: {{dir: {tmp_path / 'runs'}}}\n"
        f"artifacts: {{dir: {tmp_path / 'models'}}}\n"
    )
    out = runner.invoke(app, ["train", "-c", str(cfg_path), "-d", synthetic_csv, "-m", "baseline,linear"])
    assert out.exit_code == 0, out.output
    assert "Best model: linear" in out.output
    model_path = tmp_path / "models" / "model.joblib"
    assert model_path.exists()

    out = runner.invoke(
        app, ["predict", synthetic_csv, "--model", str(model_path), "-o", str(tmp_path / "p.csv")]
    )
    assert out.exit_code == 0, out.output
    preds = pd.read_csv(tmp_path / "p.csv")
    assert {"predicted_rings", "predicted_age_years"} <= set(preds.columns)
    assert (preds["predicted_age_years"] - preds["predicted_rings"]).round(6).eq(1.5).all()

    out = runner.invoke(
        app, ["evaluate", "--model", str(model_path), "-c", str(cfg_path), "-d", synthetic_csv, "--full"]
    )
    assert out.exit_code == 0, out.output
    assert json.loads(out.output)["n_rows"] == 400


def test_config_loading_and_validation(tmp_path):
    from pydantic import ValidationError

    from abalone.config import load_config

    cfg = load_config()
    assert cfg.cv.n_splits >= 2 and "catboost" in cfg.models.enabled
    assert load_config(overrides={"seed": 7}).seed == 7
    with pytest.raises(ValidationError):
        load_config(overrides={"features": {"clip_quantiles": [0.9, 0.1]}})
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")

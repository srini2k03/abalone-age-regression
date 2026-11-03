"""Command-line interface: ``abalone train | evaluate | tune | predict | explain``."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from abalone.config import load_config

app = typer.Typer(add_completion=False, help="Abalone age regression pipeline.", no_args_is_help=True)

ConfigOpt = typer.Option(None, "--config", "-c", help="YAML config (default: configs/default.yaml).")
DataOpt = typer.Option(None, "--data", "-d", help="Local data file (UCI .data or CSV); overrides config.")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def train(
    config: Path | None = ConfigOpt,
    data: Path | None = DataOpt,
    models: str | None = typer.Option(None, "--models", "-m", help="Comma-separated model names."),
    params: Path | None = typer.Option(None, "--params", help="JSON of tuned params (from `abalone tune`)."),
    select: str | None = typer.Option(None, help="Force this model as the final model."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Where to save the model bundle."),
    n_repeats: int | None = typer.Option(None, help="Override cv.n_repeats (e.g. 1 for a quick run)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Cross-validate models, refit the best and save it with metadata."""
    from abalone.pipeline import train as run_train

    _setup_logging(verbose)
    overrides = {"cv": {"n_repeats": n_repeats}} if n_repeats else None
    cfg = load_config(config, overrides)
    result = run_train(cfg, data, models.split(",") if models else None, params, select, output)
    board = result["leaderboard"]
    cols = [
        c
        for c in (
            "model",
            "rmsle_mean",
            "rmsle_std",
            "rmse_mean",
            "mae_mean",
            "r2_mean",
            "improvement_vs_baseline_pct",
            "fit_time_s",
        )
        if c in board.columns
    ]
    typer.echo("\nCross-validation leaderboard (lower RMSLE is better):")
    typer.echo(board[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    typer.echo(f"\nBest model: {result['best_model']}")
    if result["holdout"]:
        typer.echo("Hold-out: " + json.dumps({k: round(v, 4) for k, v in result["holdout"].items()}))
    typer.echo(f"Saved model -> {result['model_path']}\nRun log    -> {result['run_dir']}")


@app.command()
def evaluate(
    model: Path = typer.Option(Path("models/model.joblib"), "--model", help="Saved model bundle."),
    config: Path | None = ConfigOpt,
    data: Path | None = DataOpt,
    full: bool = typer.Option(False, help="Score the whole --data file instead of its hold-out split."),
) -> None:
    """Score a saved model on the hold-out split (or a full labelled file with --full)."""
    from abalone.pipeline import evaluate as run_evaluate

    _setup_logging(False)
    report = run_evaluate(load_config(config), model, data, full)
    typer.echo(json.dumps(report, indent=2))


@app.command()
def tune(
    model: str = typer.Option("xgboost", "--model", "-m", help="Model to tune."),
    n_trials: int = typer.Option(30, help="Number of Optuna trials."),
    timeout: float | None = typer.Option(None, help="Stop after this many seconds."),
    config: Path | None = ConfigOpt,
    data: Path | None = DataOpt,
    output: Path = typer.Option(Path("models/best_params.json"), "--output", "-o"),
) -> None:
    """Optuna search on the training split; writes best params as JSON."""
    from abalone.data import split_xy
    from abalone.pipeline import prepare_data
    from abalone.tuning import tune as run_tune

    _setup_logging(False)
    cfg = load_config(config)
    train_df, _, _ = prepare_data(cfg, data)
    X, y = split_xy(train_df)
    result = run_tune(model, X, y, cfg, n_trials=n_trials, timeout=timeout)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    typer.echo(json.dumps(result, indent=2))
    typer.echo(f"Saved -> {output}  (use: abalone train --params {output})")


@app.command()
def predict(
    input: Path = typer.Argument(..., help="CSV (Kaggle/UCI headers) or UCI .data file."),
    output: Path = typer.Option(Path("predictions.csv"), "--output", "-o"),
    model: Path = typer.Option(Path("models/model.joblib"), "--model"),
    kaggle: bool = typer.Option(False, help="Write a Kaggle S4E4 submission (id,Rings)."),
) -> None:
    """Predict rings / age for every row of a file."""
    from abalone.pipeline import predict_file

    preds = predict_file(model, input, output, kaggle_format=kaggle)
    typer.echo(f"Wrote {len(preds)} predictions -> {output}")


@app.command()
def explain(
    model: Path = typer.Option(Path("models/model.joblib"), "--model"),
    member: str | None = typer.Option(None, help="Ensemble member to explain (default: first tree model)."),
    config: Path | None = ConfigOpt,
    data: Path | None = DataOpt,
    output_dir: Path = typer.Option(Path("reports"), "--output-dir"),
) -> None:
    """SHAP feature importance (mean |SHAP|) for the saved model."""
    from abalone.data import split_xy
    from abalone.explain import plot_importance, shap_importance
    from abalone.persistence import load_model
    from abalone.pipeline import prepare_data

    _setup_logging(False)
    cfg = load_config(config)
    fitted, _ = load_model(model)
    train_df, _, _ = prepare_data(cfg, data)
    X, _ = split_xy(train_df)
    table, name = shap_importance(fitted, X, member=member, seed=cfg.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "shap_importance.csv", index=False)
    plot_importance(table, output_dir / "shap_importance.png", f"SHAP importance ({name})")
    typer.echo(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    typer.echo(f"Saved -> {output_dir}/shap_importance.csv|png")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

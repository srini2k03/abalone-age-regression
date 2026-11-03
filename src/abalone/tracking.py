"""Minimal experiment tracking: JSON run folders, or MLflow if installed."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        return obj.item()
    return obj


class JsonTracker:
    """Writes ``params.json``, ``metrics.json`` and artifacts to ``<dir>/<run_id>/``."""

    def __init__(self, root: str | Path, run_name: str = "run"):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{stamp}_{run_name}_{uuid.uuid4().hex[:6]}"
        self.dir = Path(root) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._params: dict[str, Any] = {}
        self._metrics: dict[str, Any] = {}

    def log_params(self, params: dict[str, Any]) -> None:
        self._params.update(_jsonable(params))
        (self.dir / "params.json").write_text(json.dumps(self._params, indent=2, default=str))

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        self._metrics.update(_jsonable(metrics))
        (self.dir / "metrics.json").write_text(json.dumps(self._metrics, indent=2, default=str))

    def log_table(self, name: str, df: pd.DataFrame) -> Path:
        path = self.dir / f"{name}.csv"
        df.to_csv(path, index=False)
        return path

    def end(self) -> None:  # symmetry with MLflow
        pass


class MlflowTracker(JsonTracker):
    """Mirrors everything to MLflow in addition to the JSON run folder."""

    def __init__(self, root: str | Path, run_name: str = "run"):
        super().__init__(root, run_name)
        import mlflow

        self._mlflow = mlflow
        mlflow.set_experiment("abalone-age-regression")
        mlflow.start_run(run_name=self.run_id)

    def log_params(self, params: dict[str, Any]) -> None:
        super().log_params(params)
        flat = {k: str(v)[:250] for k, v in self._params.items()}
        self._mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        super().log_metrics(metrics)
        numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        self._mlflow.log_metrics(numeric)

    def log_table(self, name: str, df: pd.DataFrame) -> Path:
        path = super().log_table(name, df)
        self._mlflow.log_artifact(str(path))
        return path

    def end(self) -> None:
        self._mlflow.end_run()


def make_tracker(backend: str, root: str | Path, run_name: str = "run") -> JsonTracker:
    if backend == "mlflow":
        return MlflowTracker(root, run_name)
    return JsonTracker(root, run_name)

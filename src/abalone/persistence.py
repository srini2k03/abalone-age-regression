"""Model persistence with metadata (joblib)."""

from __future__ import annotations

import contextlib
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import sklearn

from abalone import __version__
from abalone.data import FEATURES


def _lib_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "scikit-learn": sklearn.__version__}
    for mod in ("xgboost", "lightgbm", "catboost", "pandas", "numpy"):
        with contextlib.suppress(ImportError):
            versions[mod] = __import__(mod).__version__
    return versions


def save_model(model: Any, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "package_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURES,
        "target": "rings",
        "target_transform": "log1p",
        "libraries": _lib_versions(),
        **(metadata or {}),
    }
    joblib.dump({"model": model, "metadata": meta}, path)
    return path


def load_model(path: str | Path) -> tuple[Any, dict[str, Any]]:
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"{path} is not an abalone model bundle")
    return bundle["model"], bundle.get("metadata", {})

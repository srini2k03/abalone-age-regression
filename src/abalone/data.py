"""Data loading: UCI download with caching/integrity checks, or Kaggle CSVs."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from abalone.config import DataConfig

logger = logging.getLogger(__name__)

TARGET = "rings"
CATEGORICAL = ["sex"]
NUMERIC = [
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]
FEATURES = CATEGORICAL + NUMERIC
UCI_COLUMNS = [*FEATURES, TARGET]

# Kaggle Playground S4E4 uses the original UCI headers, with the shucked and
# viscera weights exported as "Whole weight.1" / "Whole weight.2".
_COLUMN_ALIASES = {
    "sex": "sex",
    "length": "length",
    "diameter": "diameter",
    "height": "height",
    "whole weight": "whole_weight",
    "whole_weight": "whole_weight",
    "whole weight.1": "shucked_weight",
    "shucked weight": "shucked_weight",
    "shucked_weight": "shucked_weight",
    "whole weight.2": "viscera_weight",
    "viscera weight": "viscera_weight",
    "viscera_weight": "viscera_weight",
    "shell weight": "shell_weight",
    "shell_weight": "shell_weight",
    "rings": "rings",
    "id": "id",
}

CACHE_FILENAME = "abalone.data"


class DataDownloadError(RuntimeError):
    """Raised when the dataset cannot be fetched from any configured URL."""


class DataIntegrityError(RuntimeError):
    """Raised when downloaded / cached data does not match the expected fingerprint."""


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map UCI / Kaggle header variants onto canonical snake_case names."""
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in _COLUMN_ALIASES:
            rename[col] = _COLUMN_ALIASES[key]
    out = df.rename(columns=rename)
    if "sex" in out.columns:
        out["sex"] = out["sex"].astype(str).str.strip().str.upper()
    return out


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_fingerprint(df: pd.DataFrame) -> str:
    """Content hash of a canonical table, robust to line endings and float repr."""
    canon = df[UCI_COLUMNS].copy()
    for col in NUMERIC:
        canon[col] = canon[col].map(lambda v: f"{v:.4f}")
    payload = canon.to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()


def parse_uci(raw: bytes | str | Path) -> pd.DataFrame:
    """Parse the headerless, comma-separated UCI ``abalone.data`` format."""
    if isinstance(raw, Path):
        source: io.StringIO | Path = raw
    else:
        text = raw.decode() if isinstance(raw, bytes) else raw
        source = io.StringIO(text)
    df = pd.read_csv(source, header=None, names=UCI_COLUMNS)
    df["sex"] = df["sex"].astype(str).str.strip().str.upper()
    return df


def _fetch(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "abalone-age-regression"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_uci(cfg: DataConfig, force: bool = False) -> Path:
    """Download the UCI dataset into ``cfg.cache_dir`` (cached; tries URLs in order)."""
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / CACHE_FILENAME
    meta_path = cache_dir / f"{CACHE_FILENAME}.meta.json"

    if target.exists() and not force:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("sha256") != file_sha256(target):
                raise DataIntegrityError(
                    f"Cached file {target} was modified (sha256 mismatch); re-run with force=True"
                )
        logger.info("Using cached dataset at %s", target)
        return target

    errors: list[str] = []
    for url in cfg.urls:
        try:
            logger.info("Downloading %s", url)
            payload = _fetch(url)
            df = parse_uci(payload)
            _verify(df, cfg.expected_fingerprint, origin=url)
        except DataIntegrityError:
            raise
        except Exception as exc:  # network / parse errors -> try next mirror
            errors.append(f"{url}: {exc}")
            logger.warning("Download failed for %s: %s", url, exc)
            continue
        target.write_bytes(payload)
        meta = {
            "source_url": url,
            "sha256": file_sha256(target),
            "fingerprint": frame_fingerprint(df),
            "n_rows": len(df),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        return target
    raise DataDownloadError("Could not download the abalone dataset:\n  " + "\n  ".join(errors))


def _verify(df: pd.DataFrame, expected: str | None, origin: str) -> None:
    if expected is None:
        return
    got = frame_fingerprint(df)
    if got != expected:
        raise DataIntegrityError(f"Fingerprint mismatch for {origin}: expected {expected}, got {got}")


def load_uci(cfg: DataConfig, force_download: bool = False) -> pd.DataFrame:
    path = download_uci(cfg, force=force_download)
    df = parse_uci(path)
    _verify(df, cfg.expected_fingerprint, origin=str(path))
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV with headers (Kaggle train/test or user-provided) and normalize columns."""
    return normalize_columns(pd.read_csv(path))


def load_dataset(cfg: DataConfig, path: str | Path | None = None) -> pd.DataFrame:
    """Load the training table according to config (or an explicit CSV path)."""
    if path is not None:
        p = Path(path)
        if p.suffix == ".data":
            return parse_uci(p)
        return load_csv(p)
    if cfg.source == "kaggle":
        return load_csv(cfg.kaggle_train_path)
    return load_uci(cfg)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURES].copy()
    y = df[TARGET].astype(float)
    return X, y


def holdout_split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/hold-out split stratified on binned ring counts."""
    if test_size <= 0:
        return df.reset_index(drop=True), df.iloc[0:0].copy()
    bins = np.clip(df[TARGET].to_numpy(), 3, 20)  # merge sparse tails into one stratum
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=bins)
    return train.reset_index(drop=True), test.reset_index(drop=True)

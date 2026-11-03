import json

import pandas as pd
import pytest

from abalone import data as data_mod
from abalone.config import DataConfig
from abalone.data import (
    DataDownloadError,
    DataIntegrityError,
    frame_fingerprint,
    holdout_split,
    load_uci,
    normalize_columns,
    parse_uci,
)
from abalone.validation import SchemaError, validate

UCI_SAMPLE = (
    "M,0.455,0.365,0.095,0.514,0.2245,0.101,0.15,15\n"
    "M,0.35,0.265,0.09,0.2255,0.0995,0.0485,0.07,7\n"
    "F,0.53,0.42,0.135,0.677,0.2565,0.1415,0.21,9\n"
    "I,0.33,0.255,0.08,0.205,0.0895,0.0395,0.055,7\n"
)


def test_parse_uci_columns_and_types():
    df = parse_uci(UCI_SAMPLE)
    assert list(df.columns) == data_mod.UCI_COLUMNS
    assert df["rings"].tolist() == [15, 7, 9, 7]
    validate(df)


def test_fingerprint_ignores_line_endings_and_trailing_newline():
    a = parse_uci(UCI_SAMPLE)
    b = parse_uci(UCI_SAMPLE.replace("\n", "\r\n").rstrip())
    assert frame_fingerprint(a) == frame_fingerprint(b)


def test_normalize_kaggle_columns():
    kaggle = pd.DataFrame(
        {
            "id": [0],
            "Sex": ["f"],
            "Length": [0.5],
            "Diameter": [0.4],
            "Height": [0.1],
            "Whole weight": [0.6],
            "Whole weight.1": [0.25],
            "Whole weight.2": [0.12],
            "Shell weight": [0.2],
            "Rings": [10],
        }
    )
    out = normalize_columns(kaggle)
    assert {"shucked_weight", "viscera_weight", "rings", "id"} <= set(out.columns)
    assert out.loc[0, "sex"] == "F"
    validate(out)


def test_download_falls_back_to_mirror_and_caches(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(url, timeout=30.0):
        calls.append(url)
        if "bad" in url:
            raise OSError("blocked")
        return UCI_SAMPLE.encode()

    monkeypatch.setattr(data_mod, "_fetch", fake_fetch)
    cfg = DataConfig(
        cache_dir=tmp_path,
        urls=["https://bad.example/a", "https://good.example/b"],
        expected_fingerprint=frame_fingerprint(parse_uci(UCI_SAMPLE)),
    )
    df = load_uci(cfg)
    assert len(df) == 4 and calls == ["https://bad.example/a", "https://good.example/b"]
    meta = json.loads((tmp_path / "abalone.data.meta.json").read_text())
    assert meta["source_url"] == "https://good.example/b"

    load_uci(cfg)  # second call served from cache
    assert len(calls) == 2


def test_download_rejects_fingerprint_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(data_mod, "_fetch", lambda url, timeout=30.0: UCI_SAMPLE.encode())
    cfg = DataConfig(cache_dir=tmp_path, urls=["https://x"], expected_fingerprint="0" * 64)
    with pytest.raises(DataIntegrityError):
        load_uci(cfg)


def test_download_all_urls_fail(tmp_path, monkeypatch):
    def boom(url, timeout=30.0):
        raise OSError("offline")

    monkeypatch.setattr(data_mod, "_fetch", boom)
    with pytest.raises(DataDownloadError):
        load_uci(DataConfig(cache_dir=tmp_path, urls=["https://x"]))


def test_tampered_cache_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(data_mod, "_fetch", lambda url, timeout=30.0: UCI_SAMPLE.encode())
    cfg = DataConfig(cache_dir=tmp_path, urls=["https://x"])
    load_uci(cfg)
    (tmp_path / "abalone.data").write_text(UCI_SAMPLE + "M,1,1,1,1,1,1,1,1\n")
    with pytest.raises(DataIntegrityError):
        load_uci(cfg)


def test_holdout_split_is_stratified_and_disjoint(synthetic_df):
    train, test = holdout_split(synthetic_df, 0.25, seed=1)
    assert len(train) + len(test) == len(synthetic_df)
    assert len(test) == pytest.approx(0.25 * len(synthetic_df), abs=1)
    assert abs(train["rings"].mean() - test["rings"].mean()) < 1.0


def test_validation_errors(synthetic_df):
    bad = synthetic_df.copy()
    bad.loc[0, "sex"] = "Z"
    bad.loc[1, "length"] = -1
    with pytest.raises(SchemaError) as exc:
        validate(bad)
    assert len(exc.value.issues) == 2
    with pytest.raises(SchemaError):
        validate(synthetic_df.drop(columns=["shell_weight"]))
    report = validate(synthetic_df.drop(columns=["rings"]), require_target=False)
    assert report.ok


def test_validation_counts_soft_anomalies(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "height"] = 0
    df.loc[1, "shucked_weight"] = df.loc[1, "whole_weight"] + 1
    report = validate(df)
    assert report.ok
    assert report.warnings["zero_height"] >= 1 and report.warnings["shucked_gt_whole"] == 1

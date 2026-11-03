"""Lightweight schema validation for abalone tables (no extra dependencies)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from abalone.data import FEATURES, NUMERIC, TARGET

VALID_SEX = {"M", "F", "I"}


class SchemaError(ValueError):
    """Raised when a table violates the hard schema contract."""

    def __init__(self, issues: list[str]):
        super().__init__("Schema validation failed:\n  - " + "\n  - ".join(issues))
        self.issues = issues


@dataclass
class ValidationReport:
    n_rows: int
    issues: list[str] = field(default_factory=list)
    warnings: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.issues


def validate(df: pd.DataFrame, require_target: bool = True, raise_on_error: bool = True) -> ValidationReport:
    """Check required columns, dtypes, domains and physical plausibility.

    Hard errors (missing columns, NaNs, unknown sex codes, negative values,
    non-positive targets) are collected in ``issues``. Soft anomalies that are
    known to exist in the real data (zero height, sub-weights exceeding whole
    weight) are only counted in ``warnings``; the outlier clipper and ratio
    features handle them downstream.
    """
    report = ValidationReport(n_rows=len(df))
    required = FEATURES + ([TARGET] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.issues.append(f"missing columns: {missing}")
        if raise_on_error:
            raise SchemaError(report.issues)
        return report

    if len(df) == 0:
        report.issues.append("table is empty")

    for col in NUMERIC + ([TARGET] if require_target else []):
        if not pd.api.types.is_numeric_dtype(df[col]):
            report.issues.append(f"column {col!r} is not numeric (dtype={df[col].dtype})")
            continue
        n_nan = int(df[col].isna().sum())
        if n_nan:
            report.issues.append(f"column {col!r} has {n_nan} missing values")
        n_neg = int((df[col] < 0).sum())
        if n_neg:
            report.issues.append(f"column {col!r} has {n_neg} negative values")

    bad_sex = sorted(set(df["sex"].dropna().astype(str)) - VALID_SEX)
    if bad_sex:
        report.issues.append(f"column 'sex' has unknown codes {bad_sex}; expected {sorted(VALID_SEX)}")
    if df["sex"].isna().any():
        report.issues.append("column 'sex' has missing values")

    if require_target and pd.api.types.is_numeric_dtype(df[TARGET]) and (df[TARGET] <= 0).any():
        report.issues.append("target 'rings' must be strictly positive")

    if not any("not numeric" in i for i in report.issues):
        report.warnings = {
            "zero_height": int((df["height"] == 0).sum()),
            "shucked_gt_whole": int((df["shucked_weight"] > df["whole_weight"]).sum()),
            "viscera_gt_whole": int((df["viscera_weight"] > df["whole_weight"]).sum()),
            "shell_gt_whole": int((df["shell_weight"] > df["whole_weight"]).sum()),
            "duplicate_rows": int(df.duplicated().sum()),
        }

    if raise_on_error and report.issues:
        raise SchemaError(report.issues)
    return report

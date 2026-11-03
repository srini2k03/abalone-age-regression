"""Request / response schemas for the inference API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AbaloneFeatures(BaseModel):
    """Morphometric measurements of one abalone (UCI units: mm / 200, grams / 200)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sex": "M",
                "length": 0.455,
                "diameter": 0.365,
                "height": 0.095,
                "whole_weight": 0.514,
                "shucked_weight": 0.2245,
                "viscera_weight": 0.101,
                "shell_weight": 0.15,
            }
        }
    )

    sex: Literal["M", "F", "I"] = Field(..., description="M = male, F = female, I = infant")
    length: float = Field(..., gt=0, le=2.0, description="Longest shell measurement")
    diameter: float = Field(..., gt=0, le=2.0, description="Perpendicular to length")
    height: float = Field(..., ge=0, le=2.0, description="With meat in shell")
    whole_weight: float = Field(..., gt=0, le=5.0, description="Whole abalone")
    shucked_weight: float = Field(..., ge=0, le=5.0, description="Weight of meat")
    viscera_weight: float = Field(..., ge=0, le=5.0, description="Gut weight after bleeding")
    shell_weight: float = Field(..., ge=0, le=5.0, description="After being dried")


class BatchRequest(BaseModel):
    instances: list[AbaloneFeatures] = Field(..., min_length=1, max_length=10_000)


class Prediction(BaseModel):
    predicted_rings: float
    predicted_age_years: float


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    model_name: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "model_not_loaded"]
    model_loaded: bool


class ModelInfo(BaseModel):
    model_name: str | None
    created_at: str | None
    cv: dict | None = None
    holdout: dict | None = None
    package_version: str | None = None

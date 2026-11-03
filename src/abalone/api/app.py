"""FastAPI app serving a persisted model bundle.

Run with ``uvicorn abalone.api.app:app`` and set ``ABALONE_MODEL_PATH`` to the
bundle written by ``abalone train`` (default ``models/model.joblib``).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from abalone import __version__
from abalone.api.schemas import (
    AbaloneFeatures,
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ModelInfo,
    Prediction,
)
from abalone.data import FEATURES
from abalone.persistence import load_model
from abalone.pipeline import predict_frame

DEFAULT_MODEL_PATH = "models/model.joblib"


def create_app(
    model: Any | None = None, metadata: dict | None = None, model_path: str | Path | None = None
) -> FastAPI:
    """Build the app. Pass ``model`` directly (tests) or let it load from disk at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.model is None:
            path = Path(model_path or os.getenv("ABALONE_MODEL_PATH", DEFAULT_MODEL_PATH))
            if path.exists():
                app.state.model, app.state.metadata = load_model(path)
        yield

    app = FastAPI(
        title="Abalone Age Regression API",
        version=__version__,
        description="Predict abalone ring count / age from morphometric measurements.",
        lifespan=lifespan,
    )
    app.state.model = model
    app.state.metadata = metadata or {}

    def _model(request: Request):
        if request.app.state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded; set ABALONE_MODEL_PATH")
        return request.app.state.model

    def _predict(request: Request, rows: list[AbaloneFeatures]) -> list[Prediction]:
        frame = pd.DataFrame([r.model_dump() for r in rows], columns=FEATURES)
        preds = predict_frame(_model(request), frame)
        return [Prediction(**rec) for rec in preds.to_dict(orient="records")]

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        loaded = request.app.state.model is not None
        return HealthResponse(status="ok" if loaded else "model_not_loaded", model_loaded=loaded)

    @app.get("/model/info", response_model=ModelInfo)
    def model_info(request: Request) -> ModelInfo:
        _model(request)
        meta = request.app.state.metadata
        return ModelInfo(
            model_name=meta.get("model_name"),
            created_at=meta.get("created_at"),
            cv=meta.get("cv"),
            holdout=meta.get("holdout"),
            package_version=meta.get("package_version"),
        )

    @app.post("/predict", response_model=Prediction)
    def predict(request: Request, features: AbaloneFeatures) -> Prediction:
        return _predict(request, [features])[0]

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(request: Request, batch: BatchRequest) -> BatchResponse:
        return BatchResponse(
            predictions=_predict(request, batch.instances),
            model_name=request.app.state.metadata.get("model_name"),
        )

    return app


app = create_app()

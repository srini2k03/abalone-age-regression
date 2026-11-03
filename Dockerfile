# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ABALONE_MODEL_PATH=/app/models/model.joblib

# libgomp is required by XGBoost / LightGBM wheels.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN pip install ".[api,lightgbm]"

RUN useradd --create-home --uid 1000 app && mkdir -p /app/models && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# Mount a trained bundle at /app/models (see `make docker-run`).
CMD ["uvicorn", "abalone.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

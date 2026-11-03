PYTHON ?= python
VENV   ?= .venv
BIN     = $(VENV)/bin
IMAGE  ?= abalone-api

.PHONY: help venv install lint format test cov train train-quick tune evaluate explain predict serve docker-build docker-run clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

venv:  ## Create a virtual environment
	$(PYTHON) -m venv $(VENV)

install: venv  ## Install the package with all extras (editable)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[all]"

lint:  ## Ruff lint + format check
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:  ## Auto-format
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

test:  ## Run the test suite
	$(BIN)/pytest

cov:  ## Tests with coverage
	$(BIN)/pytest --cov=abalone --cov-report=term-missing

train:  ## Full CV comparison (5x2 repeated K-fold) + refit best model
	$(BIN)/abalone train

train-quick:  ## Single-repeat CV on the fast models
	$(BIN)/abalone train --n-repeats 1 --models baseline,linear,ridge,xgboost,catboost

tune:  ## Optuna search for XGBoost (30 trials)
	$(BIN)/abalone tune --model xgboost --n-trials 30

evaluate:  ## Score saved model on the hold-out split
	$(BIN)/abalone evaluate

explain:  ## SHAP feature importance -> reports/
	$(BIN)/abalone explain

serve:  ## Run the FastAPI service locally
	ABALONE_MODEL_PATH=models/model.joblib $(BIN)/uvicorn abalone.api.app:app --reload --port 8000

docker-build:  ## Build the API image
	docker build -t $(IMAGE) .

docker-run:  ## Run the API image with ./models mounted
	docker run --rm -p 8000:8000 -v $(PWD)/models:/app/models:ro $(IMAGE)

clean:  ## Remove caches and generated artifacts (keeps data/)
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov runs models reports catboost_info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

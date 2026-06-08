# ZepIris — build, test, run
# Requires: Poetry 2.x, Docker Compose v2 (`docker compose`).
# Optional: `brew install make` — GNU make is preinstalled on macOS/Linux.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

POETRY ?= poetry
COMPOSE ?= docker compose
# Module invocation avoids broken .venv/bin/* shebangs after cloning or renaming the repo.
PRE_COMMIT := $(POETRY) run python -m pre_commit

.PHONY: help install install-dev install-ml lock build build-ml build-all test lint format check \
	pre-commit up down run run-api-local run-ml clean

help: ## List targets
	@echo "ZepIris targets:"
	@grep -hE '^[a-zA-Z0-9_-]+:.*?##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-22s %s\n", $$1, $$2}'

install: ## Install Python deps (runtime only; no dev tools)
	$(POETRY) install --no-interaction

install-dev: ## Install runtime + dev tools (pre-commit, ruff, pytest)
	$(POETRY) install --with dev --no-interaction

install-ml: ## Install with optional ML extras (CPU torch via lockfile)
	$(POETRY) install --no-interaction --extras ml

lock: ## Refresh poetry.lock from pyproject.toml
	$(POETRY) lock --no-interaction

check: ## Validate pyproject.toml and lockfile
	$(POETRY) check

build: ## Build API Docker image (docker-compose service: api)
	$(COMPOSE) build api

build-ml: ## Build ML inference Docker image
	$(COMPOSE) build ml-inference

build-all: ## Build all images in docker-compose.yml
	$(COMPOSE) build

# Tests import Settings; set a dummy ML URL so config validation passes.
test: ## Run pytest (pass extra args: make test ARGS="-v -k health")
	ZEPIRIS_ML_INFERENCE_SERVICE_URL=http://127.0.0.1:8001 $(POETRY) run pytest $(ARGS)

lint: ## Ruff linter
	$(POETRY) run ruff check .

format: ## Ruff formatter
	$(POETRY) run ruff format .

pre-commit: install-dev ## Run all pre-commit hooks on the whole repo (via Poetry)
	$(PRE_COMMIT) run --all-files

up: ## Start full stack in background (builds if needed)
	$(COMPOSE) up -d --build

down: ## Stop and remove compose containers (keeps volumes)
	$(COMPOSE) down

run: ## Run full stack in foreground with logs (Ctrl+C stops all)
	$(COMPOSE) up --build

# Assumes `make up` — API on host :8000 talks to compose Milvus/MinIO/ML on published ports.
run-api-local: ## Run main API via Poetry against localhost compose ports
	ZEPIRIS_ML_INFERENCE_SERVICE_URL=http://127.0.0.1:8001 \
	ZEPIRIS_MINIO_ENDPOINT=127.0.0.1:9002 \
	ZEPIRIS_MILVUS_HOST=127.0.0.1 \
	ZEPIRIS_MILVUS_PORT=19530 \
	$(POETRY) run python -m uvicorn zepiris.main:app --host 0.0.0.0 --port 8000 --reload

run-ml: ## Run ML inference API via Poetry (port from ML_SERVICE_PORT or default in app)
	$(POETRY) run zepiris-ml-inference-api

clean: ## Remove local pytest / ruff caches
	rm -rf .pytest_cache .ruff_cache

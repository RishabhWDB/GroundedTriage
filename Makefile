# GroundedTriage — native macOS (arm64) dev workflow. No Docker required.
SHELL := /bin/bash

PYTHON ?= python3.11
ML_DIR := ml-service
VENV := $(ML_DIR)/.venv
VPY := $(abspath $(VENV))/bin/python
VBIN := $(abspath $(VENV))/bin

# Load .env if present so make targets see DATABASE_URL etc.
ifneq (,$(wildcard .env))
include .env
export
endif

.PHONY: setup dev api worker ml test test-go test-py test-integration lint fmt migrate seed

setup: ## Create the Python venv, install deps, fetch Go modules
	@command -v $(PYTHON) >/dev/null || { echo "$(PYTHON) not found (brew install python@3.11)"; exit 1; }
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(VPY) -m pip install --upgrade pip
	cd $(ML_DIR) && $(VPY) -m pip install -e ".[dev]"
	go mod download
	mkdir -p models/fast_tier models/deep_tier models/urgency datasets/raw datasets/processed

dev: ## Start api, worker and ml-service together (honcho + Procfile)
	$(VBIN)/honcho start

api:
	go run ./cmd/api

worker:
	go run ./cmd/worker

ml:
	cd $(ML_DIR) && $(VBIN)/uvicorn app.main:app --host 127.0.0.1 --port $${ML_PORT:-8001} --reload

test: test-go test-py ## Unit tests (slow + integration skipped)

test-go:
	go test ./...

test-py:
	cd $(ML_DIR) && $(VBIN)/pytest -q

test-integration: ## Integration tests against TEST_DATABASE_URL (never DATABASE_URL)
	@test -n "$$TEST_DATABASE_URL" || { echo "TEST_DATABASE_URL is unset"; exit 1; }
	go test -tags integration ./...
	cd $(ML_DIR) && $(VBIN)/pytest -q -m integration

lint: ## gofmt check, go vet, ruff check + format check
	@out=$$(gofmt -l cmd internal); if [ -n "$$out" ]; then echo "gofmt needed:"; echo "$$out"; exit 1; fi
	go vet ./...
	cd $(ML_DIR) && $(VBIN)/ruff check . && $(VBIN)/ruff format --check .

fmt:
	gofmt -w cmd internal
	cd $(ML_DIR) && $(VBIN)/ruff format . && $(VBIN)/ruff check --fix .

migrate: ## Apply migrations to DATABASE_URL (implemented in P0.2)
	@echo "migrate: not implemented yet (P0.2)"; exit 1

seed: ## Seed demo data (implemented in P8.1)
	@echo "seed: not implemented yet (P8.1)"; exit 1

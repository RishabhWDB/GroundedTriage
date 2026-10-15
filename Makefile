# GroundedTriage — native macOS (arm64) dev workflow. No Docker required.
SHELL := /bin/bash

PYTHON ?= python3.11
ML_DIR := ml-service
VENV := $(ML_DIR)/.venv
VPY := $(abspath $(VENV))/bin/python
VBIN := $(abspath $(VENV))/bin

ML_PORT ?= 8001

# .env is NOT included here: make would parse it as Makefile syntax and break on
# `$`, `#` and URL-encoded passwords. Programs load .env themselves (Go: godotenv,
# Python: python-dotenv / pydantic-settings; honcho reads it for `make dev`).

.PHONY: setup dev api worker ml test test-go test-py test-integration lint fmt migrate migrate-status seed data synthetic-meetings export-splits

setup: ## Create the Python venv, install deps, fetch Go modules
	@command -v $(PYTHON) >/dev/null || { echo "$(PYTHON) not found (brew install python@3.11)"; exit 1; }
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(VPY) -m pip install --upgrade pip
	cd $(ML_DIR) && $(VPY) -m pip install -e ".[dev,data]"
	go mod download
	mkdir -p models/fast_tier models/deep_tier models/urgency datasets/raw datasets/processed

dev: ## Start api, worker and ml-service together (honcho + Procfile)
	$(VBIN)/honcho start

api:
	go run ./cmd/api

worker:
	go run ./cmd/worker

ml:
	cd $(ML_DIR) && $(VBIN)/uvicorn app.main:app --host 127.0.0.1 --port $(ML_PORT) --reload

test: test-go test-py ## Unit tests (slow + integration skipped)

test-go:
	go test ./...

test-py:
	cd $(ML_DIR) && $(VBIN)/pytest -q

test-integration: ## Integration tests against TEST_DATABASE_URL (never DATABASE_URL); tests load .env and skip if unset
	go test -tags integration -count=1 ./...
	cd $(ML_DIR) && { $(VBIN)/pytest -q -m integration; rc=$$?; [ $$rc -eq 5 ] && echo "(no Python integration tests yet)" && rc=0; exit $$rc; }

lint: ## gofmt check, go vet, ruff check + format check
	@out=$$(gofmt -l cmd internal migrations); if [ -n "$$out" ]; then echo "gofmt needed:"; echo "$$out"; exit 1; fi
	go vet ./...
	cd $(ML_DIR) && $(VBIN)/ruff check . && $(VBIN)/ruff format --check .

fmt:
	gofmt -w cmd internal migrations
	cd $(ML_DIR) && $(VBIN)/ruff format . && $(VBIN)/ruff check --fix .

migrate: ## Apply pending migrations to DATABASE_URL (idempotent; up only)
	go run ./cmd/migrate up

migrate-status: ## Show applied/pending migrations on DATABASE_URL
	go run ./cmd/migrate status

data: ## Build datasets/processed splits + manifest + dataset card (downloads raw data on first run)
	cd $(ML_DIR) && $(VPY) -m data.build

synthetic-meetings: ## (Re)generate synthetic meeting emails via Groq; cached, resumable
	cd $(ML_DIR) && $(VPY) -m data.synthetic_meetings

export-splits: ## Bundle splits + taxonomy + card for the GPU PC (datasets/export/)
	cd $(ML_DIR) && $(VPY) -m data.export

seed: ## Seed demo data (implemented in P8.1)
	@echo "seed: not implemented yet (P8.1)"; exit 1

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

.PHONY: setup setup-models setup-baseline baseline baseline-eval baseline-stress dev api worker ml test test-go test-py test-integration lint fmt migrate migrate-status migrate-verify seed data data-audit synthetic-meetings stress-set export-splits fast-tier urgency-eval tune-tau ood-set ood-eval style-aug spam-filter

setup: ## Create the Python venv, install deps, fetch Go modules
	@command -v $(PYTHON) >/dev/null || { echo "$(PYTHON) not found (brew install python@3.11)"; exit 1; }
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(VPY) -m pip install --upgrade pip
	cd $(ML_DIR) && $(VPY) -m pip install -e ".[dev,data]"
	go mod download
	mkdir -p models/fast_tier models/deep_tier models/urgency datasets/raw datasets/processed

setup-models: ## Install torch + transformers for the deep tier / urgency model (P3.2, P3.3)
	cd $(ML_DIR) && $(VPY) -m pip install -e ".[dev,data,models,eval]"

setup-baseline: ## Install the optional SCRIBE baseline extra (torch + transformers; models download on first run)
	cd $(ML_DIR) && $(VPY) -m pip install -e ".[dev,data,baseline,eval]"

baseline: ## SCRIBE baseline CLI, e.g. make baseline ARGS="classify --split test --limit 10"
	cd $(ML_DIR) && $(VPY) -m baseline $(ARGS)

baseline-eval: ## P2.3: run the SCRIBE baseline on the test split -> results/baseline/ (~90 min on M1 under swap; resumable)
	cd $(ML_DIR) && $(VPY) -m baseline evaluate --split test

baseline-stress: ## Score the reviewed Meeting stress set into results/baseline/metrics.json (needs a finished baseline-eval)
	cd $(ML_DIR) && $(VPY) -m baseline evaluate --stress-only

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

migrate-verify: ## Read-only check: vector extension, all tables, HNSW index
	go run ./cmd/migrate verify

data: ## Build datasets/processed splits + manifest + dataset card (downloads raw data on first run)
	cd $(ML_DIR) && $(VPY) -m data.build

synthetic-meetings: ## (Re)generate synthetic meeting emails via Groq; cached, resumable
	cd $(ML_DIR) && $(VPY) -m data.synthetic_meetings

data-audit: ## Synthetic diversity, train/test leakage and subject-shortcut checks -> results/data_audit/
	cd $(ML_DIR) && $(VPY) -m data.audit

stress-set: ## Generate the hard Meeting Scheduling stress-test CSV (eval only; needs human review)
	cd $(ML_DIR) && $(VPY) -m data.stress_meetings

export-splits: ## Bundle splits + taxonomy + card for the GPU PC (datasets/export/)
	cd $(ML_DIR) && $(VPY) -m data.export

fast-tier: ## P3.1: train the TF-IDF+LR fast tier on train, report on val -> models/fast_tier/, results/fast_tier/
	cd $(ML_DIR) && $(VPY) -m triage.train_fast_tier

urgency-eval: ## P3.3: score the urgency heuristic (and model, if present) on val -> results/urgency/
	cd $(ML_DIR) && $(VPY) -m triage.eval_urgency

tune-tau: ## P3.4: sweep the cascade threshold on val (needs models/deep_tier/) -> config/cascade.yaml, results/cascade/
	cd $(ML_DIR) && $(VPY) -m triage.tune_threshold

ood-set: ## P3.11: build the OOD set (Twitter pre-labelled for review, Enron spam, team template) -> datasets/ood/
	cd $(ML_DIR) && $(VPY) -m data.ood

ood-eval: ## Required gate: score SCRIBE, fast tier and cascades on the reviewed OOD set -> results/ood/
	cd $(ML_DIR) && $(VPY) -m eval.ood_eval --with-baseline

style-aug: ## P3.8: informal-style paraphrases of ~25% of train via Groq (resumable; days on the free tier) -> datasets/synthetic/style_aug.jsonl
	cd $(ML_DIR) && $(VPY) -m data.style_aug

spam-filter: ## P3.9: train the binary spam filter, thresholds on spam_val -> models/spam_filter/, results/spam_filter/
	cd $(ML_DIR) && $(VPY) -m triage.train_spam_filter

seed: ## Seed demo data (implemented in P8.1)
	@echo "seed: not implemented yet (P8.1)"; exit 1

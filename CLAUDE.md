# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

The full build spec is `BUILDPROMPT.md`. It covers the phases, the "Done when" criteria, the status-update format and the ground rules. Read it first if this is a new session. Then check `git log --oneline` against the status board below and continue from **In progress**.

## 1. Project summary

GroundedTriage is a capstone project: an agentic support-triage system with pluggable ingestion and retrieval-grounded reply generation. Its base paper is SCRIBE (Procedia CS 283, 2026), which uses regex rules, BART-MNLI zero-shot classification, twitter-roberta sentiment and ungrounded LLaMA-3 replies, with CSV-only ingestion.

What this system adds over SCRIBE:
1. pluggable ingestion (IMAP, Zendesk, CSV);
2. a TF-IDF+LR fast tier with a tuned threshold τ;
3. a fine-tuned DeBERTa-v3 deep tier;
4. an ordinal urgency model;
5. pgvector-grounded replies with citations;
6. a faithfulness gate with an abstain path;
7. an evaluation harness that compares the system against SCRIBE.

The Go side (Gin, pgx) handles ingestion, routing, the API/UI and workers. The Python side (`ml-service/`, FastAPI) handles ML, RAG and evaluation. The database is Supabase Postgres with pgvector.

## 2. How to run

Everything runs natively on macOS arm64, with no Docker. Copy `.env.example` to `.env` and fill it in.

| Task | Command |
|---|---|
| First-time setup (venv at `ml-service/.venv`, Python 3.11, Go modules, `models/` + `datasets/` dirs) | `make setup` |
| Start api + worker + ml-service (honcho, `Procfile`) | `make dev` |
| Single service | `make api` / `make worker` / `make ml` (uvicorn on `:8001`) |
| Unit tests (Go + Python; slow/integration skipped) | `make test` |
| One Go test | `go test ./internal/<pkg> -run TestName` |
| One Python test | `cd ml-service && .venv/bin/pytest tests/test_x.py::test_name` |
| Slow (model-loading) Python tests | `cd ml-service && RUN_SLOW=1 .venv/bin/pytest -m slow` |
| Integration tests (needs `TEST_DATABASE_URL`) | `make test-integration` |
| Lint (gofmt check, go vet, ruff check + format check) | `make lint` |
| Auto-format | `make fmt` |
| Migrate | `make migrate` (not implemented until P0.2) |
| Seed | `make seed` (not implemented until P8.1) |
| Evaluate | not implemented yet (Phase 2/7) |

## 3. Status board

### Done
- P0.1 Repo scaffold: Go module, Python package, Makefile, Procfile, `.env.example`, CLAUDE.md, README stub. Commit `886482d`.
- P0.1a Review fixes: `.env` is no longer included by make; programs load it themselves; python-dotenv is used in the pytest conftest; ignore rules re-verified. Commit: see `git log --grep "P0.1a"` (hash backfilled next commit).

### In progress
- none. Next up is P0.2.

### To do
- P0.2 Database + migrations (Supabase)
- P0.3 ml-service skeleton + Go client
- P1.1 Source interface + Ticket model
- P1.2 CSV adapter
- P1.3 IMAP adapter
- P1.4 Ticketing API adapter (Zendesk; Freshdesk optional)
- P1.5 Preprocess & normalize
- P1.6 Ingestion worker
- P2.1 Dataset builders
- P2.2 SCRIBE baseline
- P2.3 Baseline evaluation snapshot
- P3.1 Fast tier
- P3.2 Deep-tier inference loader (no training)
- P3.3 Urgency inference loader + heuristic fallback (no training)
- P3.4 Cascade + threshold tuning
- P3.5 Go integration
- P4.1 Rule engine
- P4.2 Mute + no-reply detection
- P4.3 Queues + SLA escalation
- P4.4 Meeting scheduler
- P5.1 KB ingestion
- P5.2 Retriever
- P5.3 Draft generator
- P5.4 Faithfulness gate + abstain
- P5.5 Pipeline wiring
- P6.1 REST API
- P6.2 Agent review UI
- P6.3 Outbound reply
- P6.4 Dashboard
- P7.1 Classification evaluation: baseline vs proposed
- P7.2 RAG evaluation
- P7.3 Monitoring
- P7.4 Results report
- P8.1 Demo setup
- P8.2 Cloud deploy
- P8.3 Docs

### Blocked / needs input
- P0.2 needs `DATABASE_URL` (Supabase **Session pooler** string) and `TEST_DATABASE_URL` in `.env`.
- P0.3 live Groq calls need `GROQ_API_KEY`. Tests use `FakeLLM` and don't need it.

## 4. Model artifacts

Arya fine-tunes these on a separate GPU PC; this repo only loads them. **Never write training code for them.** The artifact contract will go in `docs/model_artifacts.md` (P3.2).

| Dir | Status | Base model | Trained at | Val metrics |
|---|---|---|---|---|
| `models/deep_tier/` | absent | — | — | — |
| `models/urgency/` | absent | — | — | — |
| `models/fast_tier/` | absent (trained in-repo in P3.1) | TF-IDF + LR | — | — |

## 5. Decisions log

- 2026-10-15: Go module path is `github.com/RishabhWDB/GroundedTriage`. `go.mod` declares `go 1.22`, the spec's floor, so the other machine can build it; Go 1.26.4 is installed locally.
- 2026-10-15: ruff, pytest and honcho are installed into `ml-service/.venv` as `[dev]` extras, not globally. The Makefile calls them via `.venv/bin/`.
- 2026-10-15: Python runtime dependencies are added by the part that first needs them, which keeps `make setup` light on 8 GB RAM. Heavy ML dependencies (torch, transformers) will be optional extras.
- 2026-10-15: Slow tests are skipped unless `RUN_SLOW=1`. Integration tests are skipped when `TEST_DATABASE_URL` is unset. Both rules live in `ml-service/tests/conftest.py`.
- 2026-10-15: `make migrate` and `make seed` exit 1 with a "not implemented" message until their parts land, so they can't silently look like they succeeded.
- 2026-10-15: `.idea/` is gitignored (PyCharm local state). `BUILDPROMPT.md` is committed as the spec of record.
- 2026-10-15 (P0.1a, review feedback): the Makefile no longer `include`s `.env`. Make parsed it as Makefile syntax, which breaks on `$`, `#` and URL-encoded passwords. Each program loads `.env` itself instead:
  - Go uses godotenv (added in P0.2 with the first config code);
  - Python uses python-dotenv, and pydantic-settings from P0.3; pytest's conftest already loads it;
  - honcho reads `.env` for `make dev`.
  - The review suggested `set -a; . ./.env; set +a` in recipes. I didn't adopt it: shell sourcing also expands unquoted `$` and breaks on `&`/`;`, and it's redundant once each program loads `.env` itself. So no recipe sources `.env`. Values containing `$` go in single quotes, which godotenv, python-dotenv and the shell all treat literally.
  - `ML_PORT` for `make ml` is a make variable (default 8001; override with `make ml ML_PORT=...`).
- 2026-10-15: The Done entry's commit hash is backfilled in the following part's commit (no amending, per the no-history-rewrite rule).

## 6. Conventions

- **Commits:** `<type>(<scope>): <summary>  [P<phase>.<part>]`, with a bullet body. Types are `feat|fix|test|docs|chore|refactor`. Commit once per part. Never push, branch or rewrite history; Arya pushes.
- **Before committing:** `make lint` and `make test` must pass.
- **Layout:** follows BUILDPROMPT §2:
  - Go entrypoints in `cmd/`, packages in `internal/`;
  - Python in `ml-service/{app,triage,rag,baseline,data,eval,tests}`;
  - SQL in `migrations/`, YAML in `config/`;
  - results in `results/` (committed);
  - `models/`, `datasets/raw|processed|export` gitignored.
- **Shared contracts** (Ticket, Classification, Route, RetrievedChunk, ReplyDraft) are defined once per language: Go structs in `internal/` and Pydantic in `ml-service/app/schemas.py`. Keep them in sync.
- **Test markers:** `@pytest.mark.slow` for tests that load large models, `@pytest.mark.integration` for tests that need the DB. Go integration tests use the `integration` build tag.
- **External services** (Groq, IMAP/SMTP, Zendesk, Google Calendar) sit behind interfaces with fakes. Unit tests run offline.
- **Env vars:** see `.env.example` (DATABASE_URL, TEST_DATABASE_URL, GROQ_*, ML_SERVICE_URL, ML_PORT, TORCH_DEVICE, API_KEY, IMAP_*, SMTP_*, ZENDESK_*, GOOGLE_*).
- **Model RAM:** load models lazily. The SCRIBE baseline (BART-MNLI, about 1.6 GB) runs as its own CLI, never inside the always-on ml-service.

## 7. Known issues and gotchas

- **Supabase connection mode:** use the **Session pooler** connection string (IPv4, port 5432). The Transaction pooler (6543) breaks prepared statements in pgx/psycopg.
- **Supabase pausing:** free-tier projects pause after inactivity. If DB connections fail, check whether the project needs resuming in the dashboard.
- **Database safety:** never run destructive SQL (DROP, TRUNCATE, down-migrations) against `DATABASE_URL` without asking Arya. Tests only use `TEST_DATABASE_URL`, with a per-run schema.
- **`.env` quoting:** put any value containing `$` in single quotes. Make never reads `.env` (see the decisions log).
- **Dev machine:** an M1 MacBook Air with 8 GB RAM. Avoid holding several large models in memory.

## 8. Results so far

None yet.

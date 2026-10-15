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
| Single service | `make api` (Gin on `:8080`, `API_PORT`) / `make worker` / `make ml` (uvicorn on `:8001`) |
| Health checks | `curl localhost:8001/health` (ml-service), `curl localhost:8080/health` (api, includes ml-service) |
| Live Go → ml-service test (needs `make dev` or `make ml` running) | `go test -tags live -count=1 ./internal/mlclient -run TestLiveHealth` |
| Unit tests (Go + Python; slow/integration skipped) | `make test` |
| One Go test | `go test ./internal/<pkg> -run TestName` |
| One Python test | `cd ml-service && .venv/bin/pytest tests/test_x.py::test_name` |
| Slow (model-loading) Python tests | `cd ml-service && RUN_SLOW=1 .venv/bin/pytest -m slow` |
| Integration tests (needs `TEST_DATABASE_URL`) | `make test-integration` |
| Lint (gofmt check, go vet, ruff check + format check) | `make lint` |
| Auto-format | `make fmt` |
| Migrate (goose, up only, idempotent; SQL in `migrations/`, embedded) | `make migrate` / `make migrate-status` |
| One Go integration test | `go test -tags integration -count=1 ./internal/store -run TestTicketInsertAndGet` |
| Seed | `make seed` (not implemented until P8.1) |
| Evaluate | not implemented yet (Phase 2/7) |

## 3. Status board

### Done
- P0.1 Repo scaffold: Go module, Python package, Makefile, Procfile, `.env.example`, CLAUDE.md, README stub. Commit `886482d`.
- P0.1a Review fixes: `.env` is no longer included by make; programs load it themselves; python-dotenv is used in the pytest conftest; ignore rules re-verified. Commit `69d82e5`.
- P0.2 Database + migrations:
  - goose migrations (`vector` extension plus 7 core tables, with an HNSW cosine index on `kb_chunks.embedding`), `cmd/migrate`, `store.Tickets`;
  - `storetest.NewSchema` creates a migrated schema per test and drops it afterwards;
  - applied to Supabase, and a second run was a no-op.
  - Commit `6c5a9f9`.
- P0.2a Review fixes:
  - an explicit `search_path` (`public, extensions`) on every Go connection;
  - one shared test schema per package via `TestMain` (`storetest.Run`);
  - a cosine-query regression test.
  - Commit `9d3206c`.
- P0.3a ml-service skeleton (Python):
  - FastAPI `/health` (reports nothing loaded yet);
  - Pydantic contracts in `app/schemas.py`;
  - `GroqLLM` (httpx, backoff) and `FakeLLM`;
  - lazy-torch `select_device`;
  - pydantic-settings config.
  - Commit `eec9c9a`.
- P0.3b Go side:
  - all contract structs in `internal/model`;
  - `internal/mlclient` (per-attempt timeout, retries on transport/429/5xx);
  - Gin API stub whose `/health` includes ml-service health;
  - an idle worker stub;
  - `make dev` verified live.
  - Commit `732cc84`.
- P0.3c LLM config after the Phase 0 review:
  - drafter `openai/gpt-oss-120b` (reasoning low) and judge `qwen/qwen3.8-27b` (reasoning medium) via `build_llm(settings, role)`;
  - `max_tokens` 4096, with a truncation error;
  - per-call token-usage logging.
  - Commit `84f7d55`.
- P1.1 Source interface and registry:
  - `ingest.Source` (Name/Fetch/Reply over `model.Ticket`);
  - `Registry` builds the enabled entries of `config/sources.yaml`;
  - `FakeSource` records replies and filters Fetch by `since`.
  - Commit: see `git log --grep "P1.1"` (hash backfilled next commit).

### In progress
- none. Next is P1.2.

### To do
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
- Arya's local `.env` still has `GROQ_MODEL=llama-3.3-70b-versatile`, which overrides the new default. It needs changing to `openai/gpt-oss-120b`, or the line deleting.

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
- 2026-10-15 (P0.2): Migrations use goose v3, with SQL embedded via `migrations/embed.go` and run by `cmd/migrate`, so there's no CLI to install. Only `up` and `status` are exposed; down-migrations are not reachable from code.
  - Alternative considered: golang-migrate. goose's embed + Provider API is simpler.
- 2026-10-15 (P0.2): `go get` of current pgx v5.11, goose v3.29 and golang.org/x/* raised `go.mod` to `go 1.26.0`. This was accepted: Go ≥1.21 auto-downloads the toolchain (GOTOOLCHAIN=auto).
- 2026-10-15 (P0.2): Go contract structs live in `internal/model` (Ticket so far; the rest come in P0.3). This avoids import cycles between ingest, store and pipeline.
- 2026-10-15 (P0.2): Schema choices:
  - kb ids are text (`kb_documents.id` is the source doc id; `kb_chunks.id` is a stable chunk id), so citations are `text[]`;
  - every table has a uuid or identity primary key plus `created_at`;
  - `classifications` keeps history (no unique `ticket_id`) and has a `low_confidence` column for the cascade flag;
  - the embedding dimension is fixed at 384 in SQL, so changing the embedding model needs a new migration.
- 2026-10-15 (P0.2a, review): Every Go connection runs `SET search_path TO [<test schema>,] public, extensions` in AfterConnect (`store.SearchPath`). The Supabase role default isn't relied on. Python psycopg connections must do the same once they exist; none exist yet, and the first comes in P5.1. `TestCosineQueryResolvesVector` guards against regression: I removed `extensions` and confirmed the run fails.
- 2026-10-15 (P0.3a): The Groq client is plain httpx against the OpenAI-compatible endpoint, not the `groq` SDK. This gives full control of retries: 429/5xx/transport errors back off exponentially with jitter (up to 5 retries, 30s cap) and honour `Retry-After`; other 4xx errors fail immediately. Sleep and transport are injectable for tests.
- 2026-10-15 (P0.3a, review): torch is not a dependency yet. `app.device.select_device` imports it lazily and returns `cpu` with a log message when it's missing. `/health` only probes `find_spec("torch")` and never imports torch.
- 2026-10-15 (P1.1): Each source's `Name()` is its type (`csv`/`email`/`zendesk`), which is also `Ticket.Source`. The registry allows at most one enabled source per type, so outbound (P6.3) can resolve the reply adapter from `Ticket.Source` alone. Multiple instances of one type (e.g. two IMAP inboxes) would need a `source_instance` column; that's out of scope.
- 2026-10-15 (P1.1): Sources are configured in `config/sources.yaml`, which holds non-secret options only; adapters read credentials from env. Unknown YAML keys are rejected (`KnownFields`) to catch typos. Disabled entries need no registered factory.
- 2026-10-15 (Phase 0 review): **Documented deviation from SCRIBE.** `openai/gpt-oss-120b` (Groq) is the LLM for BOTH the SCRIBE baseline's draft→refine and this system's generator. SCRIBE used LLaMA-3-70B, which isn't available on the Groq key (404). Using the same LLM on both sides isolates the effect of grounding in P7.2. This must also be stated in `results/REPORT.md`. An optional P7 extra: a LLaMA-70B baseline run via another OpenAI-compatible provider if one is free, which is why `GROQ_BASE_URL` is configurable.
- 2026-10-15 (Phase 0 review): The faithfulness judge is configured separately (`JUDGE_MODEL`, `JUDGE_BASE_URL`, `JUDGE_API_KEY`, `JUDGE_REASONING_EFFORT`). It defaults to `qwen/qwen3.8-27b`, a different model family from the drafter, to reduce self-agreement bias.
  - P7.2 adds judge validation: Arya hand-labels 60–100 claims with a labeling CSV/CLI to be built. Report judge-vs-human accuracy and Cohen's κ for qwen and gpt-oss-120b, then pick the better judge and log the choice. Until then, use qwen.
- 2026-10-15 (P0.3c): Reasoning effort defaults are low for drafting and medium for judging. A live probe confirmed both models accept low/medium and return reasoning in a separate `reasoning` field, so `content` is clean JSON. qwen3 does no reasoning when the effort is omitted.
  - `max_tokens` defaults to 4096 (`LLM_MAX_TOKENS`, `JUDGE_MAX_TOKENS`). `finish_reason=length` raises `LLMError` instead of returning truncated JSON.
  - Usage is logged per call (prompt/completion/reasoning/total) and accumulated in `GroqLLM.usage`.
- 2026-10-15 (Phase 0 review): Go integration setup of ~26s per package is accepted, so the design stays as it is. The TestClient deprecation warning is left alone; fixing it means installing the `httpx2` package, which isn't a one-line change.
- 2026-10-15 (P0.3b): The worker stub blocks until SIGINT/SIGTERM, because honcho stops every process when one exits. The api's `/health` calls the ml-service with no retries and a 3s timeout, and reports `"unreachable"` rather than failing.
- 2026-10-15 (P0.3b): The Go build tag `live` is for tests that need running services. They're excluded from `make test` and `make test-integration`.
- 2026-10-15 (P0.3a): The `Classification` contract gains `low_confidence: bool`, which the cascade sets when it accepts the fast tier below τ without a deep tier. Routing reads it in P4.1, and it matches the `classifications.low_confidence` column.
- 2026-10-15 (P0.2a, review): Integration tests share one migrated schema per package (`TestMain` → `storetest.Run`) and isolate rows with `storetest.UniqueID` or `testDB.Truncate` (truncate refuses non-`test_` schemas). This replaced the per-test schema.
- 2026-10-15: The Done entry's commit hash is backfilled in the following part's commit (no amending, per the no-history-rewrite rule).

## 6. Conventions

- **Commits:** `<type>(<scope>): <summary>  [P<phase>.<part>]`, with a bullet body. Types are `feat|fix|test|docs|chore|refactor`. Commit once per part. Never push, branch or rewrite history; Arya pushes.
- **Before committing:** `make lint` and `make test` must pass.
- **Ingestion:** new adapters implement `ingest.Source` and register a `Factory` under their type. Use `ingest.FakeSource` in tests that need a source. Adapter output files go under `datasets/out/`, which is gitignored.
- **Layout:** follows BUILDPROMPT §2:
  - Go entrypoints in `cmd/`, packages in `internal/`;
  - Python in `ml-service/{app,triage,rag,baseline,data,eval,tests}`;
  - SQL in `migrations/`, YAML in `config/`;
  - results in `results/` (committed);
  - `models/`, `datasets/raw|processed|export` gitignored.
- **Shared contracts** (Ticket, Classification, Route, RetrievedChunk, ReplyDraft) are defined once per language: Go structs in `internal/` and Pydantic in `ml-service/app/schemas.py`. Keep them in sync.
- **Test markers:** `@pytest.mark.slow` for tests that load large models, `@pytest.mark.integration` for tests that need the DB.
  - Go integration tests use the `//go:build integration` tag. Each package has one `TestMain` calling `storetest.Run(m, &testDB)`, and tests call `testDB.Require(t)`, which skips when `TEST_DATABASE_URL` is unset.
- **DB connections:** always go through `store.OpenPool`/`store.OpenSQL`, which set an explicit search_path. Python must `SET search_path TO public, extensions` on connect.
- **External services** (Groq, IMAP/SMTP, Zendesk, Google Calendar) sit behind interfaces with fakes. Unit tests run offline.
- **Env vars:** see `.env.example` (DATABASE_URL, TEST_DATABASE_URL, GROQ_*, ML_SERVICE_URL, ML_PORT, TORCH_DEVICE, API_KEY, IMAP_*, SMTP_*, ZENDESK_*, GOOGLE_*).
- **Model RAM:** load models lazily. The SCRIBE baseline (BART-MNLI, about 1.6 GB) runs as its own CLI, never inside the always-on ml-service.

## 7. Known issues and gotchas

- **Supabase connection mode:** use the **Session pooler** connection string (IPv4, port 5432). The Transaction pooler (6543) breaks prepared statements in pgx/psycopg.
- **pgvector on Supabase** is pre-installed in the `extensions` schema (v0.8.2), not `public`. Any connection that overrides `search_path` must include `extensions`, or you get `type "vector" does not exist`.
- **Integration test cost:** each package pays about 26s once, to create, migrate and drop its schema through the Supabase pooler. Each test after that takes about 2–3s.
- **Supabase pausing:** free-tier projects pause after inactivity. If DB connections fail, check whether the project needs resuming in the dashboard.
- **Groq models change:** `llama-3.3-70b-versatile` isn't available on the current key. Run `GET {GROQ_BASE_URL}/models` to see which models are live.
- **Tests vs `.env`:** the pytest conftest loads `.env` into the environment. Tests that check settings defaults must clear those variables (see `test_build_llm_roles`).
- **pytest warning:** a StarletteDeprecationWarning from TestClient asks for `httpx2`. It's harmless and left as is (review decision).
- **Render Go version:** `go.mod` declares `go 1.26.0`. At P8.2, check that Render's native Go runtime supports it, or pin a toolchain.
- **Database safety:** never run destructive SQL (DROP, TRUNCATE, down-migrations) against `DATABASE_URL` without asking Arya. Tests only use `TEST_DATABASE_URL`, with a per-run schema.
- **`.env` quoting:** put any value containing `$` in single quotes. Make never reads `.env` (see the decisions log).
- **Dev machine:** an M1 MacBook Air with 8 GB RAM. Avoid holding several large models in memory.

## 8. Results so far

None yet.

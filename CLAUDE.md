# CLAUDE.md

Guidance for Claude Code in this repository. The build spec is `BUILDPROMPT.md` (phases, "Done when" criteria, status-update format, ground rules); read it in a new session. Then compare `git log --oneline` with `docs/status.md` and continue from **In progress** there.

## Project

GroundedTriage is a capstone: an agentic support-triage system that extends SCRIBE (Procedia CS 283, 2026; regex rules + BART-MNLI zero-shot + twitter-roberta sentiment + ungrounded LLaMA-3 replies, CSV only). It adds pluggable ingestion (IMAP, Zendesk, CSV), a binary spam filter plus a 5-class topic cascade (TF-IDF+LR fast tier → fine-tuned DeBERTa-v3 deep tier below τ), an ordinal urgency model, YAML routing with SLAs and meeting scheduling, pgvector-grounded replies with [n] citations, a faithfulness gate that abstains, a human review UI, and an evaluation harness against SCRIBE. Go (Gin, pgx) does ingestion, routing, API/UI and workers; Python (`ml-service/`, FastAPI) does ML, RAG and evaluation; the database is Supabase Postgres + pgvector. **Labels** (`config/taxonomy.yaml`): 5 topic classes decided by the cascade (Technical Issue, Billing, Feature Request, General Inquiry, Meeting Scheduling), plus Spam decided by the separate binary filter. This is not SCRIBE's original category set (Work, Spam, Personal, Urgent, General, Meeting Scheduling).

## Install, run, test

No Docker. Two machines: the **Mac** (macOS arm64, 8 GB) runs only the final app (`make dev`, `make demo`); the **eval PC** (Windows WSL2 Ubuntu, RTX 3050 4 GB, CUDA) runs all training-adjacent work and evaluation (setup and the files to copy: `docs/pc_setup.md`). Copy `.env.example` to `.env` and fill it in (put values containing `$` in single quotes).

| Task | Command |
|---|---|
| Setup (venv `ml-service/.venv`, Python 3.11, Go modules) | `make setup`; ML extras: `make setup-models` / `make setup-baseline` |
| All services / one service | `make dev` / `make api` (:8080) · `make worker` · `make ml` (:8001) |
| Worker once | `go run ./cmd/worker -once` |
| Unit tests (Go + Python) | `make test` |
| One test | `go test ./internal/<pkg> -run TestName` · `cd ml-service && .venv/bin/pytest tests/test_x.py::test_name` |
| Slow / integration / live tests | `cd ml-service && RUN_SLOW=1 .venv/bin/pytest -m slow` · `make test-integration` (TEST_DATABASE_URL) · `go test -tags live -count=1 ./internal/mlclient -run Live` |
| Lint / format | `make lint` / `make fmt` |
| Migrations (up only) | `make migrate` · `make migrate-status` · `make migrate-verify` |
| Model gate before accepting anything | `make ood-eval` (ood-dev) |
| Daily background jobs | `make rag-eval`, then `make style-aug` (one at a time) |

Every other target (data, training, evals, demo, deploy) and the manual UI test: `docs/commands.md`.

## Layout

- `cmd/` Go entrypoints (api, worker, migrate, demo, dbping, calendar-auth); `internal/` Go packages (`model` contracts, `ingest`, `preprocess`, `pipeline`, `routing`, `scheduler`, `store`, `mlclient`, `api`, `ui`, `outbound`, `monitoring`).
- `ml-service/{app,triage,rag,baseline,data,eval,tests}`: FastAPI app + contracts, classifiers, RAG, SCRIBE baseline, dataset builders, evaluation.
- `migrations/` SQL (goose, embedded); `config/` YAML (taxonomy, sources, routing, SLA, cascade, grounding, monitoring); `scripts/` (jobguard, demo, Render validation).
- `results/` committed eval outputs; `datasets/` (`synthetic/`, `ood/`, `stress/`, `demo/` committed; `raw|processed|export|out` gitignored); `models/` gitignored.

## Rules and conventions

- **Two-machine rule:** `git pull` before starting on either machine; push when done (Arya pushes; Claude still never pushes, so Claude ends a session with everything committed and tells Arya to push). Never work on both machines at once. While evals run, the PC owns `results/` (and `datasets/raw/` caches): don't change or commit them on the Mac. Untracked eval inputs move by copy, not git (`docs/pc_setup.md` §4).
- **Commits:** `<type>(<scope>): <summary>  [P<phase>.<part>]` with a bullet body; types `feat|fix|test|docs|chore|refactor` only. One commit per part. **Never push, branch or rewrite history; Arya pushes.** `make lint` **and** `make test` must both pass before committing.
- **Status updates and phase reviews:** print them in chat and also write them to `summary.md` (gitignored), rewritten each time with only the latest update. Keep `docs/status.md` current (finished parts, in progress, Arya's manual checklist).
- **Never write training code for the deep tier or urgency model.** Arya trains them on a GPU PC; this repo only loads them (contract: `docs/model_artifacts.md`).
- **Shared contracts** (Ticket, Classification, Route, RetrievedChunk, ReplyDraft) exist once per language: Go `internal/model`, Pydantic `ml-service/app/schemas.py`. Keep them in sync.
- **Model input** is always `app/textinput.model_input(subject, body)` in training and serving.
- **Evaluation discipline:** macro-F1 is the headline (accuracy secondary); report overall, real-only and per origin. Choose variants, thresholds, rules and τ on val + ood-dev only. Test and ood-test are scored once per final system (`make ood-eval-test` logs every look). Rows with `needs_review=true` are never scored except in tables marked PRELIMINARY. The retrieval floor is never moved from the final eval.
- **Ingestion:** new adapters implement `ingest.Source` and register a `Factory`; use `ingest.FakeSource` in tests; adapter output goes under `datasets/out/`.
- **External services** (Groq, IMAP/SMTP, Zendesk, Google Calendar) sit behind interfaces with fakes; unit tests run offline. Test markers: `@pytest.mark.slow` (large models), `@pytest.mark.integration` (DB); Go integration tests use `//go:build integration`, one `TestMain` calling `storetest.Run(m, &testDB)`, and `testDB.Require(t)`.
- **DB connections** go through `store.OpenPool`/`store.OpenSQL` (explicit `search_path` incl. `extensions`); Python runs `SET search_path TO public, extensions` on connect.
- **Database safety:** never run destructive SQL (DROP, TRUNCATE, down-migrations, blanket deletes) against `DATABASE_URL` without asking Arya. Tests use only `TEST_DATABASE_URL`. Leave the 21 Mumbai sample tickets as they are; demo rows are removed only by `make demo-reset`.
- **Human approval stays mandatory**; nothing auto-sends replies. Ticket/draft text in the UI only via `textContent`.
- **Groq budget:** gpt-oss-120b has 200k tokens/day shared by all jobs. P7.2 first, then style-aug. Every LLM batch job must be cached and resumable.
- **Dependencies:** add Python deps in the part that first needs them; heavy ML deps stay optional extras. Env vars are documented in `.env.example`.

### Memory rules (8 GB M1; 4 GB VRAM on the PC)

- **At most ONE model-loading or LLM-batch job at a time; never run evals in parallel**, and never start a job in the background while another runs (including a long `make rag-eval` / `make style-aug`).
- **Before any model job, check memory** (macOS: `sysctl kern.memorystatus_vm_pressure_level`, `memory_pressure -Q`; Linux/WSL2: `/proc/pressure/memory`, `/proc/meminfo`, `nvidia-smi`) and **refuse to start if pressure is warn/critical** or free memory is < 20%. Log peak RSS and swap afterwards.
- **Stop `make dev` services when the step doesn't need them; kill leftover python processes when a job finishes.**
- **Jobs cap their own memory:** lazy loading, bounded batch sizes, unload after use, checkpoint long passes. At most one transformer per role, fp16 on MPS/CUDA, record RSS + swap in every eval run. The SCRIBE baseline (BART-MNLI ~1.6 GB) never loads inside the ml-service.
- `scripts/jobguard.py` enforces this on every model/LLM make target (lock, pressure check, service check, process-group cleanup, peak RSS log in `datasets/raw/jobguard.jsonl`). Run those jobs through make, not `python -m`, so the guard applies. `JOBGUARD_FORCE=1` is Arya's call only.
- **Crash-safe writes:** whole files (caches, splits, manifests, reviewed CSVs, tuned configs) via `app/fileio.atomic_write_text` / `atomic_path`; append-only JSONL via `open_append` + `append_line` and read via `read_jsonl`. New writers must use these.

## Mistakes to avoid (they happened before)

- **Committing on test results alone:** P5.2 shipped a file that failed `ruff format` because the command chain ignored the lint failure. Gate commits on `make lint && make test`.
- **Commit types outside the list:** P3.8a used `data`; data commits use `chore`.
- **Parallel model jobs:** the 2026-10-17 OOM reboot (P7.2 eval + a BART-MNLI pass). One job at a time, through the guard.
- **Invisible characters in generated Go files:** `\u` escapes can land as literal BOM/nbsp/zero-width characters, and Go rejects a mid-file BOM. Rewrite them as `\uXXXX` escapes and run `go vet` after writing such files.
- **`search_path` without `extensions`** → `type "vector" does not exist`. **Transaction pooler (6543)** breaks prepared statements; use the Session pooler (5432).
- **Making make read `.env`** (`include .env` or sourcing it in recipes) breaks on `$`, `#` and URL-encoded passwords. Programs load `.env` themselves.
- **Tests vs `.env`:** the pytest conftest loads `.env`; tests of settings defaults must clear those variables (see `test_build_llm_roles`).
- **Averaging in an absent class:** a 5-class topic report that averaged in Spam (support 0) under-reported macro-F1. Pass the topic labels.
- **float32 thresholds:** a "one ulp above" threshold missed a ticket; spam scores are float64.
- **Judge `max_tokens` above qwen's 1,000 output tokens/minute** can never succeed; keep `JUDGE_MAX_TOKENS=1000`, reasoning off, and split long drafts.
- **Reporting a SUPERSEDED number** (P2.3 baseline split and swap-bound latencies, 6-class fast tier, stale `c6539fff94a1`/`db976b0585b2` exports). Use the current export `e2552b04338b` and `docs/results.md`.
- **Running the worker against DATABASE_URL** processes the 21 sample tickets and spends tokens; test pipeline steps on TEST_DATABASE_URL.

## Further reading

- `docs/status.md`: read at the start of every session and after finishing a part (done list, in progress, Arya's manual checklist, deferred Phase 7 decisions, model artifacts present).
- `docs/pc_setup.md`: read before setting up or syncing the eval PC (WSL2, CUDA check, make targets, the files to copy from the Mac and their sizes).
- `docs/commands.md`: read when you need any make target, script or curl beyond the table above, or to run the manual UI test.
- `docs/decisions.md`: read the section for an area before changing its code or config (ingestion, data/splits, classification, evaluation protocol, routing/meetings, RAG, review/API/UI/outbound, monitoring, demo/deploy, LLM limits).
- `docs/results.md`: read before comparing a new run with earlier numbers, making a Phase 7 decision, or writing results.
- `docs/report_notes.md`: read before writing `results/REPORT.md`, the README results or any evaluation claim (mandatory statements, limitations, future work).
- `docs/environment.md`: read when Supabase, Groq, Hugging Face, Kaggle, Google/Zendesk/IMAP or the machine misbehave, or before changing external-service config.
- `docs/architecture.md`: read for the per-layer overview before a cross-cutting change.
- `docs/model_artifacts.md`: read before touching model loaders, the export, or anything Arya's PC training depends on.
- `docs/openapi.yaml`: read before adding or changing any API endpoint (a test checks every route is documented).
- `docs/deploy.md`: read before changing `render.yaml`, model bundles or deploy settings.
- `docs/demo_script.md`: read before changing demo data or the demo flow.

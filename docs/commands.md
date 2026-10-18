# Command reference

Read when you need a make target or script beyond the core ones in CLAUDE.md. Model-loading and LLM-batch targets run under `scripts/jobguard.py` (see CLAUDE.md → memory rules). Everything runs natively on macOS arm64; copy `.env.example` to `.env` first.

## Services and checks

| Task | Command |
|---|---|
| First-time setup (venv at `ml-service/.venv`, Python 3.11, Go modules, `models/` + `datasets/` dirs) | `make setup` |
| torch + transformers for the deep tier / urgency model | `make setup-models` |
| SCRIBE baseline extra (torch + transformers) | `make setup-baseline` |
| api + worker + ml-service (honcho, `Procfile`) | `make dev` |
| Single service | `make api` (Gin on `:8080`, `API_PORT`) / `make worker` / `make ml` (uvicorn on `:8001`, override `make ml ML_PORT=...`) |
| Health checks | `curl localhost:8001/health` (ml-service), `curl localhost:8080/health` (api, includes ml-service) |
| Worker: one poll + pipeline / loop | `go run ./cmd/worker -once` / `make worker` (`WORKER_POLL_INTERVAL`, default 60s; calls `ML_SERVICE_URL`) |
| Classify one ticket (ml-service running) | `curl -s localhost:8001/classify -H 'content-type: application/json' -d '{"ticket_id":"t1","subject":"Refund","body":"Charged twice"}'` |
| REST API (needs a real `API_KEY` in `.env`) | `make api`; e.g. `curl -H "X-API-Key: $KEY" localhost:8080/tickets?status=open`; spec in `docs/openapi.yaml` |
| Agent review UI | `make api` (or `make dev`), open http://127.0.0.1:8080/ (manual test below) |
| Daily monitoring rows | `curl -H "X-API-Key: $KEY" localhost:8080/metrics/daily?days=14`, or the Dashboard's monitoring table |
| DB latency diagnostic | `go run ./cmd/dbping` (`-env TEST_DATABASE_URL` for the test DB) |
| Migrations (goose, up only, SQL embedded from `migrations/`) | `make migrate` / `make migrate-status` / `make migrate-verify` (read-only: vector ext, tables, HNSW index, ticket counts) |
| Google Calendar OAuth (needs `credentials.json`, Desktop-app client) | `make calendar-auth` → `token.json` |
| Kaggle auth check | `cd ml-service && .venv/bin/python -m data.kaggle_auth` |

## Tests and lint

| Task | Command |
|---|---|
| Unit tests (Go + Python; slow/integration skipped) | `make test` |
| One Go test / one Python test | `go test ./internal/<pkg> -run TestName` / `cd ml-service && .venv/bin/pytest tests/test_x.py::test_name` |
| Slow (model-loading) Python tests | `cd ml-service && RUN_SLOW=1 .venv/bin/pytest -m slow` |
| Integration tests (needs `TEST_DATABASE_URL`) | `make test-integration`; one: `go test -tags integration -count=1 ./internal/store -run TestTicketInsertAndGet` |
| Live Go → ml-service tests (needs `make ml`; classify needs `models/fast_tier/`) | `go test -tags live -count=1 ./internal/mlclient -run Live` |
| Lint / auto-format | `make lint` (gofmt, go vet, ruff check + format check) / `make fmt` |

The UI dashboard model is tested with `node --test`, run by `go test ./internal/ui` when Node is installed (skipped otherwise).

## Data

| Task | Command |
|---|---|
| Build splits + manifest + dataset card (downloads raw data first time) | `make data` |
| Data audit (synthetic diversity, cross-split near-dups, subject shortcut) | `make data-audit` → `results/data_audit/report.md` |
| Synthetic meeting emails (Groq; cached, resumable) | `make synthetic-meetings` |
| Hard Meeting stress CSV (refuses to overwrite a reviewed file without `--force`) | `make stress-set` → `datasets/stress/meeting_stress.csv` |
| Style-aug paraphrases of 25% of train (Groq; resumable; days on the free tier) | `make style-aug` (`--recheck` drops near-copies) → `datasets/synthetic/style_aug.jsonl`, then `make data` |
| OOD set (refuses to overwrite a reviewed file) | `make ood-set` → `datasets/ood/` |
| Bundle splits for the GPU PC | `make export-splits` → `datasets/export/groundedtriage-splits-<test hash>.tar.gz` |
| PSI reference distribution (after a data rebuild) | `make monitoring-ref` → `config/monitoring.yaml` |

## Models and evaluation

| Task | Command |
|---|---|
| Fast tier (~8 min) / style-aug ablation | `make fast-tier` / `make fast-tier-aug` → `models/fast_tier[_aug]/`, `results/fast_tier[_aug]/` |
| Embedding fast tier (bge-small + LR; cache `datasets/raw/emb_cache/`) | `make fast-tier-emb` / `make fast-tier-emb-aug` |
| Spam filter + thresholds on spam_val (~1 min) | `make spam-filter` → `models/spam_filter/`, `results/spam_filter/` |
| Urgency (heuristic, plus model if present) on val | `make urgency-eval` → `results/urgency/` |
| Tune τ on val (needs `models/deep_tier/`) | `make tune-tau` → `config/cascade.yaml`, `results/cascade/` |
| **Required gate before accepting any model:** OOD eval on ood-dev | `make ood-eval` → `results/ood/dev/` |
| **Final systems only:** ood-test once (every run logged) | `make ood-eval-test` → `results/ood/test/`, `results/ood/test_log.jsonl` |
| SCRIBE baseline CLI (own process; ~2 GB weights first run) | `make baseline ARGS="classify --split test --limit 10"` / `ARGS="classify --csv ../datasets/sample_emails.csv"` / `ARGS="draft --csv ../datasets/sample_emails.csv --limit 2"` |
| Baseline eval on test (resumable) / stress set after review | `make baseline-eval` → `results/baseline/` (smoke: `make baseline ARGS="evaluate --limit 60 --out /tmp/x"`) / `make baseline-stress` |
| SCRIBE val-tuned wording (P7.1, WIP) | `make scribe-tune` → `config/scribe_tuned.yaml`, `results/scribe_tune/` |
| KB ingest into pgvector on DATABASE_URL (idempotent; ~17 min first time) | `make kb` (or `cd ml-service && .venv/bin/python -m rag.kb --folder <dir>` for .md/.html) → `results/kb/ingest.json` |
| Retrieval eval (recall@k, MRR, dense vs hybrid, floor analysis) | `make retrieval-eval` → `results/retrieval/` |
| P7.2 RAG eval: plan / floor sweep (p72_dev) / the run (cached, resumable, daily) | `make rag-eval-plan` / `make rag-floor-sweep` / `make rag-eval` (`ARGS="--limit N"`) → `results/rag/` |
| Judge validation: sample / label (interactive) / score (κ) | `make judge-sample` / `make judge-label` / `make judge-score ARGS=--with-llm` → `results/rag/judge_validation.*` |

## Demo and deploy

| Task | Command |
|---|---|
| Demo: load 25 demo tickets, run the pipeline once, expected vs actual, then `make dev` | `make demo` (= `make seed`; `DEMO_NO_DEV=1` stops before `make dev`); `make demo-status`; `make demo-reset` (only demo rows, typed confirmation); `make demo-expect` (re-measure; see `datasets/demo/README.md`) |
| Render: validate blueprint / bundle models / install a bundle | `make render-validate` / `make models-pack ARGS="fast_tier spam_filter"` / `make models-fetch` (see `docs/deploy.md`) |

## Manual UI test

1. Set a real `API_KEY` in `.env`. Run `make dev` (api + worker + ml-service) or just `make api`.
2. Open http://127.0.0.1:8080/. It redirects to `/ui/`. Enter the API key, then your name in **Reviewer** (top right).
3. **Inbox:** set the status filter to *All*. The 21 Mumbai sample tickets are listed, unclassified unless the worker has run. Check that filters and paging work, and that a breached SLA shows in red.
4. **Upload:** use *Upload CSV* with `datasets/sample_emails.csv`. Expect "0 new" (all duplicates) on the demo DB.
5. **Pipeline:** `make demo` loads the 25 demo tickets; with `make ml` running, `go run ./cmd/worker -once` classifies, routes, schedules and drafts (**this also processes the 21 older sample tickets** in the DB). `make demo-status` compares with the expected outcomes; `make demo-reset` removes only the demo rows.
6. **Ticket view:** open a ticket. Check:
   - the classification card shows the tier, low-confidence/possible-spam badges, urgency with *model* or *heuristic*, and model versions;
   - the route card shows the queue, priority, SLA and reason; Meeting tickets show the meeting outcome;
   - **Draft:** each `[n]` is a button that opens the source chunk (title, heading, URL, text). Faithfulness and unsupported claims are shown. An abstained draft shows the reason and has no text.
7. **Actions:** *Edit* → change the text → *Save* (badge "edited") → *Approve* (status approved). On another draft, *Reject* with a note. Approve should be disabled on an unedited abstained draft.
8. **Safety:** a ticket whose subject or body contains `<script>…</script>` must show it as literal text.
9. **Dashboard:** open *Dashboard*. Check the tiles and bars against the inbox counts. *Models loaded* reads "ml-service unreachable" unless `make ml` is running.

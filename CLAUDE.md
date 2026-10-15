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
| DB latency diagnostic (region, TCP, fresh vs warm round-trips, pool reuse) | `go run ./cmd/dbping` (`-env TEST_DATABASE_URL` for the test DB) |
| Kaggle auth check | `cd ml-service && .venv/bin/python -m data.kaggle_auth` |
| Worker: one poll / loop | `go run ./cmd/worker -once` / `make worker` (loop, `WORKER_POLL_INTERVAL`, default 60s) |
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
  - Commit `897ed7a`.
- P1.2 CSV adapter:
  - `ingest.CSVSource` reads Sender/Subject/Body plus optional Label/Date/Id, case-insensitive and BOM-tolerant;
  - Reply appends to `datasets/out/replies.csv`;
  - `ingest.DefaultRegistry()` registers the adapters;
  - committed a hand-written 22-row `datasets/sample_emails.csv`.
  - Commit `7640cb1`.
- P1.3a Email parsing and shared text cleaning:
  - `preprocess.HTMLToText`, `StripQuoted`, `StripSignature`;
  - `ingest.ParseEmail`: multipart, charsets, threading headers;
  - 4 `.eml` fixtures in `internal/ingest/testdata/email/`.
  - Commit `67c7027`.
- P1.3b Email source:
  - `EmailSource` over the `IMAPFetcher`/`Mailer` interfaces;
  - real `IMAPClient` (go-imap v1, read-only, BODY.PEEK) and `SMTPMailer` (465 TLS / STARTTLS, PLAIN auth);
  - threaded `BuildReply`;
  - `email` registered in DefaultRegistry.
  - Commit `088614a`.
- P1.4 Zendesk adapter:
  - `ZendeskSource` uses the cursor-based incremental export with users sideloaded and pagination capped by `MaxPages`;
  - replies are a public comment via PUT;
  - 429 retries honour `Retry-After`.
  - Commit `17e198d`.
- P1.5 Preprocess & normalize:
  - `preprocess.Normalize`: HTML→text, quote/signature stripping, whitespace/NFC, sender lower-casing, content hash, empty and non-English skips;
  - `preprocess.Ingestor` persists new tickets and rejects duplicates by `(source, external_id)` and by `content_hash`.
  - Commit `ba7404d`.
- P1.6 Ingestion worker:
  - `cmd/worker` polls the enabled sources in `config/sources.yaml` every `WORKER_POLL_INTERVAL` (or once, with `-once`);
  - `pipeline.Poller` stores per-source checkpoints (`ingest_checkpoints`, migration 00003) with a 10-minute overlap and writes `audit_log` `ingest.poll`/`ingest.error` entries;
  - graceful shutdown.
  - Verified on Supabase: run 1 inserted 20 tickets, run 2 inserted 0.
  - Commit `f73048c`.
- P1.7 Phase 1 review fixes:
  - `cmd/dbping` latency diagnostic;
  - sender normalised to a bare lower-cased address before hashing;
  - non-English tickets stored with `ingest_status='skipped'`, `skip_reason='non_english'` (migration 00004).
  - Commit `39ee538`.

- P2.1a Kaggle token auth:
  - `data.kaggle_auth` (`get_api`, `download_dataset`; `python -m data.kaggle_auth` checks auth);
  - `kaggle>=2.2.4` with KAGGLE_API_TOKEN read from `.env`;
  - a `[data]` extra (kaggle, pandas, pyarrow) installed by `make setup`.
  - Commit `bb76324`.
- P2.1b Taxonomy change to 6 categories:
  - `config/taxonomy.yaml` (categories with zero-shot descriptions, urgency scale and trained levels, dropped categories, dataset mappings);
  - `app/taxonomy.py`, with `Classification.category` validated against it;
  - Go `model.Categories`, sync-tested against the YAML;
  - the sample CSV relabelled.
  - Commit `2831375`. Follow-up `c5389cd` removed a stray Kaggle metadata file.
- P2.1c Tobi-Bueck builder:
  - `data/tobi_bueck.py` (cached HF parquet download, EN filter, `taxonomy.yaml` rules, urgency map, cleaning, dedupe);
  - `data/mapping.py` (ordered-rule engine);
  - `data/textclean.py` (mirrors the Go preprocess cleaning).
  - Real run: 23,228 rows out.
  - Commit `a11e653`.
- P2.1d Spam builder: `data/spamassassin.py` reads the SpamAssassin public corpus (spam sets only) from the tar stream in memory, parses with stdlib email, keeps English only (langdetect, seeded), cleans, dedupes and truncates. Real run: 1,553 rows.
  - Commit `1153cfe`.
- P2.1e Synthetic meeting requests:
  - `data/synthetic_meetings.py` (seeded specs, fixed prompt, gpt-oss-120b, 5 per call, sha256-keyed response cache, validation and retry);
  - 400 rows committed in `datasets/synthetic/meeting_requests.jsonl`.
  - Commit: see `git log --grep "P2.1e"` (hash backfilled next commit).

### In progress
- P2.1 Dataset builders. a through e are done. `data/build.py`, `data/export.py`, their tests and `docs/model_artifacts.md` are written but uncommitted. Next:
  - P2.1f: splits, dataset card, `make data`, `make export-splits`.
  - Downloaded `Tobi-Bueck/customer-support-tickets` (HF, CC-BY-NC-4.0) to `datasets/raw/tobi_bueck/train.parquet` (gitignored).
  - Profile of the English rows (28,261 of 61,765):
    - only 10 support queues; Tech/Product/IT/Outage queues total about 17.9k;
    - priority is only low 5,774 / medium 11,570 / high 10,917, with **0 critical**;
    - 4,595 duplicate bodies (dedupe before splitting);
    - 12.9% have no subject;
    - some bodies contain `<br>`;
    - the text is LLM-generated and the queue labels are weakly tied to content.

### To do
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
- none.

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
- 2026-10-15 (P2.1 review, **supersedes BUILD_PROMPT's 11 categories**): the taxonomy is now **6 support-triage categories**: Technical Issue, Billing, Feature Request, General Inquiry, Spam, Meeting Scheduling.
  - Dropped: Work and Personal (not support topics, no clean labels); Urgent (urgency is its own axis); Complaint (a tone/escalation signal, not a topic); Account/Access (no clean labels; folded into Technical Issue).
  - Rationale: avoid evaluating mostly on synthetic data.
  - The SCRIBE baseline is evaluated on the same 6 classes via the zero-shot descriptions in `taxonomy.yaml`, and its regex rules still cover Spam and Meeting Scheduling. **The report must state this.**
  - The sample CSV labels were remapped (Account/Access→Technical Issue; Complaint, Personal, Work and General→General Inquiry; Urgent→Technical Issue for the DB-down alert, otherwise General Inquiry).
  - P4.1 routing defaults must use these 6 categories, plus `ingest_status=skipped` → human queue.
- 2026-10-15 (P2.1 D1): `albertobircoci/support-ticket-priority-dataset-50k` is **dropped**. It has no text column (50,000 × 33 tabular features; only `description_length`). Its priority is exactly 50/35/15% low/medium/high with no critical. The download was deleted.
- 2026-10-15 (P2.1 D2): Tobi-Bueck (EN only) mapping rules, first match wins (in `taxonomy.yaml`):
  - type=Change → Feature Request;
  - Billing and Payments, Returns and Exchanges → Billing;
  - Technical/Product/IT Support, Service Outages and Maintenance → Technical Issue;
  - General Inquiry, Sales and Pre-Sales, Customer Service → General Inquiry;
  - Human Resources → dropped.
  - Because Change comes first, HR rows with type=Change (39) become Feature Request; only the remaining HR rows are dropped. Priority maps very_low/low→low, medium, high, critical.
- 2026-10-15 (P2.1 D3):
  - Spam = a real public spam corpus, spam rows only (picked in P2.1d after a licence check).
  - Meeting Scheduling = about 400 synthetic support-context meeting requests from gpt-oss-120b, with fixed seed and prompt committed, generated once and cached.
  - Every row carries `origin` (real|synthetic), and all metrics must be reportable overall and split by origin.
  - Train only: cap any class at about 5× the smallest class. val/test are never resampled.
  - Splits are stratified by category × origin: 70/15/15, seed 42.
- 2026-10-15 (P2.1c): Results of the real Tobi-Bueck build (raw parquet sha256 `1e2de97c…71a9`):
  - 61,765 rows → 28,261 EN;
  - 514 dropped by the HR rule, 0 unmapped, 1 empty body, 4,518 duplicate bodies removed → **23,228 rows**;
  - categories: Technical Issue 13,349, General Inquiry 4,099, Billing 3,289, Feature Request 2,491;
  - urgency: medium 9,544, high 9,071, low 4,613.
  - The dedupe key is the lower-cased, whitespace-collapsed body, keeping the first occurrence.
  - The dataset's anonymisation placeholders (`<tel_num>`, `<acc_num>`, `<name>`, …) are kept as is.
  - Literal `\n` sequences in bodies become newlines.
- 2026-10-15 (P2.1d): **The spam corpus is SpamAssassin's public corpus**, `20030228_spam` + `20050311_spam_2` (the latest version of each spam set; ham isn't used).
  - Licence: no formal licence. It's offered "as a service to spam filter developers", and copyright stays with the senders. We download by script and never commit or redistribute it; cite the Apache SpamAssassin project in the report.
  - It was chosen over Enron-Spam (Metsis et al. 2006, no licence statement either) for clearer provenance and stated purpose, and because its raw RFC 822 emails match our email-shaped inputs.
  - Archives are untrusted: they're read from the tar stream in memory, never extracted.
  - Results: 1,896 messages → 27 too short (<20 chars) / 69 non-English / 247 duplicates removed → **1,553 rows**. 127 bodies were truncated to 4,000 chars.
  - Spam rows have **no urgency label** (NA) and are excluded from urgency training/eval.
  - Caveat: the spam is 2002–2005 era.
- 2026-10-15 (P2.1e): How the synthetic Meeting Scheduling rows were made:
  - 400 rows from gpt-oss-120b (temperature 0.9, JSON mode, reasoning effort low), 5 emails per call (80 batches; 1 batch retried after returning 4 emails). Total 80,308 tokens.
  - Specs come from `random.Random(42)` over committed option lists: action × meeting type × sender role × product × time phrasing × duration × platform × tone × length. The prompt is in the module. Raw responses are cached in `datasets/raw/llm_cache/synthetic_meetings/` (gitignored).
  - **The final rows are committed** (`datasets/synthetic/`, about 150 KB). They're our own generated data, not a downloaded dataset, and a re-run at temperature 0.9 wouldn't reproduce them byte-for-byte.
  - Quality check: 400 unique bodies; 0 bracket placeholders; 0 "finds you well"; 97% mention a meeting word; 323 unique subjects.
  - **Caveat for the report:** the emails are shorter than Tobi-Bueck's (median 260 vs 374 chars) and their subjects are keyword-heavy ("confirmation", "reschedule", "review"). The class is likely easier than the real ones, so report per-origin metrics.
  - Urgency is NA for these rows.
- 2026-10-15 (P2.1 D4): urgency is trained and evaluated on low/medium/high only; "critical" comes only from documented heuristic/escalation rules (P3.3). Neither dataset has English critical rows.
- 2026-10-15: Supabase was recreated in **ap-south-1 (Mumbai)**. Migrations 1–4 applied, and the sample CSV was re-seeded by the worker (21 tickets incl. the Portuguese one as skipped). The Portuguese backfill question is moot.
- 2026-10-15 (P2.1a): Kaggle uses a new-style API token (`KGAT_…`) in `KAGGLE_API_TOKEN`. `kaggle` 2.2.4 reads it natively via `kagglesdk` (verified in the package source).
  - `import kaggle` authenticates at import time, and `authenticate()` calls `exit(1)` on failure. `data.kaggle_auth` therefore loads `.env` first, imports lazily, silences the SDK's help text and raises `KaggleAuthError` with exact fix-it steps.
  - It also diagnoses a malformed token (stray prefix characters, quotes, whitespace) without printing it.
- 2026-10-15 (P1.7, review): The sender is normalised by `preprocess.NormalizeSender` before hashing and storing: `net/mail` parse → bare address, lower-cased; display names, angle brackets, quotes and `mailto:` stripped. Values with no address (`zendesk-user:42`) are trimmed and lower-cased. `ContentHash` normalises internally too, so a display name never changes the hash.
- 2026-10-15 (P1.7, review): Non-English tickets are **stored**, not dropped, with `tickets.ingest_status='skipped'` and `skip_reason='non_english'` (migration 00004; default `accepted`). Empty tickets are still dropped (nothing to show).
  - The column is named `ingest_status` rather than `status`, so it isn't confused with route/draft statuses.
  - **P3.5 must only classify `ingest_status='accepted'` tickets. P4 must route skipped tickets to a human queue.**
- 2026-10-15 (P1.7): Migrations containing `DO $$ … $$` blocks need `-- +goose StatementBegin/End`, because goose splits on semicolons. A failed migration rolls back, since each file runs in a transaction.
- 2026-10-15 (P1.6): Each poll fetches with `since = checkpoint − 10 min` (the overlap) and then sets the checkpoint to the max `ReceivedAt` it fetched (never "now", which avoids clock-skew gaps).
  - The overlap re-reads late arrivals, and dedupe drops the repeats.
  - A failing source is logged and audited (`ingest.error`). It doesn't stop the other sources, and its checkpoint isn't advanced.
  - Each source and poll gets one `ingest.poll` audit row with its counts and up to 100 inserted ticket ids. There's no per-ticket row, because each would cost a pooler round-trip.
  - A poll in progress at SIGTERM runs to completion (bounded by `PollTimeout`, 5 min). The Poller lives in `internal/pipeline` because later phases extend it with classify/route/reply.
- 2026-10-15 (P1.6): The 20 sample tickets (and their checkpoint and audit rows) now exist in the main Supabase DB from the P1.6 acceptance run. That's expected demo data; P8.1's seed uses the same CSV.
- 2026-10-15 (P1.5): `content_hash` is sha256 over three parts: the lower-cased sender; the subject with Re:/Fwd:/AW:/WG: prefixes stripped, lower-cased, whitespace collapsed; and the lower-cased body with whitespace collapsed.
  - The sender is included so that two customers sending the same text are **not** merged. The goal is to catch the same message arriving twice (re-sends, or the same mail via email and Zendesk).
  - Tickets whose content hash is already stored are rejected as `duplicate_content`, even with a new external id. The check-then-insert isn't atomic, which is safe with a single ingestion worker (as now).
- 2026-10-15 (P1.5): Language check uses `abadojack/whatlanggo` (a small trigram detector, no model files; lingua-go was rejected for its RAM use).
  - A ticket is skipped as `non_english` only when the detection is reliable, isn't English, and the subject+body has at least 30 letters. Shorter text is assumed English.
  - On the 22 sample rows, only the Portuguese row is flagged. Three short English rows got low-confidence `eng`, never a wrong language.
- 2026-10-15 (P1.5): `preprocess.CleanBody` is the single cleaning chain for all adapters: HTML→text when the body looks like HTML, quotes, signature, whitespace/NFC/zero-width removal. It's idempotent, so the email adapter's earlier cleaning is harmless. Skipped tickets are logged with a reason and not stored.
- 2026-10-15 (P1.4): Zendesk Fetch uses `GET /api/v2/incremental/tickets/cursor.json?start_time=…&include=users`, then follows `cursor=after_cursor` until `end_of_stream`, with at most 50 pages per poll (the rest comes next poll).
  - `start_time` is clamped to ≤ now−61s (an API rule). A zero `since` looks back 30 days.
  - Only new/open/pending/hold tickets are ingested; solved, closed and deleted are skipped.
  - Ticket fields: `ExternalID`=`ThreadID` is the ticket id; `Body` is the description (first comment); `Sender` is the requester's email from the sideload, else `zendesk-user:<id>`; `ReceivedAt` is `created_at`.
  - **Limitation:** follow-up comments on an existing ticket aren't ingested, because the incremental export is per ticket and the DB key drops repeats.
- 2026-10-15 (P1.4): Zendesk retries:
  - 429 is always retried, honouring `Retry-After`, otherwise with 2^n s backoff (up to 5 retries).
  - 5xx and transport errors are retried only for GET. A reply PUT that fails with 5xx is **not** retried, because the comment may already have posted and a retry could duplicate it. Outbound (P6.3) owns idempotency.
  - Freshdesk (optional in the spec) is not implemented.
- 2026-10-15 (P1.3b): The IMAP poll opens a connection per poll, SELECTs the mailbox read-only, and searches UNSEEN SINCE the day of `since`. It fetches `BODY.PEEK[]`, so **the user's inbox flags are never changed**. Repeat fetches are handled by `since` filtering on the server's INTERNALDATE plus the DB unique key `(source, external_id=Message-ID)`.
  - `ReceivedAt` is INTERNALDATE (trusted), falling back to the `Date` header.
  - Mail from our own `SMTP_FROM` address is skipped, to avoid reply loops. Unparseable messages are logged and skipped, without failing the batch.
- 2026-10-15 (P1.3b): Replies are text/plain UTF-8 quoted-printable, with `In-Reply-To: <ExternalID>` and `References: <thread root> <ExternalID>`. Header values are stripped of CR/LF to prevent injection. Subjects already starting with "Re:" don't get a second one. Hash-based ids (`nomsgid-…`) are never used as In-Reply-To.
- 2026-10-15 (P1.3a): HTML→text and quote/signature stripping live in `internal/preprocess` so every adapter shares them (P1.5 applies them to all sources); the email parser already uses them.
  - Quote stripping cuts at the first reply header: "On … wrote:" (also when it wraps onto two lines), Outlook `From:`+`Sent:/To:/Date:` blocks, underscore rules, or "-----Original Message-----". It also drops every `>` line.
  - Signature stripping cuts at the RFC 3676 `-- ` delimiter or a mobile footer ("Sent from my iPhone", "Get Outlook for iOS").
  - Sign-offs like "Thanks, Priya" are kept. They're too ambiguous to strip safely, and harmless for classification.
  - If stripping would leave nothing, the original text is kept.
  - HTML stripping also drops `<blockquote>`, `.gmail_quote` and Outlook `#appendonsend`/`#divRplyFwdMsg` containers.
- 2026-10-15 (P1.3a): Email parsing uses `emersion/go-message`, which handles RFC 2047 headers, transfer encodings and non-UTF-8 charsets.
  - A message's `ExternalID` is its Message-ID without brackets, falling back to a `nomsgid-` content hash.
  - `ThreadID` is the thread root: References[0], else In-Reply-To, else its own Message-ID.
  - text/plain is preferred over text/html, and attachments are ignored.
- 2026-10-15 (P1.2): CSV `external_id` is the `Id` column if present. Otherwise it's `csv-` plus the first 16 hex characters of sha256 over sender, subject, body and date. That's stable across reorders and re-reads, and two rows with the same text but different dates get distinct ids, so content-hash dedupe (P1.5) still sees them.
  - `ReceivedAt` comes from the `Date` column (RFC 3339 / `YYYY-MM-DD[ HH:MM:SS]`, UTC), falling back to the file's mtime.
  - `Label` isn't part of the Ticket contract. It's exposed via `CSVSource.ReadRows()` for evaluation tooling and not stored.
- 2026-10-15 (P1.2): `datasets/sample_emails.csv` adds an optional `Date` column to SCRIBE's format. It deliberately includes edge cases for later parts:
  - a repeated complaint with a different date (content-hash dedupe);
  - a Portuguese billing email (the language check);
  - `noreply@`, newsletter and automated senders (mute, P4.2);
  - an HTML body;
  - a quoted reply and a signature (stripping).
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
- **Zendesk is only tested against an `httptest` mock**, since no Zendesk account is configured. The incremental export also requires an admin API token.
- **Email adapter not tested against a real mailbox:** IMAP_*/SMTP_* aren't configured. `IMAPClient` is untested beyond compiling. `SMTPMailer` is tested against an in-process fake SMTP server (no TLS, no auth). Gmail needs an app password, plus IMAP enabled.
- **DB latency (`cmd/dbping`, 2026-10-15):** after the move to ap-south-1 (Mumbai):
  - warm-pool `SELECT 1` p50 is 28ms (was 395ms in Sydney);
  - TCP connect is 31ms;
  - a fresh connection is 212ms.
  - The integration suite takes about 9s.
  - Ingestion still does 2 queries per ticket; batching is deferred to P6/P8 volume.
- **Zendesk is first-message-only (accepted limitation).** Follow-up comments on an existing ticket are not ingested. The incremental export is per ticket, and the `(source, external_id)` key drops repeats.
- **Invisible characters in generated Go files:** `\u` escapes written by Claude can land in files as literal BOM/nbsp/zero-width characters, and Go rejects a mid-file BOM. Fix with a small script that rewrites them as `\uXXXX` escapes. Check `go vet` after writing any file containing such escapes.
- **Kaggle token check:** Kaggle answers 404 to the token check for an invalid token, and the SDK reports that as "Authentication required". `python -m data.kaggle_auth` names the likely cause.
- **Integration test cost:** each package pays a one-off setup (create, migrate, drop its schema), which takes a few seconds since the move to Mumbai (it was about 26s in Sydney).
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

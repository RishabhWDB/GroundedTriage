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
| Live Go → ml-service tests (needs `make dev` or `make ml` running; classify needs `models/fast_tier/`) | `go test -tags live -count=1 ./internal/mlclient -run Live` |
| Unit tests (Go + Python; slow/integration skipped) | `make test` |
| One Go test | `go test ./internal/<pkg> -run TestName` |
| One Python test | `cd ml-service && .venv/bin/pytest tests/test_x.py::test_name` |
| Slow (model-loading) Python tests | `cd ml-service && RUN_SLOW=1 .venv/bin/pytest -m slow` |
| Integration tests (needs `TEST_DATABASE_URL`) | `make test-integration` |
| Lint (gofmt check, go vet, ruff check + format check) | `make lint` |
| Auto-format | `make fmt` |
| Migrate (goose, up only, idempotent; SQL in `migrations/`, embedded) | `make migrate` / `make migrate-status` / `make migrate-verify` (read-only: vector ext, tables, HNSW index, ticket counts) |
| One Go integration test | `go test -tags integration -count=1 ./internal/store -run TestTicketInsertAndGet` |
| Seed | `make seed` (not implemented until P8.1) |
| DB latency diagnostic (region, TCP, fresh vs warm round-trips, pool reuse) | `go run ./cmd/dbping` (`-env TEST_DATABASE_URL` for the test DB) |
| Kaggle auth check | `cd ml-service && .venv/bin/python -m data.kaggle_auth` |
| Worker: one poll + classify / loop | `go run ./cmd/worker -once` / `make worker` (loop, `WORKER_POLL_INTERVAL`, default 60s; classifies via `ML_SERVICE_URL`) |
| Build splits + dataset card (downloads raw data on first run) | `make data` |
| Data audit (synthetic diversity, train→test near-dups, subject shortcut) | `make data-audit` → `results/data_audit/report.md` |
| Hard Meeting stress-test CSV (Groq; refuses to overwrite a reviewed file without `--force`) | `make stress-set` → `datasets/stress/meeting_stress.csv` |
| Informal-style paraphrases of 25% of train (Groq; resumable; days on the free tier) | `make style-aug` (`--recheck` drops near-copies) → `datasets/synthetic/style_aug.jsonl`, then `make data` |
| Train the binary spam filter and pick its thresholds on spam_val (~1 min) | `make spam-filter` → `models/spam_filter/`, `results/spam_filter/` |
| Bundle splits for the GPU PC | `make export-splits` → `datasets/export/groundedtriage-splits-<test hash>.tar.gz` |
| Regenerate synthetic meeting emails (Groq; cached, resumable) | `make synthetic-meetings` |
| Install torch + transformers for the deep tier / urgency model | `make setup-models` |
| Install the SCRIBE baseline extra (torch + transformers) | `make setup-baseline` |
| SCRIBE baseline CLI (own process; ~2 GB of weights download on first run) | `make baseline ARGS="classify --split test --limit 10"` / `ARGS="classify --csv ../datasets/sample_emails.csv"` / `ARGS="draft --csv ../datasets/sample_emails.csv --limit 2"` |
| Baseline evaluation on test (P2.3; ~90 min on M1 under swap; resumable) | `make baseline-eval` → `results/baseline/` (smoke: `make baseline ARGS="evaluate --limit 60 --out /tmp/x"`) |
| Score the reviewed stress set into the baseline results (after the CSV review) | `make baseline-stress` |
| Train the fast tier on train, report on val (~8 min on M1) | `make fast-tier` → `models/fast_tier/` (gitignored) + `results/fast_tier/` |
| Score urgency (heuristic, plus model if present) on val | `make urgency-eval` → `results/urgency/` |
| Tune the cascade threshold τ on val (needs `models/deep_tier/`) | `make tune-tau` → `config/cascade.yaml`, `results/cascade/` |
| Classify one ticket (ml-service running) | `curl -s localhost:8001/classify -H 'content-type: application/json' -d '{"ticket_id":"t1","subject":"Refund","body":"Charged twice"}'` |
| Build the OOD set (Twitter pre-labelled for review, Enron spam, team template) | `make ood-set` → `datasets/ood/` (refuses to overwrite a reviewed file) |
| **Required gate before accepting any model:** OOD eval of SCRIBE, fast tier and cascades | `make ood-eval` → `results/ood/` |
| Evaluate the proposed system | not implemented yet (P7) |

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
  - Commit `2c54858`.
- P2.1f Splits, card and export:
  - `make data` builds 70/15/15 splits stratified by category × origin (seed 42), with a train-only 5× cap, a `text` model-input column, `manifest.json` (file + content SHA-256s) and the generated `datasets/README.md` card;
  - `make export-splits` writes a deterministic archive for the PC;
  - `docs/model_artifacts.md` holds the export and artifact contract.
  - Rebuild is verified identical.
  - Commit `d089b39`.
- P2.1g Review fixes + data audit:
  - train cap now 5× the smallest **real** class (Spam 1,076 → 5,380/class): train 13,651 rows; val/test unchanged (same file and content hashes);
  - `train_urgency.parquet` (16,260 uncapped labelled train rows) for the urgency model;
  - `docs/model_artifacts.md`: class-weighted CE for the deep tier, val macro-F1 checkpoint selection for both models;
  - `make data-audit` → `results/data_audit/` (synthetic samples and diversity, train→test near-duplicates, subject-shortcut check).
  - Commit `e710c9f`.
- P2.2 SCRIBE baseline (`ml-service/baseline/`, optional `[baseline]` extra, its own CLI):
  - regex rules for Spam and Meeting Scheduling, combined by noisy-OR and accepted at ≥ 0.95;
  - BART-MNLI zero-shot fallback over the `taxonomy.yaml` descriptions;
  - twitter-roberta sentiment → −1/0/+1 → urgency;
  - ungrounded draft→refine via gpt-oss-120b;
  - shared `app/interfaces.py` (`TicketClassifier`, `ReplyDrafter`, `CategoryPrediction`, `DraftResult`).
  - Slow tests pass end to end on 10 sample emails (classify and draft).
  - Commit `33a51cd`.
- P2.2a Supabase (Mumbai) checks and hallucination example:
  - `cmd/migrate verify` / `make migrate-verify` is a read-only check of the vector extension, the 8 tables, the HNSW index and ticket counts;
  - Mumbai project verified: migrations up to date, 2 worker runs inserted 0, integration suite passes;
  - `results/examples/ungrounded_hallucination_01.md`.
  - Commit `4e2be2c`.
- P2.1h Group-aware re-split (review Q1 = a):
  - `data/neardup.py` groups near-duplicates (cosine > 0.9, connected components over all rows);
  - `stratified_split` keeps every group in exactly one split, stratified by category × origin;
  - the audit checks every split pair with the build's vectorizer: **0 cross-split pairs**;
  - new hashes (test content `c6539fff…2daf3`); export `groundedtriage-splits-c6539fff94a1.tar.gz` is **final**.
  - Commit `760b230`.
- P2.1i Hard Meeting Scheduling stress set (eval only, never trained on):
  - `data/stress_meetings.py` (seed 1337, its own prompt) → `datasets/stress/meeting_stress.csv` (100 rows, committed);
  - every row is `needs_review=true` until Arya hand-checks it; `load_reviewed()` returns None until all rows are reviewed.
  - Commit `72e6728`.
- P2.3 SCRIBE baseline evaluation snapshot on test (3,775 rows):
  - `eval/classification.py` (reporting-rule slices, confusion, ordinal urgency MAE, latency summaries, confusion plot), shared with P7.1;
  - `baseline/evaluate.py` + `make baseline-eval` (resumable) → `results/baseline/{metrics.json, confusion_matrix.png, latency.md, predictions.jsonl}`;
  - macro-F1 0.522 overall / 0.479 real-only and without Meeting Scheduling (see §8). Stress set reported as pending review.
  - Commit `c5a5f7b`.
- P2.3a Phase 2 review follow-ups: `baseline evaluate --stress-only` / `make baseline-stress` scores the reviewed stress set into an existing `metrics.json`; review decisions recorded below.
  - Commit `d38973c`.
- P3.1 Fast tier:
  - `triage/fast_tier.py` (pipeline builder, `FastTier.load` with validation, `predict`), `triage/train_fast_tier.py` (`make fast-tier`) → `models/fast_tier/{model.joblib, model_card.json}` + `results/fast_tier/`;
  - `eval/runinfo.py` (git commit, peak RSS, swap) used by every training/eval run, including the baseline; `eval.classification` gains ECE/reliability and confidence-coverage helpers;
  - val macro-F1 0.825 overall / 0.790 real-only (see §8).
  - Commit `62cf039`.
- P3.2 Deep-tier inference loader (no training): `triage/hf_model.py` (`validate_artifact`, `HFClassifier`: lazy load, MPS fp16 / CPU, batching, truncation to `max_length`, `available` + logged reason) and `triage/deep_tier.py` (`DeepTier`). Tests build a tiny random BERT + word-level tokenizer in a temp dir (nothing downloaded).
  - Commit `65bb485`.
- P3.3 Urgency loader + heuristic fallback (no training): `triage/urgency.py` (`UrgencyModel` on the shared loader, `heuristic_urgency`, `UrgencyPredictor`: model else heuristic, critical rule on top); `triage/eval_urgency.py` (`make urgency-eval` → `results/urgency/`); ordinal metrics on the full low..critical scale.
  - Commit `d28e515`.
- P3.4 Cascade + threshold tuning:
  - `triage/cascade.py` (fast → τ → deep if available, else fast with `low_confidence`; urgency always);
  - `triage/tune_threshold.py` (`make tune-tau`: τ sweep on val, cost curve, writes `config/cascade.yaml`; exits cleanly without the deep tier);
  - `app/models.py` lazy registry; `POST /classify` → `Classification`; `/health` reports artifact availability without loading weights;
  - verified live with only the fast tier present.
  - Commit `83c5f0f`.
- P3.5 Go integration:
  - `store.Classifications` (`Insert`, `Latest`, `Unclassified`: accepted tickets with no classification, oldest first);
  - `mlclient.Classify`;
  - `pipeline.Classifier` (`ClassifyPending`: per-ticket POST /classify, validates the ticket id and category, persists, one `classify.run` audit row, stops after N consecutive failures);
  - the worker runs it after every poll (`Poller.AfterPoll`) and once in `-once` mode.
  - Integration test against a stub ml-service; live Go→ml-service test passes.
  - Commit `906561f`.
- P3.5a Out-of-domain check: `eval/sample_check.py` (`make sample-check`) runs the cascade and SCRIBE on the 22 hand-written sample emails → `results/ood/`. **The fast tier gets 6/22 (13 predicted Spam); SCRIBE gets 19/22.** It learned source style, not only topic (see §5 and §8).
  - Commit `888720b`.
- P3.6 Spam-filter data source and taxonomy split:
  - `data/spamassassin.py` also builds **ham** (`20030228_easy_ham`, `_easy_ham_2`, `_hard_ham`): 4,150 messages → 4,089 rows (10 non-English, 51 duplicates, 316 truncated). No category, `source_dataset=spamassassin_ham`, ids `sah-*`;
  - `config/taxonomy.yaml` `filter_categories: [Spam]`, with `app.taxonomy.topic_categories()` (5) and `filter_categories()`;
  - `summary.md` (gitignored) convention.
  - Commit `23d8fce`.
- P3.7 Rebuild for two tasks:
  - one group-aware assignment over Tobi-Bueck + spam + ham + synthetic meetings (one RNG per label × origin stratum);
  - **topic** splits with 5 classes (`train` 15,910 capped at 8,715 / `val` 3,545 / `test` 3,545 / `train_urgency` 16,258);
  - **spam-filter** splits `spam_train` 20,476 / `spam_val` 4,388 / `spam_test` 4,388 (`is_spam`);
  - the audit covers both tasks (0 cross-split near-duplicate pairs);
  - the export ships only manifest splits;
  - `docs/model_artifacts.md` is 5-class. Export tag **`e2552b04338b`**.
  - Commit `7d07510`.
- P3.8 Style augmentation generator (data in a follow-up commit):
  - `data/style_aug.py` (`make style-aug`, resumable): 25% of each topic train class (3,978 rows), seeded order, gpt-oss-120b informal paraphrases, 8 per call; near-copy check (body cosine > 0.9 rejected); `--recheck`; JSON-mode failures retried;
  - `data.build.style_aug_split` → `train_style_aug.parquet` (rows whose source is in train, labels checked).
  - Commit `707b2e7`.
- P3.9 Binary spam filter:
  - `triage/spam_filter.py` (`SpamFilter.load`, spam/review/ham decision) and `triage/train_spam_filter.py` (`make spam-filter`, ~1 min) → `models/spam_filter/` + `results/spam_filter/`;
  - two stored threshold rules (`support`, the default, and `support_and_ham`), selectable with `SPAM_RULE`;
  - the meeting/style parser now rejects non-object email entries (a crash in the style-aug run).
  - Commit `c2891ee`.
- P3.10 Spam filter in the cascade:
  - `Cascade(..., spam=SpamFilter)`: a muted ticket → `category=Spam, tier=spam_filter` (topic models skipped); the review band → topic as usual plus `possible_spam=true`;
  - the fast and deep tiers require the 5 topic classes (6-class artifacts are rejected);
  - contract: `tier` adds `spam_filter`; `spam_score`, `possible_spam` (Python + Go);
  - migration **00005** (applied to Mumbai); `store.Classifications` persists them.
  - Commit `aa94af1`.
- P3.11 OOD evaluation set and gate:
  - `data/ood.py` (`make ood-set`) → `datasets/ood/`: `twitter.csv` (149 rows, qwen pre-labels, `needs_review=true`), `enron_spam.csv` (20, CC0 copy of Enron1), the `team_written.csv` template and a README;
  - `eval/ood_eval.py` (`make ood-eval`) scores SCRIBE, the fast tier alone and the cascade under each spam rule on reviewed rows plus the 22 samples. It **replaces `sample-check`** (removed).
  - Commit: backfilled in the next commit.

### In progress
- Phase 3 rework after the review (2026-10-17): a two-stage spam filter plus a 5-class topic classifier, a style-augmentation file, and the OOD evaluation set. Run P3.6–P3.12 through, then print an updated Phase 3 review.

### To do
- P3.8a Commit the generated `datasets/synthetic/style_aug.jsonl` and rebuild/re-export (when the Groq run finishes; several days on the free tier)
- P3.12 Re-run the fast tier, urgency eval and OOD eval; before/after; updated Phase 3 review
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
| `models/fast_tier/` | present (`fast-20261016012651-6f7999df`, `make fast-tier`) | TF-IDF word 1–2 + char_wb 3–5, LR C=16 balanced, isotonic-calibrated | in repo, 2026-10-16 | macro-F1 0.825, acc 0.788, ECE 0.083 |

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
  - **The final rows are committed** (`datasets/synthetic/`, 388 KB). They're our own generated data, not a downloaded dataset, and a re-run at temperature 0.9 wouldn't reproduce them byte-for-byte.
  - Quality check: 400 unique bodies; 0 bracket placeholders; 0 "finds you well"; 97% mention a meeting word; 323 unique subjects.
  - **Caveat for the report:** the emails are shorter than Tobi-Bueck's (median 260 vs 374 chars) and their subjects are keyword-heavy ("confirmation", "reschedule", "review"). The class is likely easier than the real ones, so report per-origin metrics.
  - Urgency is NA for these rows.
- 2026-10-15 (P2.1f): Split build results:
  - 17 cross-source duplicate bodies removed.
  - Split sizes after the train cap: train 6,956 / val 3,774 / test 3,774.
  - **The train cap is 5 × 280 (synthetic meeting rows in train) = 1,400 per class.** It removed 10,660 real rows from train: Technical Issue 7,945, General Inquiry 1,469, Billing 903, Feature Request 343. Spam (1,076) and Meeting (280) were below the cap.
  - val/test are not resampled (Technical Issue is 2,002 of 3,774 in test).
  - The test split's content sha256 is `db976b05…009b`, which is also the export tag. A rebuild gives identical file and content hashes.
  - The `text` column (`app/textinput.model_input`: subject + blank line + body) is the model input for every model; serving must use the same function.
- 2026-10-15 (P2.1g, review Q1 = (b)+(c)): **The train cap is 5× the smallest *real* class**, so synthetic rows never set it. Spam has 1,076 train rows, so the cap is 5,380, and only Technical Issue is cut (9,345 → 5,380, 3,965 removed). Train is 13,651 rows. This supersedes the P2.1f cap of 1,400.
  - `train_urgency.parquet` is every uncapped train row with an urgency label (16,260: high 6,318 / medium 6,675 / low 3,267). It comes from the same pre-cap pool as `train`, so it shares no rows with val/test. The export picks it up automatically (it's in the manifest).
  - The remaining imbalance is handled by **class weighting, not more downsampling**. The fast tier uses `class_weight="balanced"`. The deep tier must use class-weighted cross-entropy (`w_c = N/(K·n_c)` from train frequencies). Both select checkpoints on **val macro-F1**. All of this is in `docs/model_artifacts.md`.
  - The rebuild left val and test byte-identical (file and content sha256). Test content is still `db976b05…009b`, so the export tag is unchanged.
- 2026-10-15 (P2.1g): Data audit (`make data-audit`, `results/data_audit/`). Near-duplicate = cosine > 0.9 on word-unigram TF-IDF (sublinear tf). Unigrams ignore word order, so it's an upper bound. Findings:
  - Synthetic diversity: 0 exact duplicate bodies, 0 near-duplicate pairs within the 400 rows, 316 unique subjects (172 exact duplicate-subject pairs). The top subject opening is "request to" (25 rows).
  - **Leakage:** 436 of 3,774 test rows (11.6%) have a near-duplicate in the training pool (capped train ∪ train_urgency). That's Spam 19.6%, Billing 14.0%, Feature Request 14.2%, Technical Issue 10.6%, General Inquiry 9.3%, synthetic Meeting 0%. The nearest neighbour is always the same category. These are genuine paraphrases in Tobi-Bueck (LLM-generated variants of one ticket). With 1–2-gram TF-IDF, 93 rows are above 0.9 and 373 above 0.8. The splits are unchanged pending Arya's decision.
  - **Shortcut:** TF-IDF+LR on val gives Meeting Scheduling F1 0.99 from the subject alone, and 1.00 from the body alone or the full text. The class is trivially separable from either field, so neutralising subjects wouldn't make it harder. 202 real val tickets mention a meeting word, and none were predicted Meeting. Spam is also near 1.0 (subject 0.95). Model macro-F1 on val: subject-only 0.72, body-only 0.82, full text 0.82.
- 2026-10-15 (P2.1h, review Q1 = (a)): **Group-aware split.**
  - All 25,164 rows (after exact dedupe) are linked by near-duplicate pairs (word-unigram TF-IDF fitted on all rows, cosine > 0.9) into connected components. That gives 1,706 groups of 2+ rows covering 3,989 rows; the largest has 14. 2 groups mix categories; each group is stratified by its majority category (ties: taxonomy order), and no group mixes origins.
  - Per category × origin stratum: shuffle the groups (seed 42), put the largest first, and assign each to the split furthest below its target row count. Singletons hit round(n·0.15) exactly. The split is input-order independent (groups are keyed by their smallest id).
  - The audit refits the same vectorizer on the same corpus (checked against the manifest) and finds **0 near-duplicate pairs for train-pool~val, train-pool~test and val~test**. Before the re-split, 436 test rows (11.6%) leaked.
  - New sizes: train 13,651 (cap still 5,380 = 5 × 1,076 Spam; Technical lost 3,963) / val 3,775 / test 3,775 / train_urgency 16,258.
  - **This supersedes the P2.1f/P2.1g hashes.** Test content sha256 is now `c6539fff94a18757de5017c2ad60a0ff29baf9c01473b80d85ed7d4328e2daf3`, and the export is `datasets/export/groundedtriage-splits-c6539fff94a1.tar.gz`. The archive also bundles `docs/model_artifacts.md`, the card and `taxonomy.yaml`, so its own sha256 changes whenever those docs change. **Identify the data by the test content hash in the name, not the archive hash.** The rebuild is byte-identical. The old `…db976b0585b2.tar.gz` archive in `datasets/export/` is stale and must not be used.
  - Effect: the audit's diagnostic TF-IDF+LR val macro-F1 dropped from 0.825 to 0.802 (accuracy 0.760 to 0.735), so leaked near-duplicates had been inflating scores.
- 2026-10-15 (P2.1i, review Q2 = b): **Hard stress-test set for Meeting Scheduling.**
  - 100 rows from gpt-oss-120b (temperature 0.9, JSON mode), seed 1337 and a separate prompt (the training generator uses 42). 21 calls, 24,729 tokens; 1 batch retried.
  - 50 `embedded_meeting` rows: a technical (25) or billing (25) problem whose main ask is a meeting (intended label Meeting Scheduling).
  - 50 `call_mention` rows: Technical 15 / Billing 12 / Feature 12 / General 11 that mention calls, demos, Zoom or calendars but don't ask for a meeting.
  - The CSV is committed at `datasets/stress/meeting_stress.csv` with columns `intended_category`, `category` (editable), `needs_review` (all `true`) and `reviewer_notes`. **It's evaluation-only, never trained on, and not in the PC export.** No number on it is reported until Arya has reviewed every row; `load_reviewed()` enforces this.
  - Checks: 100 unique bodies (median 459 chars); 0 near-duplicates with the 25,164-row corpus and 0 within the set.
  - Also tracked: **real** test tickets that mention a meeting word but aren't Meeting Scheduling. That's 196 rows (Technical 83, Spam 50, Feature 27, Billing 19, General 17), selected by the audit's `MEETING_WORDS` regex. P2.3 and P7 report this slice. The regex also matches e.g. "call 1-888…" in spam.
- 2026-10-17 (P3.11): OOD set:
  - **Twitter:** Kaggle `thoughtvector/customer-support-on-twitter`, licence **CC BY-NC-SA 4.0** (checked via the Kaggle API), so the 149 rows are committed with attribution (`datasets/ood/README.md`). Inbound thread-opening tweets; `@mentions` stripped (the first is kept as `brand`); English (langdetect); ≥ 60 chars; deduplicated. Seeded (4242) 600-candidate pool.
    - Pre-labels from the **judge model (qwen3.8-27b) with no reasoning**: a separate Groq rate-limit bucket from the gpt-oss style-aug run; 30 calls, 36k tokens.
    - Candidate labels: General 265, Technical 194, Billing 59, None 58, Feature 18, Spam 5, Meeting 1.
    - Quotas: Technical 45 / Billing 40 / General 40 / Feature 20 / Meeting 10 / Spam 10. **Feature, Meeting and Spam are rare on Twitter**, giving 18 / 1 / 5, so 149 rows in total. The team-written emails are where meeting requests come from.
  - **Enron spam:** the original AUEB server's TLS chain fails verification (I didn't bypass it), so it comes from the CC0 Kaggle copy `venky73/spam-mails-dataset` (Enron1, 1,499 spam). 20 seeded English rows, committed. The text is the corpus's lower-cased tokenised form, a style the SpamAssassin-trained filter has never seen.
  - **Scoring:** only `needs_review=false` Twitter rows are scored (plus Enron, team rows and the 22 samples), the same rule as the stress set. `make ood-eval` is the **required gate** before accepting any model or threshold rule, and is reported next to test in P7.
  - The P3.5a `results/ood/sample_emails.{md,json}` stay as the before snapshot (6-class fast tier: 6/22).
- 2026-10-17 (P3.10): Spam filter in the cascade:
  - The spam filter runs first on every ticket. **≥ mute threshold → `Spam`, `tier=spam_filter`**, confidence = filter score, and the topic models don't run. Review band → the topic cascade runs normally and the result carries **`possible_spam=true`** (P4 routes it to a "possible spam — review" queue instead of muting). Below the review threshold → topic only.
  - Without `models/spam_filter/`, nothing is ever Spam, `spam_score` is null and `model_versions.spam_filter="absent"`. The filter version string includes the rule (`spam-…:support`).
  - The fast tier and deep tier now **require the 5 topic classes**. The P3.1 6-class `models/fast_tier/` is rejected, so `/classify` returns 503 until `make fast-tier` is re-run (P3.12). A 6-class deep-tier label_map is rejected.
  - Contract (Python `app/schemas.py`, Go `internal/model`): `Tier` adds `spam_filter`; new `spam_score` (0–1, nullable) and `possible_spam` (bool). `Classification.category` can still be Spam.
  - **Migration 00005** widens the `classifications.tier` check and adds `spam_score real` (0–1 check, nullable) and `possible_spam boolean not null default false`. It's additive, so I applied it to the Mumbai DB (`make migrate`: 1 applied; `migrate verify` OK). No existing rows changed.
- 2026-10-17 (P3.9): Spam filter:
  - Same features as the fast tier (word 1–2 + char_wb 3–5 TF-IDF), binary LR `class_weight="balanced"`, no calibration (only the ranking and thresholds matter). C from {1, 4, 16} by **val average precision**: C=16.
  - Fitted on `spam_train`: 1,076 spam vs 2,862 ham + 16,538 support tickets.
  - **Thresholds (rule `support`, your spec, the default):** mute = the lowest score with **support-ticket FPR ≤ 1%** on spam_val; review = the lowest with support FPR ≤ 5%. Support = Tobi-Bueck + synthetic meetings; ham is reported separately. Scores are float64 (float32 made the "one ulp above" threshold miss by one ticket: 36/3,545).
  - **Finding:** support tickets (LLM-written) almost all score near 0, so the 1% rule sets mute at **0.0195**. At that point the filter mutes **20% of ham** (human-written legitimate mail) on val (19.7%) and test (21.0%). Real customers write like humans, so this rule likely over-mutes them.
  - **Alternative rule `support_and_ham`** (stored in the card, `SPAM_RULE=support_and_ham`): the same FPR limits on support **and** ham. Mute ≥ 0.4365, review ≥ 0.0941. On test: spam recall 0.978 at mute (0.996 with review), support FPR 0.0, ham FPR 0.021. **Not adopted yet:** your rule stays the default until the OOD eval (P3.11/P3.12) measures both on human-written customer mail; the Phase 3 review asks you to choose.
  - Spam test numbers are reported for this separate component (its own splits); the thresholds and C were chosen on spam_val only.
- 2026-10-17 (P3.8): Style augmentation:
  - Selection: 25% of each class of the capped topic `train` (`select`, seed **2026**) = **3,978 rows**, shuffled so any prefix is a stratified sample.
  - Prompt fixed in the module: keep the same request, product and facts; replace `<name>`/`<acc_num>`-style tokens with invented values; no stock phrases. Each row gets a seeded style spec: register (6), length (2), greeting (5), typos (3), subject style (3).
  - gpt-oss-120b, temperature 0.9, JSON mode, 8 rows per call; responses cached by request sha256.
  - Rows are rejected when the body is too short, has a placeholder, or has word-unigram cosine > 0.9 to the source **body**. The first run compared against subject+body; `--recheck` re-applied the body rule and removed 2 of 172.
  - Output rows: `id = aug-<source id>`, `aug_of`, the source's category and urgency, `origin=augmented`, the style spec. `data.build` keeps the rows whose source is in the current capped train and checks labels match.
  - **Groq free-tier limits on this key:** 1,000 requests/day and 8,000 tokens/minute (from response headers), plus a daily token cap (about 200k/day for gpt-oss-120b on the free tier, my recollection, not confirmed). The pilot cost about 450 tokens per accepted row, so all 3,978 rows ≈ 1.8M tokens ≈ **~9 days on the free tier** (well under $1 on the paid dev tier). The run is resumable and is restarted until done; the PC's without-augmentation run doesn't need it.
  - Groq JSON mode sometimes returns HTTP 400 `json_validate_failed`; that's now a bad attempt (retried) instead of stopping the run.
- 2026-10-17 (P3.7): **Two-task rebuild. This supersedes the P2.1h split hashes and export.**
  - Corpus: Tobi-Bueck 23,228 + SpamAssassin spam 1,553 + ham 4,089 + synthetic meetings 400, minus 18 cross-source duplicate bodies = 29,252 rows. Near-duplicate groups: 1,862 groups covering 4,328 rows (largest 14).
  - One group-aware 70/15/15 assignment, stratified by **label × origin** (label = category, or "Ham"). Both tasks are cut from it, so a row in one task's train is never in the other task's val/test. **Each stratum now has its own RNG** (seeded by seed + a hash of the stratum), so adding a stratum never reshuffles the others. This change, plus ham joining the near-duplicate corpus, **reshuffled the test split**: only 641 of the 3,545 new topic test rows were in the old test.
  - **Topic task:** 5 classes. The cap is 5 × the smallest real topic class (Feature Request, 1,743) = **8,715**, so only Technical Issue is cut (628 rows). Train is 15,910: Technical 8,715 / General 2,869 / Billing 2,303 / Feature 1,743 / Meeting 280. Val and test have 3,545 each. `train_urgency` is unchanged in size (16,258).
  - **Spam task:** every row of each split, with topic tickets **uncapped**. `spam_train` = 1,076 spam vs 2,862 ham + 16,258 Tobi-Bueck + 280 meeting. `spam_val` and `spam_test` = 230 spam vs 613 ham + 3,485 + 60 each.
  - Hashes: test content `e2552b04338b4d5872aab17662d750294ce894602f3de0752b90c5049e3d0e81` (export `datasets/export/groundedtriage-splits-e2552b04338b.tar.gz`); spam_test `e6997f05…3272`. **The `c6539fff94a1` and `db976b0585b2` archives are stale (6-class).** The archive is regenerated after P3.8 adds `train_style_aug`; the tag stays the same.
  - **The P2.3 SCRIBE baseline numbers are on the superseded test split** (c6539fff). They stay in `results/baseline/` as a historical snapshot, and P7.1 re-runs the baseline on the new test, under the same conditions as the other systems.
  - `models/fast_tier/` (6-class, P3.1) is stale until P3.12 retrains it; the loaders switch to 5 topic classes in P3.10.
  - The audit's diagnostic TF-IDF+LR on the 5-class topic val: macro-F1 0.777, accuracy 0.728 (the old 0.802 was 6-class with the easy Spam class).
- 2026-10-17 (Phase 3 review): **The style shortcut is a first-class finding for the report.** Decisions:
  - **Q1 = (a), modified: two stages.**
    - **Spam filter (binary):** positives = SpamAssassin spam. Negatives = SpamAssassin ham **plus** support tickets (Tobi-Bueck + synthetic meetings), so writing style or source isn't predictive. Group-aware split with its own train/val/test, reported separately.
    - **Operating point:** chosen on val for **FPR ≤ 1% on support tickets**, because a muted real customer is the costliest error. Scores between a lower bound and that threshold go to a **"possible spam — review"** queue instead of mute. Report FPR and recall at the chosen point.
    - **Topic classifier (fast + deep tiers): 5 classes, no Spam.** The contract is unchanged: `Classification.category` can still be Spam, set by the filter with `tier="spam_filter"`.
    - **Rebuild and re-export** with a new manifest/hash and the 5-class label_map documented in `docs/model_artifacts.md`.
    - Also export an **optional `train_style_aug`**: ~25% of train tickets paraphrased by gpt-oss-120b into informal, human customer-email style (label-preserving, `origin=augmented`, train only, fixed seed and prompt committed). Arya trains with and without it as an ablation.
  - **Q2 = yes: a human-written OOD evaluation set** (~200 rows, never trained on):
    - ~150 inbound customer messages from Customer Support on Twitter (Kaggle `thoughtvector/customer-support-on-twitter`; check the licence), mapped to the 5 topic classes + Spam, LLM pre-labelled with `needs_review=true`; Arya verifies. Rows that fit no class are dropped.
    - A template CSV for ~40 emails written by Arya's team (support-style, including informal and meeting-request ones).
    - ~20 spam emails from a corpus other than SpamAssassin (e.g. Enron-spam), to test whether the filter generalises.
    - **`make ood-eval` replaces `make sample-check`** (it includes the 22 sample emails) and scores every system (SCRIBE, fast tier, cascade, later the deep tier). It's a **required gate before any model is accepted**, and OOD is reported alongside test as a headline in P7.
  - **Q3:** keep CalibratedClassifierCV for now; revisit prefit-on-half-of-val after the 5-class rebuild.
  - After the rebuild, re-run the fast tier, urgency eval and OOD check, and report before/after (especially the fast tier on the 22 samples).
- 2026-10-16 (P3.5a): **Finding: source-style shortcut for Spam.**
  - On the 22 hand-written sample emails, the fast tier is right on 6 and predicts Spam for 13, often at high confidence: "API returning 500 errors" → Spam 0.96; "Refund for annual plan downgrade" → Spam 0.95. SCRIBE gets 19/22 with 1 Spam.
  - The fast tier's strongest Spam features are common informal words (`all`, `no`, `will`, `get`, `your`, `here`, `from`), not only spam cues (`click here`, `free`, `http`).
  - Cause: every real class has exactly one source. Spam is SpamAssassin, the only *human-written* text. The other four real classes are Tobi-Bueck's formal LLM-written tickets. So "informal human English" ≈ Spam. Length isn't the cue (Spam median 1,127 chars against ~400).
  - val/test can't show this, because there the source and the class always coincide. The val macro-F1 (0.825) overstates real-world quality.
  - The sample labels are ours, and several rows are off-domain (Personal/Work remapped to General Inquiry), so this is a sanity check, not a metric. But the Technical and Billing misses are unambiguous.
  - **This will likely affect the DeBERTa deep tier too** (same splits). It needs Arya's decision before PC training; see the Phase 3 review for options. Nothing was changed in the data or models.
- 2026-10-16 (P3.5): Go classification step:
  - The worker classifies after each poll, inside the same timeout-bounded context. **Only `ingest_status='accepted'` tickets with no `classifications` row are sent**, so skipped tickets never reach the ml-service.
  - Each run handles up to 100 tickets (`BatchSize`), oldest first. One request per ticket (the ml-service batches internally only within a request; volume is low).
  - A failed ticket (transport error, non-2xx after the client's retries, a mismatched `ticket_id`, or an unknown category) is logged and left unclassified, so the next run retries it. **3 consecutive failures stop the run** with `ErrMLUnavailable`, so a down ml-service isn't hammered ticket by ticket.
  - One `classify.run` audit row per run: counts, by-category and by-tier tallies, up to 100 ticket ids, last error. A 503 (no fast tier) is retried by the client like any 5xx.
  - `worker -once` exits 1 if classification fails.
  - **Not run against DATABASE_URL:** that would add classification rows for the 20 Mumbai demo tickets, and the review said to leave the demo rows as they are. The integration test covers it on TEST_DATABASE_URL.
- 2026-10-16 (P3.4): Cascade:
  - Fast tier on every ticket. Calibrated confidence **< τ** goes to the deep tier when it's available; the deep tier sees only those tickets, as one batch. Otherwise the fast prediction is kept with `low_confidence=true`, which also happens when the deep tier fails at load. Urgency runs on every ticket.
  - `model_versions` = `{fast_tier, tau, deep_tier (if available), urgency (version or "heuristic")}`. `models_loaded` = deep/urgency artifact availability.
  - **τ = 0.7 is an untuned default** (`config/cascade.yaml`, `tuned: false`) until the deep tier exists. Without a deep tier, τ only sets the `low_confidence` flag: at 0.7, half of val is flagged, and the other half is 93% accurate.
  - **τ rule** (`make tune-tau`): sweep τ 0.00–1.00 in 0.01 steps on val. Pick the smallest deep-tier share whose macro-F1 (over the classes present in gold) is within 0.005 of the sweep's best; ties go to the smaller τ. It writes τ, the rule, the val numbers and both model versions to `config/cascade.yaml`, and the sweep and cost curve to `results/cascade/`. Without a valid `models/deep_tier/` it prints why, changes nothing and exits 0 (no fake curve).
  - Serving: `app/models.Registry` is built once per process. The fast tier loads on the first `/classify` (0.6 s); deep/urgency validate on first access and load weights on first use. `/classify` returns 503 with "run `make fast-tier`" when there's no fast tier. τ is re-read from YAML on each request, so a re-tune needs no restart. `POST /classify` takes `{ticket_id, subject, body}` (`ClassifyRequest`).
  - `Cascade.classify` also implements `app.interfaces.TicketClassifier` (stage fast|deep) for the P7 harness.
- 2026-10-16 (P3.3): Urgency:
  - The model (`models/urgency/`, task `urgency`) uses the same loader and contract. label_map must be exactly `[low, medium, high]` in id order, optionally plus `critical`.
  - **The heuristic** (documented table in `triage/urgency.py`): six regex groups, each counted once per ticket. Urgency keywords +3 (alone enough for high); deadline/time +1; escalation +1; negative phrasing +1; business impact +2; low-urgency cues −2. Score ≤ −1 → low, 0..2 → medium, ≥ 3 → high.
  - **critical = business impact AND (urgency keywords OR escalation).** This rule applies on top of the model too, since the data has no critical rows (P2.1 D4); a critical override is marked `urgency_source="heuristic"`.
  - Heuristic confidence is a fixed, uncalibrated 0.4 + 0.1 per fired group (max 0.8).
  - The cue lists were written from general support-desk vocabulary **before** the val run and were not tuned on val afterwards. During unit testing I raised the urgency-keyword weight from +2 to +3, so that "URGENT … ASAP" alone means high; that was a design fix from a toy example, not tuning.
  - If the model fails to load at first use, `UrgencyPredictor` falls back to the heuristic for that and later calls.
  - Ordinal metrics: `eval.classification.urgency_metrics` gives macro-F1 over low/medium/high plus MAE over positions on low < medium < high < critical (critical vs high = 1 step) and the share off by ≥ 2.
- 2026-10-16 (P3.2): Deep-tier loader (`triage/hf_model.py`, shared with urgency):
  - **Validation at construction, weights at first use.** Construction only reads files: required `config.json`, `label_map.json`, `model_card.json`, `model.safetensors` and `tokenizer.json` or `tokenizer_config.json`. label_map ids must be exactly 0..n-1 with no duplicate labels, and the labels must be exactly the 6 taxonomy names (any id order; predictions follow label_map ids). model_card needs the right `task`, a positive integer `max_length` and `val_metrics.macro_f1`. `config.json`'s `num_labels` must match label_map.
  - Any failure → `available=False` with a logged `reason`; nothing raises to the service. A failure at weight load (corrupt safetensors, wrong output size) also flips `available` to False, and callers fall back.
  - Inference: `AutoTokenizer` + `AutoModelForSequenceClassification` from the directory; **fp16 on MPS** (`MODEL_FP16`, default true; CPU stays fp32); `torch.inference_mode`; batches of 16; truncation to the card's `max_length`; softmax computed in fp32. One model per role stays resident after the first call.
  - torch/transformers are a `[models]` extra (`make setup-models`). Without them the loader reports unavailable instead of failing.
  - `/health` reports artifact availability (file checks only; it never loads weights).
- 2026-10-16 (P3.1): Fast tier design:
  - Features: word 1–2-gram TF-IDF (min_df 2, ≤200k) ∪ char_wb 3–5-gram TF-IDF (min_df 3, ≤300k), sublinear tf, float32. LR `class_weight="balanced"`, lbfgs.
  - **C** is picked from {1, 4, 16, 64} by *uncalibrated* val macro-F1 (cheap; C=16 won at 0.8296, C=64 at 0.8287, so the optimum is inside the grid). The first run used {1, 4, 16}; C=16 sat on the edge, so 64 was added.
  - **Calibration:** `CalibratedClassifierCV(cv=5, ensemble=False)` on train, so the artifact is one model with one vocabulary (4.2 MB). The method (sigmoid or isotonic) is picked by **val ECE** (top-class confidence, 15 bins). Isotonic won (0.0831 against sigmoid's 0.0892).
  - **Finding: both calibrated models are worse calibrated on val than the plain LR (ECE 0.038), and they are *under*-confident.** For example, isotonic tickets at confidence ≥ 0.7 are 93% correct. Likely cause: the calibrators are fitted on cross-validated *train* predictions, and train is class-capped (Technical 39% of train vs 53% of val) and fitted with balanced weights, so the learned priors don't match val. The spec asks for CalibratedClassifierCV, so it's kept. τ is chosen on val by the coverage rule, so a monotone mis-calibration shifts τ but doesn't break the cascade. Alternative for the Phase 3 review: calibrate on half of val (`cv="prefit"`) and tune τ on the other half.
  - Selection and every reported number use val only; **test is untouched until P7.1**.
  - The artifact is a joblib pickle: only load models this repo trained. `FastTier.load` returns None (logged) on missing files, unreadable pickles or classes that don't match the taxonomy, and warns on a scikit-learn version mismatch. scikit-learn and numpy moved to core deps (the ml-service serves the fast tier).
  - The version stamp is `fast-<UTC yyyymmddHHMMSS>-<train content sha[:8]>`.
- 2026-10-16 (Phase 2 review):
  1. **The SCRIBE baseline is reported as is** (faithful, untuned). P7.1 also adds **"SCRIBE (val-tuned descriptions)"**: tune ONLY the zero-shot label descriptions and hypothesis template on val (a small, documented search, 3–5 candidate wordings per class), freeze, evaluate once on test, and label it a *strengthened baseline*.
     - For the report: SCRIBE's paper reports zero-shot BART beating fine-tuned BERT. On these real support tickets, zero-shot collapses into General Inquiry. Evidence (P2.3, test): 1,818 of 3,712 zero-shot predictions are General Inquiry (41% of Technical, 52% of Billing, 43% of Feature, 53% of Spam rows), giving General Inquiry precision 0.25 against recall 0.73, Billing recall 0.32 and Spam recall 0.21. Even the val-only TF-IDF+LR diagnostic (macro-F1 0.802 on val) far exceeds the zero-shot baseline (0.522 on test).
  2. `--stress-only` exists (`make baseline-stress`); run it after Arya finishes reviewing the stress CSV.
  3. **The P2.3 latencies were taken under heavy swap and must not be used in any comparison.** P7.1 re-measures latency for all systems in one session under the same conditions (other apps closed, models warmed, RSS and swap recorded) and reports those instead.
  4. **Demo data:** leave the Mumbai rows as they are. P8.1 plans a `make demo-reset` that clears only demo-tagged rows (or uses a separate demo schema) and asks for confirmation. Never a blanket truncate of DATABASE_URL.
  5. **Memory (from P3 on, assume 8 GB):** lazy-load models in the ml-service; at most one transformer resident per role; fp16 on MPS for inference; record RSS + swap in every eval run; never load the baseline models in the ml-service process.
- 2026-10-16 (P2.3): Baseline evaluation design:
  - Run on the full test split with batch size 4 on MPS (fastest in smoke tests: batch 1 = 0.72, 4 = 0.58, 16 = 0.77 s/ticket). Models are warmed up before timing; load time is reported separately.
  - Slices (`eval.classification.sliced_metrics`): overall (6 labels), `origin_real`/`origin_synthetic` (labels present in that slice's gold), `without_meeting_scheduling` (gold Meeting rows dropped, 5 labels; a non-Meeting row predicted Meeting is an error), and `real_meeting_mentions` (real, non-Meeting rows matching `MEETING_WORDS`). Every Meeting row in test is synthetic, so origin_real and without_meeting_scheduling are the same rows and the same numbers.
  - Latency is reported two ways: batched (batch time / batch size) and single-ticket (100 seeded random test tickets, one at a time). Swap in use is recorded with the run, because 8 GB RAM makes tail latency swap-bound (11.1 GB swap in use at the end; max single-ticket 73 s, batched max 97 s per ticket).
  - SCRIBE urgency = sentiment → high/medium/low, scored on the 3,485 test rows with an urgency label (Spam and synthetic rows have none).
  - The stress set is gated by `load_reviewed()`; `metrics.json` says "pending" until every row is reviewed. Re-run `make baseline-eval` after the review to fill it in (the main predictions take ~90 min; a stress-only rerun can be added if needed).
  - `predictions.jsonl` is committed (790 KB) so P7.1 can compare per ticket without re-running the baseline.
- 2026-10-15 (P2.2): SCRIBE baseline design:
  - **Rules:** each regex cue has a weight. A category's confidence is the noisy-OR `1 − Π(1 − w)` of its matched cues, and a decision is accepted only at ≥ 0.95 (SCRIBE's fixed cutoff). One strong cue (pharma/casino, advance-fee, weight 0.95) is enough; weak cues (Zoom, "click here") must co-occur. The rules cover only Spam and Meeting Scheduling, per the taxonomy decision. The cue lists are generic vocabulary, written before any evaluation run and **not tuned on val/test**. The noisy-OR scoring is our interpretation of "regex rules with a fixed ~95% cutoff".
  - **Zero-shot:** `facebook/bart-large-mnli`, hypothesis `"This email is {}."` over the six `taxonomy.yaml` descriptions (single-label softmax). Input is `model_input(subject, body)`, the same text the other models see. Long inputs are truncated by the pipeline.
  - **Sentiment:** `cardiffnlp/twitter-roberta-base-sentiment-latest` → −1/0/+1 → urgency high/medium/low (SCRIBE prioritises by sentiment). This gives the baseline urgency predictions for P7, compared on low/medium/high.
  - **Draft:** two gpt-oss-120b calls (draft, then refine), temperature 0, no retrieval. The prompts are in `baseline/scribe.py`.
  - The baseline sits behind `app/interfaces.py` (`TicketClassifier.classify(list[TicketText]) → list[CategoryPrediction]`, `ReplyDrafter.draft → DraftResult`). The P3.4 cascade and P5.3 generator implement the same protocols. These are evaluation-level types; the persisted `Classification` keeps tier `fast|deep`.
  - The extra is `torch>=2.4` + `transformers>=4.44` (installed: torch 2.14.1, transformers 5.19.0, MPS available). It's not part of `make setup`; `make setup-baseline` installs it.
  - First observations (sample CSV, 22 rows, *not an evaluation*): 19/22 correct, 1 rule-handled. Zero-shot takes about 0.8 s/ticket on MPS, including model load. An ungrounded draft for the double-charge email claimed "I've processed a refund" and left `[Your Name]` placeholders. That's the failure mode grounding targets.
- 2026-10-15 (P2.1g, review): **Reporting rule.** Macro-F1 is the headline classification metric. Accuracy is secondary, because Technical Issue is 53% of test. Every classification result is also reported with and without synthetic rows (overall / real only), plus per origin.
- 2026-10-15 (P2.1g): scikit-learn is in the `[data]` extra for the audit. P3.1 moves it to the core deps when the fast tier serves from the ml-service.
- 2026-10-15 (P2.1 D4): urgency is trained and evaluated on low/medium/high only; "critical" comes only from documented heuristic/escalation rules (P3.3). Neither dataset has English critical rows.
- 2026-10-15 (P2.2a, review): Mumbai project verification:
  - Both DATABASE_URL and TEST_DATABASE_URL use host `aws-0-ap-south-1.pooler.supabase.com:5432` (Session pooler).
  - `make migrate` applied 0 migrations, so 00001–00004 were already applied. `migrate verify` found vector 0.8.2 in `extensions`, all 8 tables (incl. `ingest_checkpoints`) and `kb_chunks_embedding_hnsw_idx` using `hnsw`.
  - `cmd/dbping` (n=20): warm-pool SELECT 1 p50 30ms / p95 31ms, against ~395ms p50 in Sydney (about 13× faster). TCP connect p50 32ms; fresh connection p50 211ms.
  - `worker -once` twice: both runs inserted 0 (1 fetched, a content duplicate). The project was **already seeded** by the earlier Mumbai re-seed: 20 `csv/accepted` + 1 `csv/skipped/non_english` (the Portuguese ticket), and the csv checkpoint is 2026-10-09T06:00Z. A "first run inserts 21" demo would need deleting rows in DATABASE_URL, which wasn't done (destructive; Arya's call).
  - `make test-integration` passes in about 15s.
  - The old Sydney project was not touched; Arya deletes it.
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
- **Status updates and phase reviews** are also written to `summary.md` in the repo root, rewritten each time with only the latest updates (gitignored), so Arya can copy them in one go. Still print them in the chat too.
- **Ingestion:** new adapters implement `ingest.Source` and register a `Factory` under their type. Use `ingest.FakeSource` in tests that need a source. Adapter output files go under `datasets/out/`, which is gitignored.
- **Layout:** follows BUILDPROMPT §2:
  - Go entrypoints in `cmd/`, packages in `internal/`;
  - Python in `ml-service/{app,triage,rag,baseline,data,eval,tests}`;
  - SQL in `migrations/`, YAML in `config/`;
  - results in `results/` (committed);
  - `models/`, `datasets/raw|processed|export` gitignored.
- **Shared contracts** (Ticket, Classification, Route, RetrievedChunk, ReplyDraft) are defined once per language: Go structs in `internal/` and Pydantic in `ml-service/app/schemas.py`. Keep them in sync.
- **Reporting:** the headline classification metric is **macro-F1**. Accuracy is secondary (Technical Issue is 53% of test). Always report overall, real-only (without synthetic rows) and per origin.
- **Test markers:** `@pytest.mark.slow` for tests that load large models, `@pytest.mark.integration` for tests that need the DB.
  - Go integration tests use the `//go:build integration` tag. Each package has one `TestMain` calling `storetest.Run(m, &testDB)`, and tests call `testDB.Require(t)`, which skips when `TEST_DATABASE_URL` is unset.
- **DB connections:** always go through `store.OpenPool`/`store.OpenSQL`, which set an explicit search_path. Python must `SET search_path TO public, extensions` on connect.
- **External services** (Groq, IMAP/SMTP, Zendesk, Google Calendar) sit behind interfaces with fakes. Unit tests run offline.
- **Env vars:** see `.env.example` (DATABASE_URL, TEST_DATABASE_URL, GROQ_*, ML_SERVICE_URL, ML_PORT, TORCH_DEVICE, API_KEY, IMAP_*, SMTP_*, ZENDESK_*, GOOGLE_*).
- **Model RAM (8 GB):** load models lazily, at most one transformer resident per role, fp16 on MPS for inference, record RSS + swap in eval runs. The SCRIBE baseline (BART-MNLI, about 1.6 GB) runs as its own CLI, never inside the always-on ml-service.

## 7. Known issues and gotchas

- **Supabase connection mode:** use the **Session pooler** connection string (IPv4, port 5432). The Transaction pooler (6543) breaks prepared statements in pgx/psycopg.
- **pgvector on Supabase** is pre-installed in the `extensions` schema (v0.8.2), not `public`. Any connection that overrides `search_path` must include `extensions`, or you get `type "vector" does not exist`.
- **Zendesk is only tested against an `httptest` mock**, since no Zendesk account is configured. The incremental export also requires an admin API token.
- **Email adapter not tested against a real mailbox:** IMAP_*/SMTP_* aren't configured. `IMAPClient` is untested beyond compiling. `SMTPMailer` is tested against an in-process fake SMTP server (no TLS, no auth). Gmail needs an app password, plus IMAP enabled.
- **DB latency (`cmd/dbping`, 2026-10-15):** after the move to ap-south-1 (Mumbai):
  - warm-pool `SELECT 1` p50 is 28–30ms, p95 31ms (was 395ms p50 in Sydney);
  - TCP connect is 31ms;
  - a fresh connection is 212ms.
  - The integration suite takes about 9s.
  - Ingestion still does 2 queries per ticket; batching is deferred to P6/P8 volume.
- **Zendesk is first-message-only (accepted limitation).** Follow-up comments on an existing ticket are not ingested. The incremental export is per ticket, and the `(source, external_id)` key drops repeats.
- **Invisible characters in generated Go files:** `\u` escapes written by Claude can land in files as literal BOM/nbsp/zero-width characters, and Go rejects a mid-file BOM. Fix with a small script that rewrites them as `\uXXXX` escapes. Check `go vet` after writing any file containing such escapes.
- **Kaggle token check:** Kaggle answers 404 to the token check for an invalid token, and the SDK reports that as "Authentication required". `python -m data.kaggle_auth` names the likely cause.
- **Integration test cost:** each package pays a one-off setup (create, migrate, drop its schema), which takes a few seconds since the move to Mumbai (it was about 26s in Sydney).
- **Supabase pausing:** free-tier projects pause after inactivity. If DB connections fail, check whether the project needs resuming in the dashboard.
- **HF downloads:** the baseline's first run downloads about 2 GB to `~/.cache/huggingface`. The "unauthenticated requests" warning is harmless; set `HF_TOKEN` for faster downloads. transformers 5 prints an UNEXPECTED `pooler` keys load report for twitter-roberta, which is expected (the classification head doesn't use the pooler).
- **Groq models change:** `llama-3.3-70b-versatile` isn't available on the current key. Run `GET {GROQ_BASE_URL}/models` to see which models are live.
- **Tests vs `.env`:** the pytest conftest loads `.env` into the environment. Tests that check settings defaults must clear those variables (see `test_build_llm_roles`).
- **pytest warning:** a StarletteDeprecationWarning from TestClient asks for `httpx2`. It's harmless and left as is (review decision).
- **Render Go version:** `go.mod` declares `go 1.26.0`. At P8.2, check that Render's native Go runtime supports it, or pin a toolchain.
- **Database safety:** never run destructive SQL (DROP, TRUNCATE, down-migrations) against `DATABASE_URL` without asking Arya. Tests only use `TEST_DATABASE_URL`, with a per-run schema.
- **`.env` quoting:** put any value containing `$` in single quotes. Make never reads `.env` (see the decisions log).
- **Dev machine:** an M1 MacBook Air with 8 GB RAM. Avoid holding several large models in memory.

## 8. Results so far

- Ungrounded hallucination example: `results/examples/ungrounded_hallucination_01.md`. The SCRIBE baseline told a customer "I've processed a refund … within 3–5 business days" with no grounding. P7.2 collects more automatically.
- **Spam filter (P3.9, `results/spam_filter/`, `spam-20261017022247-319c633f`):** AP 0.9929 / ROC-AUC 0.9996 on spam_test (val 0.9958 / 0.9998).
  - Rule `support` (default; mute ≥ 0.0195, review ≥ 0.0052): spam recall 1.0 at mute; support FPR 0.0099 (val 0.0099); support in the review band 0.0449; **ham muted 0.2104** + 0.2594 in review (test).
  - Rule `support_and_ham` (mute ≥ 0.4365, review ≥ 0.0941): test spam recall 0.9783 at mute / 0.9957 with review; support FPR 0.0; ham FPR 0.0212; ham in review 0.0424.
- Data audit after the P3.7 rebuild: 0 near-duplicate pairs train-pool~val, train-pool~test and val~test across both tasks (29,252-row corpus). Topic diagnostic TF-IDF+LR on val: macro-F1 0.777, accuracy 0.728 (5 classes).
- **Fast tier on val (P3.1, `results/fast_tier/`, `fast-20261016012651-6f7999df`; val only, test untouched):**
  - Macro-F1 **0.825**, accuracy 0.788 (n=3,775). Real only = without Meeting (n=3,715): macro-F1 **0.790**, accuracy 0.785.
  - Per-class F1: Technical 0.835, Billing 0.675, Feature 0.951, General Inquiry 0.491, Spam 1.000, Meeting 1.000. General Inquiry is the weak class (P 0.53 / R 0.45).
  - Real val tickets mentioning a meeting word (n=173): accuracy 0.873, 0 predicted Meeting.
  - Coverage (calibrated confidence ≥ τ → share / accuracy): 0.5 → 87% / 0.834; 0.6 → 68% / 0.899; 0.7 → 50% / 0.934; 0.8 → 31% / 0.977; 0.9 → 18% / 0.992.
  - Cost: training 7.9 min (4 C fits + 2 calibrated fits); load 0.5 s; single ticket p50 1.7 ms, p95 2.5 ms; batch 0.5 ms/ticket; peak RSS 612 MB (7.5 GB swap in use).
  - Not comparable with the baseline yet: different split (val vs test). P7.1 scores both on test.
- **Out-of-domain sample check (P3.5a, `results/ood/sample_emails.md`, n=22 hand-written, not a metric):** fast tier / cascade 6/22 correct with 13 predicted Spam (gold Spam = 1); SCRIBE baseline 19/22 with 1 predicted Spam.
- **Urgency heuristic on val (P3.3, `results/urgency/`, n=3,485 labelled rows):** macro-F1 **0.273**, accuracy 0.419, ordinal MAE 0.615, off by ≥ 2 on 3.2%.
  - It predicts medium for 84% of rows (2,922), so it's a conservative fallback, not a classifier. Gold is medium 1,429 / high 1,366 / low 690.
  - 74 rows are predicted critical; gold has no critical, so all 74 count as errors.
  - Fired groups: urgency 551, impact 215, deadline 88, low 17, escalation 10, negative 2. The Tobi-Bueck texts rarely use escalation or negative phrasing.
  - For reference only (different split): SCRIBE's sentiment-as-urgency on test has macro-F1 0.357 / MAE 0.750. The heuristic is closer on the ordinal scale but worse on macro-F1. The fine-tuned urgency model is expected to replace both.
- **SCRIBE baseline on the superseded test split (P2.3, `results/baseline/`, test content `c6539fff…`, n=3,775, MPS, 2026-10-16; re-run on the new test in P7.1):**
  - Overall macro-F1 **0.522**, accuracy 0.527 (weighted F1 0.557).
  - Real only = without Meeting Scheduling (n=3,715, 5 classes): macro-F1 **0.479**, accuracy 0.521. Synthetic Meeting rows (n=60): recall 0.917.
  - Per class F1: Technical 0.657 (P 0.83 / R 0.55), Billing 0.444 (R 0.32), Feature 0.573, General Inquiry 0.368 (P 0.25 / R 0.73), Spam 0.349 (P 0.96 / R 0.21), Meeting 0.738 (P 0.62 / R 0.92).
  - Dominant error: zero-shot sends everything to General Inquiry (1,818 of 3,712 zero-shot predictions; 41% of Technical, 52% of Billing, 53% of Spam).
  - Rules decided 1.67% of tickets (63: 48 Spam, 15 Meeting), all correct (precision 1.0). Zero-shot accuracy 0.519 on the rest.
  - 34 real tickets were predicted Meeting Scheduling (false positives). Real tickets mentioning a meeting word (n=196): accuracy 0.418, 27 predicted Meeting.
  - Sentiment-as-urgency (n=3,485, low/medium/high): macro-F1 0.357, accuracy 0.397, ordinal MAE 0.750.
  - Latency (M1 8 GB, heavy swap; **not for comparison**, P7.1 re-measures): model load 16.5 s; batched 0.705 tickets/s (p50 911 ms/ticket); single ticket p50 640 ms, p95 1.9 s, max 73 s (swap).
  - Stress set: pending Arya's review.
- Data audit (P2.1g, `results/data_audit/report.md`):
  - After the P2.1h re-split, TF-IDF+LR (1–2-gram, balanced) trained on capped train and scored on val gives macro-F1 0.802 / accuracy 0.735 on full text and 0.709 / 0.647 on the subject alone. Meeting Scheduling F1 is 1.0 from either field. This is a data diagnostic, not the P3.1 fast tier.
  - Cross-split near-duplicate pairs: 0. Before the re-split, 11.6% of test rows had a train near-duplicate.

# Build Prompt — Agentic Support Triage System with Retrieval-Grounded Replies

You are building my capstone project from scratch in this repository. Read this whole document before writing any code. Work through it **phase by phase, part by part, in order**.

---

## 0. Project context

**Title:** An Agentic Support Triage System with Pluggable Ingestion and Retrieval-Grounded Reply Generation

**Base paper:** *SCRIBE: A Context-Aware AI Bot for Semantic Email Prioritization and Generative Communication* (Procedia Computer Science 283, 2026). SCRIBE uses:
- regex rules with a fixed ~95% confidence cutoff;
- `facebook/bart-large-mnli` zero-shot classification as a fallback;
- `cardiffnlp/twitter-roberta-base-sentiment-latest` for sentiment;
- LLaMA-3-70B two-stage draft→refine replies.

It ingests CSV only and never grounds or fact-checks its replies.

**What this system adds over SCRIBE (the novelty — keep it visible in code and docs):**
1. **Pluggable ingestion.** One `Source` interface with email (IMAP), ticketing-API (Zendesk/Freshdesk) and CSV adapters.
2. **Learned fast classification tier.** TF-IDF + logistic regression replaces the regex rules, with a confidence threshold tuned on data instead of fixed.
3. **Fine-tuned deep tier.** A fine-tuned DeBERTa-v3 classifier replaces zero-shot BART-MNLI. *Arya trains it separately; see Section 2.*
4. **Urgency model.** A fine-tuned ordinal model (low / medium / high / critical) trained on email/ticket text replaces the tweet-trained sentiment model. *Arya trains it separately; see Section 2.*
5. **Retrieval-grounded replies.** Replies are built from a documentation knowledge base stored in pgvector, with cited sources.
6. **Faithfulness gate and abstain path.** Every claim in a draft is checked against the retrieved context. Unsupported drafts are not sent; the ticket is escalated to a human instead.
7. **Evaluation harness.** Reports per-class P/R/F1, retrieval recall@k, and RAGAS-style faithfulness. It compares the SCRIBE baseline against this system on the same test split.

**Architecture (six layers):**
```
1 Ingestion      Email(IMAP) | Ticketing API | CSV  ->  Adapter interface  ->  Preprocess & normalize
2 Classification Fast tier (TF-IDF+LR) -> confidence >= tau ? -> (no) Deep tier (fine-tuned DeBERTa-v3) -> Urgency model -> category + urgency + confidence
3 Routing        Rule engine -> Mute | Queue + escalation | Reply needed | Meeting scheduler (Google Calendar)
4 Grounded reply Retriever (pgvector top-k) -> Draft generator (LLaMA 3 70B via Groq) -> Faithfulness gate -> Grounded draft | Abstain -> human
5 Serving        REST API (Go + Gin) -> Agent review UI (approve/edit/send) -> Outbound reply via source adapter ; Dashboard
6 Data/eval      Supabase Postgres + pgvector | KB ingestion | Google Calendar API | Evaluation harness | Monitoring
```

---

## 1. Ground rules (follow for every part)

### Git workflow — mandatory
- Initialise the repo in P0.1 if it isn't already a git repo. Work on `main` unless I say otherwise.
- **Commit after every part, as soon as that part's "Done when" criteria pass.** Never batch several parts into one commit. Never leave a part half-done when you commit.
- Before each commit:
  - run the formatter and linter (`gofmt`/`go vet` for Go, `ruff` for Python);
  - run the tests that cover that part;
  - fix anything that fails. Never commit with failing tests.
- Commit message format, using the Conventional Commits types `feat|fix|test|docs|chore|refactor`:
  ```
  <type>(<scope>): <summary>  [P<phase>.<part>]

  - bullet list of what changed
  - acceptance criteria met
  ```
  Example: `feat(ingest): add CSV source adapter  [P1.2]`
- Do **not** push, create branches, rewrite history, or force anything unless I ask. I push to GitHub myself so I can sync the repo to my other machine.
- Never commit secrets, `.env`, model weights (`models/`), datasets (`datasets/raw`, `datasets/processed`), or `token.json`/`credentials.json`. Put them in `.gitignore` in P0.1 and provide `.env.example`.

### CLAUDE.md — the project's living memory
Create `CLAUDE.md` at the repo root in P0.1. Claude Code loads this file automatically at the start of every session, so it must always be enough for a fresh session to pick up exactly where the last one stopped. Keep it current. Update it as part of **every** part's commit, and again whenever something important changes mid-part.

Required sections, kept in this order:
1. **Project summary.** Three to five lines: what this is, the base paper, and the novelty over SCRIBE.
2. **How to run.** The exact commands to start, test, lint, migrate, seed and evaluate. Keep them accurate as they change.
3. **Status board:**
   - **Done** — part id, one-line summary, commit hash.
   - **In progress** — the current part, what's finished inside it, what's next inside it.
   - **To do** — every remaining part id with its title, in order.
   - **Blocked / needs input** — what's blocked and what's needed to unblock it.
4. **Model artifacts.** For `models/deep_tier/` and `models/urgency/`: present or absent, and if present, the base model, training date and validation metrics from its `model_card.json`.
5. **Decisions log.** Every non-obvious choice: what was chosen, the alternatives, why, and the date. Include defaults you picked because the spec was ambiguous.
6. **Conventions.** Commit format, folder layout, contracts location, test markers, env vars.
7. **Known issues and gotchas.** Flaky tests, slow steps, model download sizes, Supabase quirks, API rate limits.
8. **Results so far.** Headline metrics from real runs only, each with the path to its file under `results/`.

Rules:
- Keep it concise and scannable: aim for under ~250 lines. Move long detail into `docs/` and link to it.
- Never record a metric or a "done" status that isn't true. If a part was only partly verified, say exactly what wasn't verified.
- Never put secrets or connection strings in it.

### Coordinating with my other Claude (status updates)
I'm coordinating this build with a separate Claude chat that designed the architecture and this prompt. I'll relay your updates there and bring its feedback back to you, so write updates that make sense to a reader who **cannot see this repo or your terminal**.

**After every part**, once you've committed, print a status update in exactly this format inside a fenced code block, so I can copy it in one go:

```
=== STATUS UPDATE — P<phase>.<part> <part title> ===
Result: DONE | PARTIAL | BLOCKED
Commit: <short hash> "<commit subject>"
Built: <2–5 bullets on what now exists>
Verified: <tests/commands run and their outcome; say plainly what was NOT verified>
Decisions: <non-obvious choices made, or "none">
Deviations from BUILD_PROMPT: <anything done differently and why, or "none">
Issues/risks: <bugs, flaky bits, slow steps, open questions, or "none">
Numbers: <any real metrics produced, with the results/ path, or "none">
Next: <next part id and title>
Need from Arya: <decisions/credentials/files/model artifacts needed, or "nothing">
```

**At the end of each phase**, also print a longer update in the same code-block style:

```
=== PHASE <n> REVIEW — <phase name> ===
Summary: <what the phase delivered, 3–6 bullets>
How to run/demo: <exact commands>
Parts: <each part id → DONE/PARTIAL with commit hash>
Test status: <totals; anything skipped and why>
Real results: <metrics with file paths, or "none this phase">
Deviations + rationale: <list>
Open questions for review: <specific questions I should ask the other Claude>
Proposed adjustments for next phase: <if any>
```

Then **stop and wait**. When I paste feedback from the other Claude, treat it as review input:
- apply what's clearly correct;
- push back with reasons where you disagree, or where it conflicts with what you can see in the code;
- record the outcome in the CLAUDE.md decisions log.

Keep updates factual and short. No marketing language, and never claim something works without having run it.

### Engineering rules
- Keep each part small and reviewable. If a part grows past roughly 400 changed lines, split it into sub-parts (e.g. P3.3a, P3.3b), each with its own commit.
- External services (Groq, IMAP/SMTP, Zendesk, Google Calendar) must sit behind an interface with a fake/mock implementation, so unit tests run offline with no API keys.
- Testing: write focused tests for logic (rules, routing, chunking, threshold selection, parsers). Don't over-test glue code.
  - Tests must not download large models; mark model-dependent tests `@pytest.mark.slow` and skip them by default.
  - Tests that need the database are marked `integration` and are skipped when `TEST_DATABASE_URL` is unset.
- **Never invent results.** Metrics in docs or reports must come from an actual run, saved under `results/`. If something couldn't be run (for example, a fine-tuned model isn't present yet), say so explicitly.
- If a requirement is ambiguous and the choice is hard to reverse, ask me. Otherwise pick the sensible default, note it in `CLAUDE.md`, and continue.

---

## 2. Environment, stack and repo layout

### Development machine
- Development happens on an **M1 MacBook Air** (Apple Silicon, possibly 8 GB RAM). Everything must run natively on macOS arm64.
- **No Docker is required for development.** Go and Python run as native processes, and the database is hosted on Supabase.
- Be careful with memory. Don't keep large models loaded when they aren't needed: load them lazily, and give the SCRIBE baseline's models (BART-MNLI ~1.6 GB) their own process or command, not the always-on service.
- Use PyTorch with the MPS device when available, falling back to CPU. Device selection must be automatic and overridable via `TORCH_DEVICE`.

### Fine-tuned models — DO NOT train them
The **deep tier (DeBERTa-v3)** and the **urgency model** are fine-tuned **by me, separately, on my GPU PC**. I then copy the artifacts to this Mac under `models/`.

- **Do not write fine-tuning or training code for these two models.** Don't download base checkpoints to train them, and don't build Colab notebooks or Trainer configs.
- Only build **inference loaders** that read the artifact format below, plus graceful fallbacks for when the artifacts aren't there yet.
- When the deep tier is missing, the cascade runs **fast tier only**. When the urgency model is missing, a documented keyword/heuristic fallback is used. `/health` and every `Classification` response must report which models are loaded.
- When I later ask for training scripts, they become a separate part, added only on request.
- The fast tier (TF-IDF + logistic regression) **is** in your scope. It trains on CPU in seconds.

**Model artifact contract.** My models will arrive in this format, so the loaders must accept exactly this:
```
models/deep_tier/                       models/urgency/
├── config.json                         ├── config.json
├── model.safetensors                   ├── model.safetensors
├── tokenizer.json (+ tokenizer files)  ├── tokenizer.json (+ tokenizer files)
├── label_map.json                      ├── label_map.json      # ordered: low, medium, high, critical
└── model_card.json                     └── model_card.json
```
Both are Hugging Face `AutoModelForSequenceClassification` directories.
- `label_map.json` is `{"0": "<label>", ...}`. For the deep tier, its labels must match `config/taxonomy.yaml`.
- `model_card.json` is `{"base_model", "task", "trained_on", "trained_at", "max_length", "val_metrics": {...}}`.

At load time, validate that:
- every required file is present;
- the labels match the taxonomy or the urgency scale;
- `max_length` is present.

If validation fails, log a clear error saying what's wrong and fall back. Never crash the service.

Write `docs/model_artifacts.md` describing this contract precisely. I'll follow it when training on the PC.

### Database — Supabase
- Postgres 16 with the `pgvector` extension, hosted on **Supabase**. Read the connection string from `DATABASE_URL`.
- Use Supabase's **Session pooler** connection string, which is IPv4-compatible. Do **not** use the Transaction pooler (port 6543): it breaks prepared statements in pgx/psycopg. Document this in `.env.example` and `CLAUDE.md`.
- The first migration runs `create extension if not exists vector;`. Use HNSW indexes for embeddings.
- Use a migration tool such as `goose` or `golang-migrate`, with SQL files in `migrations/`. Make every migration idempotent where reasonable.
- Integration tests use `TEST_DATABASE_URL`: create a uniquely named schema per test run, run the migrations into it, and drop it afterwards. Never run tests against `DATABASE_URL`.
- Never run destructive SQL (`DROP TABLE`, `TRUNCATE`, down-migrations) against `DATABASE_URL` without asking me first.
- Gotcha for `CLAUDE.md`: free-tier Supabase projects pause after inactivity. If connections fail, tell me to check whether the project needs resuming.

### Stack
- **Go 1.22+ with Gin:** ingestion adapters, preprocessing, routing/rule engine, scheduler, REST API, outbound replies, workers. Use pgx for Postgres.
- **Python 3.11 with FastAPI (`ml-service`):** classification (fast tier, deep-tier inference, urgency inference), embeddings, retrieval, draft generation, the faithfulness gate, the SCRIBE baseline, and evaluation. Use a virtualenv at `ml-service/.venv` and serve with uvicorn.
- **LLM:** Groq, model `llama-3.3-70b-versatile` (configurable via `GROQ_MODEL`). Handle rate limits (HTTP 429) with backoff, and cache identical requests during evaluation.
- **Embeddings:** `BAAI/bge-small-en-v1.5`, 384 dimensions (configurable). Small enough to run locally on the Mac.
- **Frontend:** a minimal server-rendered UI (Go templates + HTMX, or a small static page). No heavy SPA framework.
- **Process management:** a `Makefile`, plus a `Procfile` run by `honcho` (pip) so `make dev` starts the API, worker and ml-service together natively.

### Repo layout
```
.
├── cmd/
│   ├── api/            # Gin REST API + UI
│   └── worker/         # ingestion + pipeline worker
├── internal/
│   ├── ingest/         # Source interface + adapters (csv, imap, zendesk)
│   ├── preprocess/
│   ├── store/          # Postgres repositories
│   ├── mlclient/       # HTTP client for ml-service
│   ├── routing/        # rule engine, queues, escalation
│   ├── scheduler/      # Google Calendar integration
│   ├── pipeline/       # orchestrates layers 1–4 per ticket
│   └── outbound/
├── web/                # templates/static for review UI + dashboard
├── ml-service/
│   ├── app/            # FastAPI: /classify, /retrieve, /draft, /verify, /health
│   ├── triage/         # fast_tier (train+infer), deep_tier (infer only), urgency (infer only + fallback), cascade
│   ├── rag/            # kb_ingest, retriever, generator, faithfulness
│   ├── baseline/       # SCRIBE re-implementation
│   ├── data/           # dataset builders + taxonomy mapping
│   ├── eval/           # evaluation harness
│   └── tests/
├── migrations/
├── config/             # routing_rules.yaml, taxonomy.yaml
├── docs/               # model_artifacts.md, architecture.md, deploy.md, ...
├── models/             # gitignored; fast_tier/ (yours), deep_tier/ + urgency/ (mine, copied in)
├── datasets/           # gitignored raw/processed data (+ committed datasets/README.md)
├── results/            # eval outputs (metrics CSV/JSON + figures; commit these)
├── Procfile
├── Makefile
├── .env.example
├── CLAUDE.md           # living status + decisions (see Ground rules)
└── README.md
```

### Shared data contracts
Define these exactly once in each language: Go structs in `internal/`, Pydantic models in `ml-service/app/schemas.py`. Keep the two in sync.

```
Ticket          { id, source ("email"|"zendesk"|"csv"), external_id, sender, subject, body,
                  received_at, thread_id?, raw_ref, content_hash }
Classification  { ticket_id, category, category_confidence, tier ("fast"|"deep"),
                  urgency ("low"|"medium"|"high"|"critical"), urgency_confidence,
                  urgency_source ("model"|"heuristic"), models_loaded {deep_tier: bool, urgency: bool},
                  model_versions }
Route           { ticket_id, action ("mute"|"queue"|"reply"|"schedule"), queue, priority, sla_due_at, reason }
RetrievedChunk  { chunk_id, doc_id, title, text, score }
ReplyDraft      { ticket_id, subject, body, citations[chunk_id], faithfulness_score,
                  unsupported_claims[], status ("grounded"|"abstained"|"approved"|"sent"|"rejected") }
```

**Category taxonomy:** start from SCRIBE's six categories (Work, Spam, Personal, Urgent, General, Meeting Scheduling). Extend it in `config/taxonomy.yaml` with support-ticket categories (Billing, Technical Issue, Account/Access, Feature Request, Complaint). Map every dataset's labels onto this taxonomy. Urgency is a **separate axis** from category; do not reuse the "Urgent" category as the urgency signal.

---

## 3. Phases

Each part lists **Build** and **Done when**. Commit when "Done when" holds.

### Phase 0 — Foundations

**P0.1 Repo scaffold**
- Build:
  - the repo layout above with a Go module and a Python package (`pyproject.toml`, ruff, pytest with `slow` and `integration` markers);
  - `.gitignore`, and `.env.example` listing: DATABASE_URL, TEST_DATABASE_URL, GROQ_API_KEY, GROQ_MODEL, ML_SERVICE_URL, TORCH_DEVICE, IMAP_*, SMTP_*, ZENDESK_*, GOOGLE_*, API_KEY, each with a one-line comment;
  - a `Makefile` with targets `setup`, `dev`, `api`, `worker`, `ml`, `test`, `test-integration`, `lint`, `migrate`, `seed`;
  - a `Procfile`;
  - `CLAUDE.md` with every required section filled in, and the status board listing every part P0.1–P8.3 under To do;
  - a stub `README.md`.
- Done when: `make setup`, `make lint` and `make test` all run on macOS arm64, even if there are no tests yet.

**P0.2 Database + migrations (Supabase)**
- Build:
  - migrations that enable `vector` and create `tickets`, `classifications`, `routes`, `reply_drafts`, `kb_documents`, `kb_chunks` (with `embedding vector(384)` and an HNSW cosine index), and `audit_log`;
  - a unique constraint on `(source, external_id)` and an index on `content_hash`;
  - a migration runner wired to `make migrate`;
  - an integration-test helper that creates and drops a per-run schema in `TEST_DATABASE_URL`.
- Done when:
  - `make migrate` applies cleanly to Supabase, and running it a second time is a no-op;
  - a Go integration test inserts and reads back a ticket in a throwaway schema.
  - If I haven't provided `DATABASE_URL` yet, stop and list exactly what I need to do in Supabase: create the project, copy the Session pooler string, fill in `.env`.

**P0.3 ml-service skeleton + Go client**
- Build:
  - a FastAPI app with `/health`, which reports service status plus which models are loaded (all absent at this stage);
  - Pydantic schemas for all the shared contracts;
  - a Groq client wrapper with backoff, plus a `FakeLLM` for tests;
  - automatic device selection (MPS/CPU);
  - a Go `mlclient` with timeouts and retries;
  - `make ml` starts uvicorn natively, and `make dev` starts everything via honcho.
- Done when: `make dev` brings up the API stub and ml-service, `/health` returns OK, and a Go test hits `/health` through `mlclient`.

**→ Stop: Phase 0 review.**

### Phase 1 — Ingestion (Layer 1, Go)

**P1.1 Source interface + Ticket model**
- Build:
  - `type Source interface { Name() string; Fetch(ctx, since time.Time) ([]Ticket, error); Reply(ctx, Ticket, ReplyDraft) error }`;
  - a registry that enables adapters from config.
- Done when: the interface, the registry, and a `FakeSource` used in tests all compile, and the tests pass.

**P1.2 CSV adapter**
- Build: reads `Sender, Subject, Body[, Label]` (SCRIBE-compatible) into `Ticket`. `Reply` writes to an output CSV.
- Done when: `datasets/sample_emails.csv` is ingested in a test. Commit a small, hand-written ~20-row sample file; it is not a dataset.

**P1.3 IMAP adapter**
- Build:
  - polls unseen messages since a checkpoint;
  - parses multipart/plain/html bodies;
  - strips quoted replies and signatures;
  - captures `Message-ID` and `In-Reply-To` for threading.
  - `Reply` sends via SMTP. Both IMAP and SMTP sit behind interfaces.
- Done when: parser tests pass on fixture `.eml` files (plain, html, multipart, a reply chain).

**P1.4 Ticketing API adapter (Zendesk; Freshdesk optional)**
- Build:
  - fetches tickets via the incremental API with pagination;
  - `Reply` posts a public comment;
  - an `httptest` mock server for tests.
- Done when: tests against the mock cover pagination, the 429 retry, and posting a reply.

**P1.5 Preprocess & normalize**
- Build:
  - HTML→text, whitespace normalization;
  - quote/signature stripping shared by all adapters;
  - `content_hash` dedupe;
  - language check (English only — skip others with a logged reason);
  - persist to `tickets`.
- Done when: table-driven tests cover each rule, and an integration test shows duplicate tickets are rejected.

**P1.6 Ingestion worker**
- Build:
  - a `cmd/worker` loop that polls the enabled sources on an interval;
  - stores checkpoints;
  - writes `audit_log` entries;
  - shuts down gracefully.
- Done when: running the worker with the CSV source fills the DB idempotently, so a second run adds 0 tickets.

**→ Stop: Phase 1 review.**

### Phase 2 — Data and SCRIBE baseline

**P2.1 Dataset builders**
- Build: in `ml-service/data/`, loaders for:
  - Kaggle labeled support-ticket datasets (Customer IT Support / multilingual tickets: English rows only; Support Ticket Priority 50K);
  - an optional Enron + LLM-synthetic email set, as in the SCRIBE paper.

  Then:
  - map labels to `taxonomy.yaml`;
  - derive urgency labels (low/medium/high/critical) from priority fields where present;
  - create stratified `train/val/test` splits with a fixed seed;
  - save splits as parquet under `datasets/processed/` and write a dataset card (`datasets/README.md`) with class counts per split.

  Datasets are downloaded by script and never committed. If Kaggle credentials are missing, print exact instructions instead of failing silently.

  **These splits are what I'll use to fine-tune my models on the PC.** Add `make export-splits`, which bundles the processed splits, `taxonomy.yaml` and the urgency scale into one archive I can copy to the PC. Document it in `docs/model_artifacts.md`. The test split must stay the same across machines, so fix the seed and record a hash of the split files.
- Done when: `make data` produces the splits and the dataset card, `make export-splits` produces the archive, and the mapping logic is unit-tested.

**P2.2 SCRIBE baseline**
- Build: in `ml-service/baseline/`, re-implement SCRIBE's classifier faithfully:
  - regex rules with the 0.95 cutoff;
  - BART-MNLI zero-shot fallback against natural-language category descriptions;
  - twitter-roberta sentiment mapped to −1/0/+1;
  - the two-stage draft→refine reply with no grounding.

  Expose it behind the same `classify`/`draft` interfaces as the proposed system, so the evaluation can swap them. Run it as its own CLI or command, not inside the always-on ml-service (RAM).
- Done when: the rule layer is unit-tested, and the slow tests (opt-in) run end to end on 10 samples.

**P2.3 Baseline evaluation snapshot**
- Build: run the baseline on the test split.
- Done when: `results/baseline/` has the outputs below, generated from a real run.
  - `metrics.json` with accuracy, macro and per-class P/R/F1, and the share of tickets handled by rules;
  - a confusion-matrix PNG;
  - a latency table.

**→ Stop: Phase 2 review, including the actual baseline numbers. Note in "Need from Arya" that the splits are ready to export for fine-tuning.**

### Phase 3 — Classification (Layer 2, Python)

**P3.1 Fast tier**
- Build:
  - TF-IDF (word 1–2 grams plus char 3–5 grams) + logistic regression with `class_weight="balanced"`;
  - probability calibration via `CalibratedClassifierCV`;
  - a training script that saves to `models/fast_tier/` with a version stamp;
  - an inference module.
- Done when: the script trains on the train split, reports validation metrics to `results/fast_tier/`, and a load/predict test passes.

**P3.2 Deep-tier inference loader (no training)**
- Build:
  - a loader for `models/deep_tier/` following the artifact contract, with validation, lazy loading, MPS/CPU, batching and truncation to `max_length`;
  - when the artifact is absent or invalid → `available = False` with a logged reason.
- Done when: tests cover (a) a missing directory, (b) an invalid `label_map`, and (c) the happy path using a **tiny randomly initialised model built in the test**. Build a small config, `save_pretrained` it to a temp dir, add a tokenizer and the JSON files. Nothing is downloaded.

**P3.3 Urgency inference loader + heuristic fallback (no training)**
- Build:
  - a loader for `models/urgency/` (same contract, ordinal labels);
  - a heuristic fallback that combines urgency keywords, deadline/time expressions, escalation language, and negative phrasing. It must be documented and unit-tested, and it marks `urgency_source="heuristic"`;
  - ordinal-aware evaluation utilities: macro-F1 plus mean absolute error over the ordinal positions.
- Done when: the loader tests (same three cases as P3.2) and the heuristic tests pass, and the evaluation utility is tested on toy arrays.

**P3.4 Cascade + threshold tuning**
- Build:
  - the cascade: run the fast tier → if calibrated confidence ≥ τ accept, else use the deep tier if available, otherwise accept the fast-tier prediction with a `low_confidence` flag → always run urgency (model or heuristic);
  - `tune_threshold.py` sweeps τ on the validation split and plots accuracy vs. the share of tickets sent to the deep tier (cost curve). Pick τ by a documented rule (e.g. the smallest deep-tier share within 0.5 pt of the best macro-F1) and save τ to config;
    - **If the deep tier isn't available, the script exits cleanly with a message saying it needs `models/deep_tier/`.** Don't fake the curve;
  - a `/classify` endpoint returning `Classification`, including `tier`, `models_loaded` and `model_versions`.
- Done when:
  - cascade logic is unit-tested with stub models, both with and without the deep tier;
  - `/classify` works with only the fast tier present;
  - the tuning script is tested with stubs.

**P3.5 Go integration**
- Build: the pipeline calls `/classify` for new tickets and persists the `classifications` row.
- Done when: an integration test with a stubbed ml-service classifies ingested tickets.

**→ Stop: Phase 3 review. In "Need from Arya", say clearly that the deep tier and urgency model can now be dropped into `models/`, and what to run afterwards (validation command, threshold tuning).**

### Phase 4 — Routing and actions (Layer 3, Go)

**P4.1 Rule engine**
- Build:
  - `config/routing_rules.yaml` maps `(category, urgency, confidence, low_confidence flag)` to an action, queue and priority, with first-match semantics and a default rule;
  - hot reload on file change.
- Done when: table-driven tests cover every rule plus the default.

**P4.2 Mute + no-reply detection**
- Build: Spam and no-reply cases (FYI, newsletters, `no-reply` senders, automated notifications) route to `mute` and are recorded with a reason.
- Done when: tests pass.

**P4.3 Queues + SLA escalation**
- Build:
  - set `sla_due_at` per priority;
  - an escalation job raises the priority of breached tickets and writes to `audit_log`.
- Done when: SLA computation and escalation are tested with a fake clock.

**P4.4 Meeting scheduler**
- Build:
  - LLM extraction of date/time/duration/venue via ml-service (Groq JSON mode);
  - a Google Calendar free/busy check: create the event if free, otherwise find the next free working-hours slot and produce a counter-offer;
  - OAuth2 installed-app flow;
  - a `FakeCalendar` for tests.
- Done when: the scheduling logic is tested with `FakeCalendar` (free, busy→counter-offer, no slot found).

**→ Stop: Phase 4 review.**

### Phase 5 — Grounded reply (Layer 4, Python)

**P5.1 KB ingestion**
- Build:
  - load a documentation corpus: TechQA (IBM technotes) or WixQA (Wix help-centre KB + QA pairs), and also any folder of `.md`/`.html` docs;
  - heading-aware chunking (~400 tokens, ~60-token overlap) with doc/title metadata;
  - embed with bge-small and upsert into `kb_chunks` on Supabase in batches;
  - keep the QA pairs for evaluation.
- Done when:
  - chunking is unit-tested;
  - `make kb` fills pgvector;
  - `CLAUDE.md` records the chunk count and approximate storage used, which matters against Supabase free-tier limits.

**P5.2 Retriever**
- Build:
  - query = subject + cleaned body;
  - pgvector cosine top-k (default k=5), with an optional hybrid BM25 merge behind a flag;
  - a `/retrieve` endpoint.
- Done when: `eval/retrieval.py` reports recall@1/3/5 and MRR on the KB QA pairs, and the results are saved to `results/retrieval/`.

**P5.3 Draft generator**
- Build:
  - a Groq prompt that must answer **only** from the numbered retrieved chunks and cite them as `[n]`;
  - tone set from urgency and category;
  - JSON output `{subject, body, citations}`;
  - SCRIBE's draft→refine second pass, kept but constrained to add no new facts;
  - a `/draft` endpoint.
- Done when: prompt assembly and citation parsing are tested with `FakeLLM`.

**P5.4 Faithfulness gate + abstain**
- Build:
  - split the draft into atomic claims and judge each claim as supported or unsupported against the cited chunks (LLM-as-judge, temperature 0, JSON output);
  - `faithfulness_score` = the fraction of claims that are supported;
  - status rules:
    - below threshold (default 0.9), or retrieval max score below a floor → `abstained`, with `unsupported_claims` recorded;
    - otherwise → `grounded`;
  - a `/verify` endpoint.
- Done when: the gate logic is tested with `FakeLLM` covering supported, partly unsupported, and empty retrieval.

**P5.5 Pipeline wiring**
- Build:
  - Go pipeline: a `reply` action calls retrieve → draft → verify;
  - persist the `ReplyDraft`;
  - abstained drafts go to a human queue with a reason.
- Done when: an end-to-end test with a stubbed ml-service takes a CSV ticket to a stored grounded draft, and another ticket to an abstention.

**→ Stop: Phase 5 review.**

### Phase 6 — Serving and review (Layer 5)

**P6.1 REST API**
- Build: Gin endpoints:
  - `GET /tickets` (filters: category, urgency, status, queue)
  - `GET /tickets/:id`
  - `GET /tickets/:id/draft`
  - `POST /drafts/:id/approve|reject|edit`
  - `POST /ingest/csv` (upload)
  - `GET /metrics/summary`

  Use request validation, consistent JSON errors, and API-key middleware.
- Done when: handler tests pass and an OpenAPI spec is written to `docs/openapi.yaml`.

**P6.2 Agent review UI**
- Build:
  - an inbox view sorted by priority/SLA;
  - a ticket view showing the classification (including which tier and whether urgency came from the model or the heuristic), route, draft with highlighted citations (click to view the source chunk), faithfulness score, and any unsupported claims;
  - approve / edit / reject actions.
- Done when: the UI works against seeded data, and you describe a manual test in `CLAUDE.md`.

**P6.3 Outbound reply**
- Build:
  - approved drafts are sent through the originating adapter's `Reply` (SMTP/Zendesk/CSV);
  - status → `sent`, recorded in `audit_log`;
  - an idempotency key per draft.
- Done when: tests with `FakeSource` confirm exactly one send per approval.

**P6.4 Dashboard**
- Build: tiles and charts for:
  - volume by source and category;
  - urgency distribution;
  - SLA breaches;
  - fast-vs-deep tier share;
  - abstain rate;
  - average faithfulness;
  - models loaded;
  - top senders (SCRIBE-style analytics).
- Done when: the dashboard renders from `/metrics/summary`.

**→ Stop: Phase 6 review.**

### Phase 7 — Evaluation and monitoring (Layer 6)

**P7.1 Classification evaluation: baseline vs proposed**
- Build: run, on the **same test split**:
  - the SCRIBE baseline;
  - the fast tier alone;
  - the deep tier alone (if present);
  - the full cascade;
  - the urgency model vs the heuristic (if the model is present).
- Done when: `results/classification/` contains the outputs below, all from real runs.
  - accuracy, macro and per-class P/R/F1, and confusion matrices for each;
  - urgency macro-F1 and ordinal MAE;
  - p50/p95 latency and the deep-tier invocation share;
  - a SCRIBE Fig. 4-style grouped bar chart.
  - Configurations that couldn't run because a model was missing are listed as "not run", not omitted silently.

**P7.2 RAG evaluation**
- Build: on the KB QA pairs, generate replies:
  - with grounding;
  - **without** grounding, i.e. SCRIBE-style.

  Compute faithfulness, answer relevance and context precision/recall (RAGAS-style: the `ragas` library if it works with Groq, otherwise a documented equivalent implementation), plus hallucination rate, abstain rate, and abstain precision (abstentions on questions that really have no KB answer). Include deliberately out-of-KB questions to test abstention.

  Respect Groq rate limits: use backoff, a response cache, and a `--limit` flag for quick runs.
- Done when: `results/rag/` holds the metrics and a comparison table from real runs.

**P7.3 Monitoring**
- Build:
  - a `metrics_daily` table plus a job that computes category drift: the Population Stability Index (PSI) of the category distribution vs. the training distribution;
  - p95 latency per stage, abstain rate, and SLA breach rate;
  - show them on the dashboard, and flag PSI > 0.2.
- Done when: the PSI computation is unit-tested and the job populates the table.

**P7.4 Results report**
- Build: `make report` generates `results/REPORT.md` from the files in `results/`: tables, figures, and what improved or regressed vs. the baseline, with honest caveats about synthetic data, dataset differences, and any missing models.
- Done when: the report builds entirely from saved results, with no hand-typed numbers.

**→ Stop: Phase 7 review, with the headline numbers.**

### Phase 8 — Deployment and documentation

**P8.1 Demo setup**
- Build:
  - `make demo` seeds Supabase with the sample CSV and a small KB, runs the pipeline, and starts everything natively, so the review UI opens with tickets and drafts ready;
  - an **optional** `docker-compose.yml` for a one-command containerised demo. It points at Supabase via `DATABASE_URL` and contains no local Postgres. Nothing else may depend on Docker.
- Done when:
  - a fresh clone + `.env` + `make setup && make migrate && make demo` gives a working UI on the Mac;
  - the compose file builds, if Docker is available. If it isn't, report that as not verified.

**P8.2 Cloud deploy**
- Build:
  - a Render blueprint (`render.yaml`) using native Go and Python runtimes for the api/worker/ml-service, with Supabase as the database;
  - model artifacts loaded from a configurable path or URL;
  - a CPU-only profile that works with only the fast tier present.
- Done when: the config validates, and deploy steps are documented in `docs/deploy.md`.

**P8.3 Docs**
- Build:
  - a README covering the overview, the architecture diagram (ASCII or embedded image), a SCRIBE vs this-system table, setup (Supabase + native), run, evaluation, adding fine-tuned models, and repo map;
  - `docs/architecture.md` with per-layer design decisions;
  - `docs/demo_script.md`, a five-minute walkthrough for the review panel.
- Done when: the docs are accurate to the final code, and every command in them has been run.

**→ Stop: final review — what's built, the headline metrics, known limitations, and suggested next steps.**

---

## 4. Start now

Begin with **P0.1**. Before writing code, reply with:
1. a short plan for Phase 0;
2. what you need from me for Phase 0 (expected: a Supabase project with `DATABASE_URL` and `TEST_DATABASE_URL`, and a Groq API key);
3. any other blocking questions (only if they are truly blocking).

Then implement P0.1, create `CLAUDE.md`, commit, and print the P0.1 status update block.

**If this is a new session partway through the build:** read `CLAUDE.md` first, check `git log --oneline` against its status board, resolve any mismatch, then continue from the "In progress" part.
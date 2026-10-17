# GroundedTriage

GroundedTriage is an agentic support-triage system. It ingests tickets from email, Zendesk or CSV and classifies them by topic, spam and urgency. It then routes them with SLAs, books meetings, and drafts replies **grounded in a knowledge base, with citations**. A faithfulness gate makes the drafter abstain rather than guess. A human agent reviews every reply before it is sent.

It is a capstone project that extends **SCRIBE** (Procedia CS 283, 2026). SCRIBE combines regex rules, BART-MNLI zero-shot classification, sentiment-as-priority and ungrounded LLaMA-3 replies over CSV input.

> **Status:** the system is complete through serving, monitoring and deployment config. The headline evaluation, `results/REPORT.md` (P7.4), is generated once the fine-tuned models and human labels exist. `CLAUDE.md` holds the live status board, every design decision, and the results measured so far.

## SCRIBE vs GroundedTriage

| | SCRIBE (baseline, reimplemented in `ml-service/baseline/`) | GroundedTriage |
|---|---|---|
| Ingestion | CSV | IMAP/SMTP, Zendesk, CSV, behind one `Source` interface with checkpoints and dedupe |
| Preprocessing | — | HTML→text, quote/signature stripping, language check, content-hash dedupe |
| Spam | regex rules | binary spam filter (TF-IDF+LR) with a support-ticket FPR ≤ 1% operating point and a review band |
| Topic | regex + BART-MNLI zero-shot | cascade: TF-IDF+LR fast tier → (below τ) fine-tuned DeBERTa-v3 deep tier |
| Urgency | sentiment → priority | ordinal urgency model, with a documented heuristic fallback and a *critical* rule |
| Routing | — | hot-reloaded YAML rules, queues, SLAs with escalation, no-reply muting |
| Meetings | — | LLM extraction → Google Calendar (FreeBusy, book or counter-offer) |
| Replies | ungrounded draft → refine | retrieve (pgvector) → floor check → cited draft → constrained refine → **faithfulness gate** (abstain to a human) |
| Review | — | agent UI: approve, edit or reject; cited sources one click away; exactly-once sending |
| Evaluation | — | baseline vs proposed on fixed splits, an OOD set of human-written mail, RAG faithfulness and abstention, judge validation (κ) |
| Ops | — | daily drift (PSI), p95 latency per stage, abstain and SLA-breach rates; Render blueprint |

The same LLM (`openai/gpt-oss-120b` on Groq) drafts for both systems, so the reply comparison isolates the effect of grounding. SCRIBE's LLaMA-3-70B isn't available on the Groq key. The faithfulness judge comes from a different model family (`qwen/qwen3.8-27b`).

## Architecture

```
            ┌──────────── Go (Gin, pgx) ─────────────┐        ┌──────── Python ml-service (FastAPI) ────────┐
 IMAP ─┐    │ cmd/worker                             │  HTTP  │ /classify   spam filter → fast tier → τ →    │
 Zendesk├──►│  poll → preprocess/dedupe → classify ──┼───────►│             deep tier; urgency               │
 CSV  ─┘    │  → route (rules, SLA) → escalate       │        │ /extract_meeting   LLM (JSON)                │
            │  → meetings → replies ─────────────────┼───────►│ /retrieve   bge-small → pgvector top-k       │
            │  → outbound (exactly once)             │        │ /draft      cited draft → refine (gpt-oss)   │
            │  → monitoring (PSI, p95, rates)        │        │ /verify     claim-level judge (qwen) → gate  │
            │ cmd/api   REST + review UI + dashboard │        └───────────────────────┬──────────────────────┘
            └───────────────────┬────────────────────┘                                │
                                ▼                                                     ▼
                  Supabase Postgres + pgvector: tickets, classifications, routes, meetings,
                  reply_drafts, kb_documents/kb_chunks (HNSW), audit_log, metrics_daily
```

Per-layer design decisions: [`docs/architecture.md`](docs/architecture.md).

## Setup (macOS arm64, native; no Docker needed)

Prerequisites:
- Go ≥ 1.22 (the toolchain auto-downloads the `go.mod` version);
- Python 3.11;
- a Supabase project with pgvector (use the **Session pooler** URL, port 5432);
- a Groq API key.

```bash
cp .env.example .env        # DATABASE_URL, TEST_DATABASE_URL, GROQ_API_KEY, API_KEY (openssl rand -hex 24)
make setup                  # venv in ml-service/.venv, Go modules
make setup-models           # torch + transformers (embeddings, deep tier, urgency)
make migrate && make migrate-verify
make kb                     # WixQA help-centre KB into pgvector (~17 min, idempotent)
make fast-tier && make spam-filter   # train the in-repo models (minutes; needs `make data` once)
```

The deep tier and the urgency model are fine-tuned on a separate GPU machine. Their contract is in [`docs/model_artifacts.md`](docs/model_artifacts.md). Without them the cascade runs on the fast tier alone, flags low-confidence tickets, and uses the urgency heuristic.

## Run

```bash
make dev                    # api :8080 + worker + ml-service :8001
open http://127.0.0.1:8080/ # review UI (enter API_KEY)
make demo                   # 25 demo tickets → pipeline once → expected vs actual → make dev
make demo-reset             # remove ONLY the demo tickets (asks for confirmation)
```

- API: [`docs/openapi.yaml`](docs/openapi.yaml).
- Five-minute walkthrough: [`docs/demo_script.md`](docs/demo_script.md).
- Cloud deploy (Render + Supabase): [`docs/deploy.md`](docs/deploy.md).

## Evaluation

| What | Command | Output |
|---|---|---|
| SCRIBE baseline on test | `make baseline-eval` | `results/baseline/` |
| Fast tier / spam filter / urgency on val | `make fast-tier`, `make spam-filter`, `make urgency-eval` | `results/fast_tier/`, `results/spam_filter/`, `results/urgency/` |
| OOD gate (human-written mail; choices made on ood-dev) | `make ood-eval` (final systems: `make ood-eval-test`) | `results/ood/` |
| Retrieval (WixQA recall@k, MRR) | `make retrieval-eval` | `results/retrieval/` |
| Grounded vs ungrounded replies (resumable over days at the free cap) | `make rag-eval-plan`, `make rag-eval`, `make rag-floor-sweep` | `results/rag/` |
| Judge validation (human labels, κ) | `make judge-sample`, `make judge-label`, `make judge-score` | `results/rag/judge_validation.*` |
| Results report | `make report` (P7.4) | `results/REPORT.md` |

Reporting rules:
- macro-F1 is the headline classification metric;
- every result is reported overall, real-only and per origin (synthetic rows are labelled);
- test and ood-test are scored once per final system.

## Adding fine-tuned models

Copy the artifacts to `models/deep_tier/` and `models/urgency/`. They need `config.json`, `label_map.json`, `model_card.json`, `model.safetensors` and the tokenizer; [`docs/model_artifacts.md`](docs/model_artifacts.md) has the full contract. Then:

```bash
make ml && curl localhost:8001/health     # models_loaded true, or a logged reason
make urgency-eval && make tune-tau && make ood-eval
```

For a cloud deploy, bundle them with `make models-pack ARGS="fast_tier spam_filter deep_tier urgency"` and set `MODELS_URL` and `MODELS_SHA256`.

## Repo map

| Path | Contents |
|---|---|
| `cmd/` | `api`, `worker`, `migrate`, `demo`, `calendar-auth`, `dbping` |
| `internal/ingest`, `internal/preprocess` | source adapters (CSV, IMAP/SMTP, Zendesk), parsing, cleaning, dedupe |
| `internal/routing`, `internal/scheduler` | rule engine, no-reply detection, SLAs; calendar scheduling (Google, fake) |
| `internal/pipeline`, `internal/outbound` | worker steps (poll, classify, route, meetings, replies), exactly-once sending |
| `internal/api`, `internal/ui` | REST API and the embedded review UI and dashboard |
| `internal/store`, `internal/monitoring` | Postgres access; daily drift and latency metrics |
| `ml-service/app` | FastAPI app, contracts, LLM clients, settings, model registry, artifact bundles |
| `ml-service/triage` | fast tier, spam filter, deep tier and urgency loaders, cascade, threshold tuning |
| `ml-service/rag` | chunking, KB ingestion, retriever, generator, faithfulness gate |
| `ml-service/baseline` | the SCRIBE reimplementation (own CLI; never loaded in the service) |
| `ml-service/data`, `ml-service/eval` | dataset builders and audits; every evaluation |
| `config/` | taxonomy, routing rules, SLAs, cascade τ, grounding thresholds, monitoring reference, sources |
| `migrations/` | goose SQL (embedded, up only) |
| `datasets/` | committed: synthetic/stress/OOD/demo sets and cards; downloaded data is gitignored |
| `results/` | committed evaluation outputs |
| `docs/` | architecture, deploy, demo script, model artifact contract, OpenAPI |

## Data and licences

- Tobi-Bueck customer-support tickets (CC BY-NC 4.0);
- the SpamAssassin public corpus (no formal licence; downloaded by script, never redistributed);
- WixQA (MIT);
- Customer Support on Twitter (CC BY-NC-SA 4.0; 149 rows committed with attribution);
- Enron spam (CC0 copy);
- synthetic meeting requests and style paraphrases generated for this project (seeds and prompts committed).

Details are in `datasets/README.md` and `datasets/ood/README.md`.

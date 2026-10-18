# Architecture and design decisions

GroundedTriage is two processes around one database:
- a **Go** side (`cmd/api`, `cmd/worker`) for I/O, state, routing, review and sending;
- a **Python ml-service** (FastAPI) for every model and LLM call.

They share the contracts `Ticket`, `Classification`, `Route`, `RetrievedChunk` and `ReplyDraft`. Each is defined once per language (`internal/model`, `ml-service/app/schemas.py`) and kept in sync by tests. The full, dated decision log with the numbers behind each choice is in `docs/decisions.md`. This page is the per-layer summary.

## Layer 1: ingestion and preprocessing (`internal/ingest`, `internal/preprocess`, `internal/pipeline/poller.go`)

- **One interface, three adapters.** Every source implements `ingest.Source` (`Name`/`Fetch`/`Reply`), and `config/sources.yaml` enables them. The CSV source reads SCRIBE's format. The **email** source uses IMAP read-only (`BODY.PEEK`, so the user's flags never change) and replies over SMTP with threading headers. The **Zendesk** source uses the cursor-based incremental export and replies with a public comment. At most one source per type is enabled, so a reply goes back through the channel the ticket came from.
- **Checkpoints with overlap.** Each poll fetches from `checkpoint − 10 min` and moves the checkpoint to the newest fetched time, never to "now". Repeats are dropped by the `(source, external_id)` key.
- **A single cleaning chain** for every adapter: HTML→text, quote and signature stripping, NFC and whitespace normalisation. Then:
  - `content_hash` (sender + normalised subject + body) rejects the same message arriving twice through different channels;
  - non-English tickets are **stored as skipped**, not dropped, and routed to a human.
- Every step writes `audit_log` rows.

## Layer 2: classification (`ml-service/triage`, `POST /classify`, `internal/pipeline/classifier.go`)

- **Two tasks, not one six-way classifier.** The first fast tier learned writing *style*: the only human-written class was Spam. So spam is a separate **binary filter**, with negatives that include both ham and support tickets so that style isn't predictive. Its operating point holds the FPR on support tickets **and** on ham at ≤ 1%. A review band yields `possible_spam` instead of muting.
- **Topic cascade (5 classes).** A TF-IDF+LR fast tier (word and char n-grams, balanced weights, uncalibrated after calibration lost on val macro-F1) handles every ticket. Below τ, the fine-tuned DeBERTa-v3 deep tier decides. Without a deep tier, the ticket keeps the fast prediction with `low_confidence=true`.
- **Urgency** comes from an ordinal model (low/medium/high), with a documented regex heuristic as fallback. *critical* comes only from an explicit business-impact rule, because neither dataset has critical rows.
- **Selection discipline.** Choices are made on val and **ood-dev**, an out-of-domain set of human-written mail. Test and ood-test are scored once per final system, and every ood-test run is logged. Macro-F1 is the headline metric, reported overall, real-only and per origin.
- **Serving.** Models load lazily, at most one transformer per role, fp16 on MPS. The SCRIBE baseline never loads inside the service, because the dev machine has 8 GB of RAM.

## Layer 3: routing, SLAs and meetings (`internal/routing`, `internal/scheduler`, `config/*.yaml`)

- **Ordered YAML rules, first match wins,** validated strictly against the taxonomy and hot-reloaded by content hash (an invalid edit keeps the old rules). The rules read only the classification contract, so they work with whatever models are present.
  - Shipped order: skipped → human; Spam → mute; possible spam → review queue; **critical → P1 before no-reply** (so automated outage alerts are never muted); no-reply/newsletters → mute; low confidence → human; Meeting → schedule; Technical/Billing/General → reply; Feature → product.
- **SLAs start at routing time** (wall-clock P1 1h … P4 72h). Escalation raises the priority one step per breach, with a conditional update so concurrent runs can't escalate twice.
- **Meetings.** An LLM extracts the request (relative dates resolved against the received time; invalid values nulled). The scheduler books the slot only if it's free, inside working hours and ≥ 2 h ahead; otherwise it counter-offers the first free slot. It runs against Google Calendar (OAuth loopback + PKCE) or an in-memory fake.

## Layer 4: grounded replies (`ml-service/rag`, `internal/pipeline/replier.go`)

- **KB.** WixQA help-centre articles (MIT) are chunked by headings, with short sections packed into ~400-token chunks. They're embedded with bge-small and stored in pgvector (HNSW). Ingestion is idempotent per document hash, which includes the chunker version.
- **Retrieval.** Dense cosine top-5 with bge's query instruction. Hybrid (RRF with Postgres full-text) and a cross-encoder reranker were both measured and **rejected**, because both lowered recall@1.
- **Floor before tokens.** If the best cosine is below the **retrieval floor** (the 5% quantile over answerable questions), the ticket abstains without any LLM call.
- **Draft.**
  - The prompt answers only from numbered sources, with `[n]` citations, and never claims an action (such as a refund) unless a source says it happens.
  - A refine pass may change only clarity and tone; a refined draft that changes the citation set is discarded.
  - Invalid citations are stripped and reported.
- **Faithfulness gate.** One judge call (qwen, a different family from the drafter) splits the draft into atomic claims and checks each against its cited chunks. If the claim list would overflow the judge's output budget, the draft is judged in halves. Score = supported / all; below 0.9 it **abstains**, and the ticket moves to `human_review` with the unsupported claims shown.
- All LLM calls run at temperature 0 and are cached on disk by request hash, so evaluations resume across Groq's daily cap.

## Layer 5: review, sending and ops (`internal/api`, `internal/ui`, `internal/outbound`, `internal/monitoring`)

- **REST API** (OpenAPI in `docs/openapi.yaml`). One API key (constant-time compare), validated inputs, one JSON error shape, and every review action audited.
- **Review UI.** Plain JS embedded in the Go binary: no build step and no framework. Untrusted text goes only through `textContent`, under a strict CSP. Citations `[n]` open the exact chunk the drafter saw.
  - Approve, edit and reject are conditional state transitions (409 on conflict). Reject also reassigns the ticket to `human_review`.
  - Human approval is mandatory; an "eligible for auto-send" badge is informational only.
- **Exactly-once sending.**
  - A sender claims an approved draft (`FOR UPDATE SKIP LOCKED`), calls the ticket's own adapter with the idempotency key `draft-<id>`, and records the send and resolves the route in one transaction.
  - A claim older than its lease means the outcome is unknown. It is flagged for a human and **never resent**: at-most-once beats a duplicate customer email.
- **Monitoring.** The worker times every ml-service call through a client hook (one batched COPY per cycle). After each poll it recomputes today and yesterday into `metrics_daily`:
  - category PSI against the val distribution (alert > 0.2);
  - p50/p95 per stage, including ingest→classified and ingest→draft;
  - abstain rate;
  - SLA-breach rate.

  The dashboard shows them and flags drift.
- **Deploy.** The Render blueprint has the api (HTTPS, generated key), the worker, and the ml-service as a *private* service (it has no auth of its own). Models come from a hash-pinned bundle URL. See `docs/deploy.md`.

## Cross-cutting

- **Database safety:** migrations are up-only and additive. Tests run only on `TEST_DATABASE_URL`, in a schema per package. Demo data is tagged (`demo-*`) and is the only data `make demo-reset` can delete.
- **External services sit behind interfaces with fakes** (Groq, IMAP/SMTP, Zendesk, Google Calendar, the ml-service), so unit tests run offline.
- **Honest limits** (`docs/report_notes.md`):
  - wall-clock SLAs;
  - a single shared API key;
  - Zendesk ingests the first message only;
  - accepted counter-offers aren't auto-booked;
  - the style shortcut: models trained on LLM-written tickets transfer poorly to human mail, which the OOD set measures.

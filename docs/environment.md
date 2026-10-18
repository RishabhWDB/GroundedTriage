# Environment and external services

Read when a DB connection, external service, model download or the machine itself misbehaves, or before touching Supabase, Groq, Google, Zendesk, IMAP/SMTP or Render config. Rules that apply every session are in CLAUDE.md; this is the detail.

## Dev machine

- M1 MacBook Air, **8 GB RAM**, macOS arm64, no Docker. Swap is routinely several GB during model work; tail latency is swap-bound.
- **2026-10-17 21:23 out-of-memory reboot.** The P7.2 run (resumed 21:11: bge + DB + Groq) was running and the P7.1 `scribe-tune` code (BART-MNLI NLI pass, all logits held in memory, no checkpoint) had just been written; the likely cause is the two overlapping. Recovery check: every cache/JSONL/npz/parquet parsed; nothing was lost. This led to the memory rules, `scripts/jobguard.py` and `app/fileio.py`.
- After a reboot, Spotlight re-indexing keeps memory pressure at "warn" for a while, so the guard refuses jobs until it settles.

## Supabase (Postgres + pgvector)

- Project in **ap-south-1 (Mumbai)**. Use the **Session pooler** string (IPv4, port 5432); the Transaction pooler (6543) breaks prepared statements in pgx/psycopg.
- pgvector 0.8.2 is pre-installed in the **`extensions`** schema; any connection overriding `search_path` must include it, or you get `type "vector" does not exist`.
- Free-tier projects **pause after inactivity**: if connections fail, check whether it needs resuming in the dashboard.
- The pooler can **silently drop idle connections** during long runs; Python connections use keepalives + timeouts + one reconnect (decisions.md §3).
- Latency (`cmd/dbping`, 2026-10-15): warm-pool `SELECT 1` p50 28–30 ms, p95 31 ms (Sydney was 395 ms); TCP connect 31 ms; fresh connection ~212 ms. The integration suite takes ~9–15 s; each package pays a few seconds of schema setup.
- Free-tier storage 500 MB; the KB uses ~66 MB.

## Groq

- Drafter `openai/gpt-oss-120b`; judge `qwen/qwen3.8-27b`. Models change: `llama-3.3-70b-versatile` isn't on this key. `GET {GROQ_BASE_URL}/models` lists the live ones.
- Limits on this key: 1,000 requests/day, 8,000 tokens/minute, **200k tokens/day rolling for gpt-oss-120b (shared by every gpt-oss job)**; qwen 1,000 output tokens/minute. 429 bodies name the limit ("tokens per day (TPD)" vs per-minute).

## Hugging Face

- The baseline's first run downloads ~2 GB to `~/.cache/huggingface`; bge-small ~130 MB. The "unauthenticated requests" warning is harmless (`HF_TOKEN` speeds downloads).
- transformers 5 prints an UNEXPECTED `pooler` keys report for twitter-roberta; expected (the classification head doesn't use the pooler).
- Installed: torch 2.14.1, transformers 5.19.0, MPS available.

## Kaggle

- New-style `KGAT_…` token in `KAGGLE_API_TOKEN`. Kaggle answers 404 to the token check for an invalid token and the SDK reports "Authentication required"; `python -m data.kaggle_auth` names the likely cause.

## External services not tested against real accounts

- **Google Calendar:** tested only against `httptest` (FreeBusy, events.insert, the whole OAuth loopback flow). Real use needs a Google Cloud project with the Calendar API, a Desktop-app OAuth client saved as `credentials.json`, and `make calendar-auth` once. A test-mode consent screen only allows listed test users, and refresh tokens expire after 7 days.
- **Zendesk:** tested only against an `httptest` mock; the incremental export needs an admin API token.
- **Email:** `IMAPClient` untested beyond compiling; `SMTPMailer` tested against an in-process fake (no TLS, no auth). Gmail needs an app password and IMAP enabled.
- **Docker** (`docker-compose.yml`, `deploy/*.Dockerfile`): not verified (daemon not running).
- **Render:** blueprint validated (`make render-validate`), never deployed. Check that Render's native Go runtime supports `go 1.26.0` or pin a toolchain (`docs/deploy.md`).

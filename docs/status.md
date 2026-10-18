# Project status

Read at the start of a session (after `git log --oneline`) and whenever you finish a part. Keep it current: one line per finished part, the manual checklist up to date.

## Done

Each finished part is one commit tagged `[P<phase>.<part>]`; `git log --oneline` has the details. A Done entry's hash is backfilled in the next part's commit (no amending).

- **Phase 0** scaffold, DB + migrations, ml-service skeleton, Go contracts/mlclient, LLM roles: P0.1, P0.1a, P0.2, P0.2a, P0.3a, P0.3b, P0.3c.
- **Phase 1** ingestion: source registry, CSV, email parsing + IMAP/SMTP, Zendesk, normalize/dedupe, worker + checkpoints, review fixes: P1.1–P1.7.
- **Phase 2** data + baseline: Kaggle auth, taxonomy, Tobi-Bueck, SpamAssassin, synthetic meetings, splits/card/export, audit, group-aware re-split, stress set, SCRIBE baseline + eval: P2.1a–i, P2.2, P2.2a, P2.3, P2.3a.
- **Phase 3** models: fast tier, deep-tier + urgency loaders, cascade + τ tuning, Go classify step, OOD check, spam filter (data, two-task rebuild, filter, cascade wiring, blind-spot fix), style-aug generator + first batch, OOD set + gate, retrain, embedding fast tier: P3.1–P3.14, P3.8a.
- **Phase 4** routing: rule engine, no-reply, SLA escalation, meeting extraction/scheduling/Google Calendar/meeting step: P4.1–P4.3, P4.4a–d.
- **Phase 5** RAG: KB ingestion, retriever, generator, faithfulness gate, Go reply step, reranker (rejected): P5.1–P5.6.
- **Phase 6** review: queries, REST API, UI, outbound, dashboard, review follow-ups: P6.1a, P6.1b, P6.2–P6.5.
- **Phase 7 so far:** P7.1a val-tuned SCRIBE wording (`make scribe-tune`, not run yet), P7.2a RAG eval harness, P7.2b judge-labelling CLI, P7.2c dropped-DB-connection fix, P7.2d Groq 429 message fix, P7.2e crash-safe writes + one-job memory guard (after the OOM reboot), P7.3 monitoring.
- **Phase 8 so far:** P8.1 demo set + load/status/reset, P8.2 Render deploy + model bundles, P8.3a docs skeleton (finalised after P7.4), P8.3b CLAUDE.md split into `docs/`.

## In progress

- Phase 7 in Arya's order: P7.2 (RAG eval, multi-day at the free cap) → P7.3 monitoring (done) → P8 parts not tied to final numbers (done) → stop with the consolidated manual checklist. P7.1 / P7.4 wait for Arya's artifacts and labels.
- **Paused after the 2026-10-17 OOM reboot, awaiting Arya's go-ahead** to resume P7.2 (`make rag-eval`) or run `make scribe-tune`, one job at a time. Recovery and scribe-tune are committed (P7.2e, P7.1a). Uncommitted:
  - `ml-service/eval/compare.py` (P7.1 comparison harness): lint-clean, untested, no make targets yet (`compare`, `compare-final`, `report` planned);
  - `results/rag/{metrics.json,report.md}` from the day-1 stop (23 of 556 rows; PRELIMINARY, rewritten by each run).
- P7.2 day-1 status: stopped at the rolling daily cap after 23 of 556 rows (style-aug had used the earlier part of the 24 h window); resumed 21:11 IST, then the reboot.

## To do

- P3.8b Finish `style_aug.jsonl` (daily `make style-aug`), then `make data && make export-splits`.
- P7.1 Classification evaluation: baseline vs proposed (incl. "SCRIBE (val-tuned descriptions)", see decisions.md → Evaluation).
- P7.2 RAG evaluation (finish the run, judge sample + labels + κ).
- P7.4 Results report (`results/REPORT.md`; see report_notes.md).
- P8.3 Finalise docs with the P7.4 numbers.

**Claude's background item:** `make rag-eval` once per day until all 556 rows are done (P7.2 has priority), then `make style-aug` once per day (Groq free cap) until 3,978 rows, then `make data && make export-splits` (P3.8b). One job at a time (memory rules).

## Blocked / needs input

- none.

## Manual work pending (Arya)

Keep this list current: add items as they arise, remove them when done.

| # | Item | Where | After it, run |
|---|---|---|---|
| 1 | Review the 149 Twitter OOD rows: fix `category` (`None` drops a row), set `needs_review=false`, notes optional. Don't touch `ood_split`. | `datasets/ood/twitter.csv` (guide: `datasets/ood/README.md`) | `make ood-eval` (ood-dev; choices only). `make ood-eval-test` is only for final systems |
| 2 | Write ~40 support-style emails (informal ones and meeting requests included); any unique `id`, `category` one of the 6 | `datasets/ood/team_written.csv` | `make ood-eval` |
| 3 | Review the 100-row Meeting stress set: fix `category`, set `needs_review=false` | `datasets/stress/meeting_stress.csv` | `make baseline-stress` (SCRIBE on the stress set; the proposed system is scored in P7) |
| 4 | Check the ood-dev rows the default cascade muted but whose label isn't Spam (7 Twitter rows at P3.13, 2 after P3.14). If a row really is spam, fix its label in item 1; thresholds stay as they are (decided in Phase 7) | `results/ood/muted_review.md` | `make ood-eval` |
| 5 | Spot-check the spam rows: the 5 Twitter rows pre-labelled Spam (part of item 1) and the 20 Enron rows (corpus labels) | `datasets/ood/twitter.csv`, `datasets/ood/enron_spam.csv` | `make ood-eval` |
| 6 | PC training: deep tier (5 topic classes; once with `train` and once with `train ∪ train_style_aug`, recording `style_aug` in `model_card.json`) and the urgency model (`train_urgency`). Contract in `docs/model_artifacts.md` | export `datasets/export/groundedtriage-splits-e2552b04338b.tar.gz`. It's regenerated with the same tag once style-aug finishes; take the newest file | — |
| 7 | Drop in artifacts: copy to `models/deep_tier/` and `models/urgency/` (a second deep tier, e.g. the aug variant, can live in another folder for comparison) | `models/` | `make setup-models` → `make ml` and `curl localhost:8001/health` (models_loaded true; reasons in the log) → `make urgency-eval` → `make tune-tau` → `make ood-eval`. Restart the ml-service after copying weights |
| 8 | Google Calendar for the meeting scheduler (optional; the worker runs without it): in Google Cloud, enable the Calendar API, create an OAuth client of type **Desktop app**, save its JSON as `credentials.json` (gitignored), add yourself as a test user, set `GOOGLE_TIMEZONE` | repo root, `.env` | `make calendar-auth` → `token.json`; the worker then uses Google automatically (or set `MEETING_CALENDAR=google`). Test-mode tokens expire after 7 days, so re-run it then |
| 9 | **Set a real `API_KEY` in `.env`** (any long random string, e.g. `openssl rand -hex 24`). It's still the `.env.example` placeholder, and `make api` refuses to start with it | `.env` | `make api` (the UI asks for the same key) |
| 10 | **Hand-label judge claims** (judge validation, P7.2): for each claim, answer whether the shown sources support it (y/n/u). The judge's verdicts are hidden. About 80 claims, ~30–40 min; resumable (q quits). Ready once Claude has run `make judge-sample` on the first day of P7.2 rows | `datasets/judge_labels/claims.csv` (via the CLI) | `make judge-label` → `make judge-score ARGS=--with-llm` (κ for qwen and gpt-oss-120b; gate-judge κ < 0.6 → revisit the judge) |
| 11 | Demo day: `make demo` (loads demo rows, runs the worker once — this also processes the 21 older sample tickets in DATABASE_URL and spends ~5k gpt-oss tokens per draft) | DATABASE_URL | `make demo-status`; `make demo-reset` afterwards if wanted |
| 12 | Deploy to Render (your account) following `docs/deploy.md` | Render | — |

## Deferred decisions (decide in Phase 7)

- **Deep-tier acceptance:** it must beat SCRIBE on **reviewed ood-dev with ≥ 60 rows**, reported with **bootstrap CIs**. The test split and ood-test are reported once afterwards.
- **Whether the fast tier stays in the cascade** at all (OOD may show it adds nothing over the deep tier).
- **Which fast tier:** TF-IDF + LR vs bge-small embeddings + LR (P3.13), each with and without style-aug.
- **Prefit calibration** (calibrate on half of val, tune τ on the other half): skipped unless the deep tier needs it.
- **Hybrid fallback option:** if the deep tier doesn't beat SCRIBE on reviewed ood-dev, evaluate a hybrid where low-confidence tickets fall back to zero-shot (SCRIBE-style BART-MNLI) or an LLM classifier. Decide with the reviewed OOD data.
- **Spam thresholds**, after Arya's muted-row review: P3.14 traded Enron recall for fewer muted informal tickets (results.md).
- **Judge choice:** qwen (reasoning off) until P7.2's κ; κ < 0.6 → revisit.
- Already chosen on val + ood-dev, revisitable then: spam rule `support_and_ham`; fast tier uncalibrated; plain (not style-aug) fast tier; τ = 0.7 untuned until the deep tier exists; retrieval floor 0.7177.
- Nothing in the worker/routing/review code may depend on these decisions (it reads only the contracts).

## Model artifacts

Arya fine-tunes the deep tier and urgency model on a separate GPU PC; this repo only loads them (contract: `docs/model_artifacts.md`).

| Dir | Status | Base model | Trained at | Val metrics |
|---|---|---|---|---|
| `models/deep_tier/` | absent | — | — | — |
| `models/urgency/` | absent | — | — | — |
| `models/fast_tier/` | present (`fast-20261017043430-9848709e`, `make fast-tier`) | 5 topic classes; TF-IDF word 1–2 + char_wb 3–5, LR C=64 balanced, uncalibrated | in repo, 2026-10-17 | macro-F1 0.823, acc 0.795, ECE 0.070 |
| `models/fast_tier_aug/` | present (`fast-20261017044330-9848709e`, ablation) | as above + 589 style-aug rows | in repo, 2026-10-17 | macro-F1 0.823, acc 0.795, ECE 0.077 |
| `models/fast_tier_emb/`, `models/fast_tier_emb_aug/` | present (comparison only) | bge-small-en-v1.5 CLS embeddings + LR C=16, uncalibrated (+589 aug rows) | in repo, 2026-10-17 | macro-F1 0.714 / 0.712, ECE 0.018 |
| `models/spam_filter/` | present (`spam-20261017045337-319c633f`, `make spam-filter`) | binary TF-IDF + LR C=16 balanced; rules `support_and_ham` (default) / `support` | in repo, 2026-10-17 | AP 0.996; see results.md |

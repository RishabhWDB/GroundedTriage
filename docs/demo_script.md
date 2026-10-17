# Five-minute demo script

For the review panel. The demo uses the 25 tickets in `datasets/demo/demo_tickets.csv`: Wix help-centre questions written as emails, plus out-of-KB, meeting, spam and billing tickets. Each row lists the outcome expected for it.

## Before the panel (10 min, once)

```bash
make demo-reset          # only if an earlier demo run left demo-* rows (asks to type "reset demo")
make demo                # loads the tickets, starts the ml-service, runs the worker once,
                         # prints expected vs actual, then starts api + worker + ml (make dev)
```

- **Token budget.** Drafting uses Groq's gpt-oss-120b daily cap: about 5k tokens per grounded draft, about 17 drafts. Don't run it on a day the P7.2 evaluation has used up the cap. Check with `make rag-eval-plan` / the last `results/rag/metrics.json`.
- **Meetings.** `MEETING_CALENDAR=fake` (the `make demo` default) books into an in-memory calendar. Use `google` after `make calendar-auth` for real invites.
- **Before the panel arrives:**
  - open http://127.0.0.1:8080/ and enter `API_KEY` and a reviewer name;
  - keep a terminal open with `make demo-status` output.

## The walkthrough

**0:00 — The problem (30 s).** SCRIBE classifies support mail with regex and zero-shot NLI, and drafts replies with an LLM that has no sources. Here is the failure we fix: `results/examples/ungrounded_hallucination_01.md`, where the baseline promises a refund it can't issue.

**0:30 — Ingestion and triage (1 min).** Open the **Inbox**: tickets are ordered by priority, then by SLA.
- Point at the **tier** column (`fast`, `deep` or `spam_filter`) and the urgency badge (*model* or *heuristic*).
- Spam (demo-22, demo-23) is **muted**, and the reason is visible on the route card.
- Sources are pluggable: the same pipeline takes IMAP, Zendesk or CSV (*Upload CSV* button).

**1:30 — A grounded reply (1.5 min).** Open **demo-07** (product page redesign) or **demo-04** (deleting CMS items).
- The classification card shows the category, confidence and tier. The route card shows the queue, priority and SLA deadline.
- The **draft** shows each sentence ending in `[n]`. Click `[1]`: the side panel shows the exact help-centre chunk (title, heading, URL).
- Show the **faithfulness** score and the claim list: every claim is supported. Point out the "eligible for auto-send" badge, and that a human still approves.
- *Edit* a word → *Save* → *Approve*. The worker sends it exactly once through the ticket's own channel (CSV here: `datasets/out/replies.csv`).

**3:00 — Abstention (1 min).** Open **demo-16** (VPN drops) or **demo-18** (Outlook password).
- The Wix KB doesn't cover these, so the best retrieval score is under the floor. The system **abstained without spending tokens**, and the ticket moved to `human_review` with the reason shown.
- If demo-24 (duplicate charge) was drafted: its claims about refunds aren't in the KB, so the judge marks them unsupported and the gate abstains. That's exactly the SCRIBE failure, caught.

**4:00 — Meetings and the dashboard (45 s).**
- Open **demo-20**: "Thursday at 3pm, 30 min, Zoom" was extracted and booked, or a counter-offer was proposed if the slot was taken.
- Open the **Dashboard**: volume by source and category, urgency, tier share, abstain rate, average faithfulness, and the **monitoring table** (daily PSI drift against the training distribution, p95 latency per stage, SLA-breach rate).

**4:45 — Evaluation (15 s).** The baseline-vs-proposed numbers are in `results/REPORT.md` (P7.4): classification on test and on human-written OOD mail, and grounded vs ungrounded faithfulness and hallucination rates with a validated judge.

## If something goes wrong

| Symptom | Fix |
|---|---|
| Login says the key was rejected | Use the `API_KEY` from `.env` (not `change-me`) |
| Tickets stay unclassified | `make ml` isn't running, or `models/fast_tier/` is missing: `curl localhost:8001/health` |
| Drafts fail with 429 / "tokens per day" | Groq daily cap; demo the abstain path and the stored drafts, or rerun tomorrow |
| DB connection errors | The Supabase free project paused; resume it in the dashboard |
| Start over | `make demo-reset` (only `demo-*` rows), then `make demo` |

# Report notes: what results/REPORT.md must say

Read before writing or editing `results/REPORT.md`, the README results section, or any slide/claim about the evaluation. Each item was agreed in a review; don't drop one without asking Arya.

## Must be stated

- **Taxonomy deviation:** SCRIBE's original 6 email categories (Work, Spam, Personal, Urgent, General, Meeting Scheduling), which BUILDPROMPT extended to 11 with support categories, became 5 topic classes (Technical Issue, Billing, Feature Request, General Inquiry, Meeting Scheduling) plus Spam from a separate binary filter (Work, Personal, Urgent, Complaint, Account/Access dropped or folded; decisions.md §1). SCRIBE is evaluated on these same 6 labels via `taxonomy.yaml` zero-shot descriptions; its regex rules cover Spam and Meeting Scheduling.
- **LLM deviation:** gpt-oss-120b replaces LLaMA-3-70B (unavailable) for both SCRIBE's draft→refine and our generator, which isolates the grounding effect.
- **Two baselines:** SCRIBE as is (faithful, untuned) and "SCRIBE (val-tuned descriptions)", labelled a *strengthened baseline*.
- **Zero-shot collapse:** on real support tickets BART-MNLI sends most tickets to General Inquiry, contrary to SCRIBE's paper (results.md, SUPERSEDED section, re-measured in P7.1).
- **The style shortcut** (P3.5a/P3.12): models trained on LLM-written tickets transfer poorly to human-written mail (Spam = "human English" in the 6-class model; Technical Issue after the split). Reported via the OOD set as a headline next to test.
- **Synthetic data caveats:** Meeting Scheduling is synthetic (shorter, keyword-heavy subjects → likely easier); every number is reported overall, real-only and per origin; macro-F1 is the headline.
- **Spam corpus provenance:** SpamAssassin (cite the Apache SpamAssassin project; 2002–2005 era; not redistributed); Enron OOD rows from a CC0 copy; Twitter rows CC BY-NC-SA 4.0 with attribution.
- **Selection discipline:** choices on val + ood-dev; test and ood-test reported once per final system (the ood-test log shows every look).
- **Judge validation:** κ of qwen and gpt-oss-120b vs Arya's labels; PRELIMINARY below 60 labels.
- **RAG metrics** are a RAGAS-style equivalent, not the `ragas` package; ungrounded replies are judged against the same retrieved chunks.
- **Latency:** only the P7.1 single-session measurements (other apps closed, models warmed, RSS and swap recorded). Never the P2.3 swap-bound numbers.
- **Negative result:** the cross-encoder reranker lowered WixQA recall@1 0.43 → 0.36 (expert) and 0.34 → 0.29 (simulated) and tripled latency (p50 154 → 457 ms).

## Future work and limitations

- **Accepted counter-offers aren't booked automatically** (Phase 4 review Q1): needs reply threading and a re-extraction.
- **SLAs are wall-clock** (Phase 4 review Q2): no business-hours or holiday calendar.
- **Groq free tier:** gpt-oss-120b is capped at 200k tokens/day (rolling); style-aug and P7.2 take several days, so their runs are resumable and cached.
- **Auto-send policies** (e.g. grounded, faithfulness 1.0, high retrieval score) are future work. Human approval is mandatory; the UI only shows an "eligible for auto-send" badge.
- **User accounts / RBAC** are future work; the capstone uses one shared API key.
- **Deploy security:** HTTPS only; the API key is never in page source or the JS bundle (the agent types it; it lives in that browser's localStorage); rotate it by changing `API_KEY` in the environment (documented from `.env.example`).
- **Zendesk ingests the first message only**; follow-up comments aren't ingested.
- **The KB is per-deployment content**; the demo uses the Wix help centre (any docs folder can be ingested with `rag.kb --folder`).
- **Untested against real accounts:** Google Calendar, Zendesk, IMAP/SMTP (httptest/fakes only), Docker files (daemon not running), the Render deploy (validated, not deployed).

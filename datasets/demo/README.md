# Demo tickets (P8.1)

`demo_tickets.csv` holds 25 hand-written support tickets for the live demo.
They're in the Wix help-centre domain, because the KB is WixQA.

| Rows | Kind | Source of the content |
|---|---|---|
| demo-01 … demo-15 | `wixqa` | The 20-question **`demo` split** of `eval/wixqa_splits.py` (expert-written WixQA questions; P7.2 never evaluates on them), rewritten as customer emails with the facts kept and the tone varied. `WixQA_Id` points at the question. |
| demo-16 … demo-19 | `out_of_kb` | Generic IT/warranty tickets the Wix KB doesn't cover (abstention). |
| demo-20, demo-21 | `meeting` | Meeting requests with a concrete time (scheduler). |
| demo-22, demo-23 | `spam` | Lottery and pharmacy spam (spam filter → mute). |
| demo-24, demo-25 | `billing` | Billing requests outside the KB (duplicate charge, VAT invoices). |

- **Columns.** `Id, Sender, Subject, Body, Label` are what the CSV adapter reads; it ignores the rest. `Label` is the gold category.
- **`Retrieval_Max_Score` and `Expected_Outcome`** are measured by `make demo-expect`. That command makes no LLM calls; it reads the live KB on DATABASE_URL.
  - Spam → `mute`.
  - Meeting → `schedule`.
  - Otherwise `abstain` when the best top-5 cosine is below the gate's floor (0.7177).
  - Above the floor: `ground` for the WixQA rows, and `draft (gate decides)` for the others, because the faithfulness judge decides those live.
- **Measured on 2026-10-17:**
  - All 15 WixQA rows clear the floor, though 4 of them don't have their gold article in the top 5 (demo-05, 08, 09, 10).
  - The 4 out-of-KB rows score between 0.588 and 0.672, so they abstain.
  - demo-24 (duplicate charge) scores 0.7415, just above the floor, so it is drafted and the judge decides.
  - demo-25 scores 0.7157, just below the floor, so it abstains.
- **The actual outcome also depends on the classifier** (the style shortcut, see CLAUDE.md §5). `make demo-status` prints expected against actual.
- **Lifecycle:**
  - `make demo` loads the rows. It's additive, and any row without a `demo-` Id is refused.
  - Then run `go run ./cmd/worker -once` (with `make ml` running).
  - `make demo-status` shows expected against actual.
  - `make demo-reset` deletes **only** CSV tickets whose Id starts with `demo-`, plus their classifications, routes, drafts, meetings and audit rows. It asks you to type `reset demo` first. The other rows in the database are never touched.

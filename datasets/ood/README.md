# OOD evaluation set (human-written, never trained on)

Built by `make ood-set` (`ml-service/data/ood.py`). **Required gate before any model is
accepted**; reported next to test in P7.

**Two halves.** Every row is in `ood-dev` (40%) or `ood-test` (60%), column `ood_split`
(stratified by source × category with a seeded hash order; stored in the CSVs so your
relabelling never moves a row). Variants, thresholds and rules are chosen on val + ood-dev
only (`make ood-eval`); ood-test is reported once per final system (`make ood-eval-test`,
every run logged in `results/ood/test_log.jsonl`). Team-written rows get their half from a
per-id hash; the 22 sample emails are split at load time (their labels are fixed).

**Unreviewed rows** (`needs_review=true`) carry LLM pre-labels. Reports show a table on
reviewed rows and a separate one marked **PRELIMINARY** that includes them.

| file | rows | labels | what to do |
|---|---|---|---|
| `twitter.csv` | 149 | LLM pre-labels (qwen, no reasoning), `needs_review=true` | **Review every row:** fix `category`, set `needs_review=false`, add `reviewer_notes` if useful. To drop a row that fits no class, set `category` to `None` (it's then never scored). |
| `enron_spam.csv` | 20 | corpus labels (Spam), `needs_review=false` | nothing (optional spot check) |
| `team_written.csv` | 0 | yours | **Add ~40 support-style emails** written by you and your team: `id` (any unique string, e.g. `team-001`), `author`, `subject`, `body`, `category` (one of the six below), `notes`. Include informal ones and meeting requests. |
| `../sample_emails.csv` | 22 | hand-written (P1.2) | added automatically |

Categories: `Technical Issue`, `Billing`, `Feature Request`, `General Inquiry`,
`Meeting Scheduling`, `Spam`.

`cd ml-service && .venv/bin/python -m data.ood --force` rebuilds `twitter.csv` and
**overwrites your review**; `make ood-set` (no `--force`) refuses when the file exists. The pre-labels are cached, so a rebuild is free and deterministic.

## Sources and licences

- **Customer Support on Twitter** (Kaggle `thoughtvector/customer-support-on-twitter`,
  ThoughtVector), CC BY-NC-SA 4.0. Inbound, thread-opening tweets; `@mentions` removed
  (the first one is kept as `brand`); English, ≥ 60 characters, deduplicated; a seeded
  600-tweet candidate pool (seed 4242) was pre-labelled and picked by per-class quota
  (Technical 45, Billing 40, General 40, Feature 20, Meeting 10, Spam 10; Feature,
  Meeting and Spam are rare on Twitter and take what exists: 18, 1, 5). Used for
  non-commercial evaluation with attribution under the same licence.
- **Enron-Spam** (Metsis, Androutsopoulos & Paliouras, 2006), `enron1` spam, via the CC0
  Kaggle copy `venky73/spam-mails-dataset`. Text is the corpus's preprocessed form
  (lower-cased, tokenised). A different corpus from the SpamAssassin data the spam filter
  is trained on, to test generalisation.

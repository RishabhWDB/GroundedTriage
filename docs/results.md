# Results so far

Read before writing `results/REPORT.md` (P7.4), when comparing a new run against earlier numbers, or before making a Phase 7 decision. The raw outputs are in `results/<area>/`. Numbers are val unless stated; test is untouched except where marked SUPERSEDED.

## Classification

- **Fast tier (P3.12, val, 5 topic classes, uncalibrated LR C=64):** macro-F1 **0.823**, accuracy 0.795, ECE 0.070. Real only (no synthetic Meeting): macro-F1 0.778, accuracy 0.791. Per-class F1: Technical 0.842, Billing 0.738, Feature 0.961, General 0.573, Meeting 1.000.
  - Calibrated variants: 0.703 (sigmoid), 0.687 (isotonic).
  - Style-aug ablation (+589 rows): macro-F1 0.823, accuracy 0.795, ECE 0.077.
  - Coverage at τ = 0.7: 82% of val at 0.860 accuracy.
  - Before snapshot (P3.1, 6-class, superseded split): macro-F1 0.825 / real-only 0.790; not comparable.
- **Embedding vs TF-IDF fast tier (P3.13):**
  - Embedding + LR: macro-F1 **0.7141**, accuracy 0.6463, real-only 0.6447, **ECE 0.0182**; with 589 style-aug rows 0.7121 / 0.6477.
  - Per-class F1 (embedding vs TF-IDF): Technical 0.7113 vs 0.842; Billing 0.5422 vs 0.738; Feature 0.9046 vs 0.961; General 0.4206 vs 0.573; Meeting 0.9917 vs 1.0.
  - Cost (M1, ~8 GB swap in use): embedder load 14.69 s; single ticket p50 12.2 ms (p95 67.2); batched 11.4 ms/ticket; peak RSS 470.7 MB. TF-IDF: p50 1.0 ms, artifact 2.3 MB. First embedding of 19.5k texts: 217.8 s.
  - ood-dev reviewed (17 rows): all four fast variants 2/9 on sample emails; every cascade[support_and_ham] variant 9/17. PRELIMINARY (76 rows): fast alone TF-IDF 20 / +aug 25 / emb 24 / emb+aug 26; cascade 26 / 30 / 29 / 31; SCRIBE 40.
- **Urgency heuristic (P3.12, val):** macro-F1 0.269, accuracy 0.410, MAE 0.625. Earlier split: 0.273 / 0.419 / 0.615, off by ≥ 2 on 3.2%; predicts medium for 84% (a conservative fallback, not a classifier); 74 predicted critical (gold has none, all errors); fired groups: urgency 551, impact 215, deadline 88, low 17, escalation 10, negative 2. For reference (different split): SCRIBE sentiment-as-urgency macro-F1 0.357 / MAE 0.750.
- **Data diagnostics:** topic TF-IDF+LR on 5-class val macro-F1 0.777 / accuracy 0.728 (audit model, not the fast tier). The group-aware split removed leakage that had inflated 6-class val macro-F1 0.825 → 0.802.

## Spam filter

- **P3.9 (`spam-20261017022247-319c633f`):** AP 0.9929 / ROC-AUC 0.9996 on spam_test (val 0.9958 / 0.9998).
- **Before → after P3.14 (spam_test, n=4,388; thresholds re-derived on spam_val by the same rules):**

  | rule | mute threshold | spam recall (mute / +review) | support FPR | ham FPR | ham in review | AP |
  |---|---|---|---|---|---|---|
  | `support_and_ham` (default), before | 0.4365 | 0.978 / 0.996 | 0.000 | 0.021 | 0.042 | 0.993 |
  | `support_and_ham` (default), after | 0.5655 | 0.978 / 0.996 | 0.000 | 0.020 | 0.034 | 0.990 |
  | `support`, before | 0.0195 | 1.000 / 1.000 | 0.0099 | 0.210 | 0.259 | |
  | `support`, after | 0.0326 | 1.000 / 1.000 | 0.0096 | 0.346 | 0.336 | |

  Review thresholds: `support_and_ham` 0.0941 → 0.1985; `support` before 0.0052.
  - ood-dev, default cascade: muted Twitter rows (preliminary labels) **7 → 2** (still muted: tw-1576489 at 0.58, tw-2122609 at 0.83). Reviewed Enron spam caught **7/8 → 4/8**; reviewed sample emails muted 0/9 → 0/9.

## Out-of-domain

- **ood-dev (P3.12, `results/ood/dev/`; reviewed = 9 sample emails + 8 Enron spam):**

  | system | reviewed correct | sample emails | Enron spam caught | reviewed non-spam muted | PRELIMINARY Twitter correct (59, LLM labels) |
  |---|---|---|---|---|---|
  | SCRIBE | 8/17 | 7/9 | 1/8 | 0/9 | 32/59 |
  | fast tier alone | 2/17 | 2/9 | 0/8 | 0/9 | 18/59 |
  | cascade, `support_and_ham` (default) | 9/17 | 2/9 | 7/8 | 0/9 | 17/59 (7 muted) |
  | cascade, `support` | 9/17 | 1/9 | 8/8 | **6/9** | 2/59 (52 muted) |
  | fast tier + aug (589) alone | 2/17 | 2/9 | 0/8 | 0/9 | 23/59 |

  ood-test not run (no final system yet).
- **Before snapshot (P3.5a, 6-class fast tier, `results/ood/sample_emails.md`, n=22, not a metric):** fast tier / cascade 6/22 with 13 predicted Spam (gold Spam = 1); SCRIBE 19/22 with 1 Spam.

## Retrieval (P5.2, `results/retrieval/`, document-level, 6,221 documents / 13,220 chunks)

| retriever | questions | recall@1 | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|
| dense | expert-written (200) | 0.43 | 0.645 | 0.77 | 0.561 |
| dense | simulated (200) | 0.34 | 0.535 | 0.675 | 0.471 |
| hybrid | expert-written | 0.4 | 0.61 | 0.75 | 0.5317 |
| hybrid | simulated | 0.28 | 0.5 | 0.64 | 0.4276 |
| dense + rerank (rejected) | expert-written | 0.36 | 0.61 | 0.75 | 0.5124 |
| dense + rerank (rejected) | simulated | 0.29 | 0.52 | 0.625 | 0.4235 |

- Latency (embed + SQL, Mac → Mumbai): dense p50 144 ms / p95 152 ms (first query with embedder load 5.9 s); hybrid p50 184 / p95 216 ms; rerank p50 457 ms (vs 154), model load 18.55 s.
- Best top-5 cosine: answerable WixQA median 0.8004 (5% quantile 0.7177 = the floor); 400 Tobi-Bueck val tickets (out of KB) median 0.6703 (95% quantile 0.7397).

## RAG and generation

- Live smoke check (P5.4): 2 WixQA expert questions retrieved their gold article (0.867 / 0.846) and drafted grounded replies citing [1] / [1][2] with faithfulness 1.0 (4 and 5 claims), judge at the old "medium" setting. **The reasoning-off judge isn't verified live yet** (leftover-budget item).
- Ungrounded hallucination example: `results/examples/ungrounded_hallucination_01.md` (SCRIBE told a customer "I've processed a refund … within 3–5 business days" and left `[Your Name]` placeholders).
- P7.2 run: 23 of 556 rows so far; `results/rag/` is PRELIMINARY.

## SUPERSEDED: SCRIBE baseline on the old test split

P2.3, `results/baseline/` (marked SUPERSEDED), test content `c6539fff…`, n=3,775. Re-run on the current test in P7.1. Kept for the zero-shot-collapse evidence; **latencies were taken under heavy swap: never compare them**.

- Overall macro-F1 **0.522**, accuracy 0.527; real only (5 classes) 0.479 / 0.521; synthetic Meeting recall 0.917.
- Per class F1: Technical 0.657 (P 0.83 / R 0.55), Billing 0.444 (R 0.32), Feature 0.573, General 0.368 (P 0.25 / R 0.73), Spam 0.349 (P 0.96 / R 0.21), Meeting 0.738 (P 0.62 / R 0.92).
- **Zero-shot collapses into General Inquiry:** 1,818 of 3,712 zero-shot predictions (41% of Technical, 52% of Billing, 43% of Feature, 53% of Spam rows). SCRIBE's paper reports zero-shot BART beating fine-tuned BERT; on real support tickets it doesn't.
- Rules decided 1.67% (63: 48 Spam, 15 Meeting), all correct; zero-shot accuracy 0.519 on the rest. 34 real tickets predicted Meeting; real meeting-word tickets (n=196): accuracy 0.418, 27 predicted Meeting.
- Sentiment-as-urgency (n=3,485): macro-F1 0.357, accuracy 0.397, MAE 0.750.
- Stress set: pending Arya's review.

# Model artifacts: export, train on the PC, copy back

The deep tier (DeBERTa-v3 category classifier) and the urgency model are
fine-tuned **outside this repo** on the GPU PC. This repo only exports the
training data and loads the finished artifacts. This page is the contract for
both directions.

## 1. Export the training data (Mac → PC)

```bash
make data            # builds datasets/processed/{train,val,test}.parquet + manifest + card
make export-splits   # -> datasets/export/groundedtriage-splits-<hash>.tar.gz
```

The archive contains:

| path | what |
|---|---|
| `processed/train.parquet`, `val.parquet`, `test.parquet` | the category splits (`train` is capped, val/test are not) |
| `processed/train_urgency.parquet` | the urgency model's train file: every uncapped train row with an urgency label |
| `processed/manifest.json` | seed, fractions, train cap, per-split row counts and SHA-256s |
| `config/taxonomy.yaml` | the 6 categories (canonical order) and the urgency scale |
| `datasets/README.md` | dataset card: sources, licences, build steps, counts |
| `docs/model_artifacts.md` | this file |

### Split columns

| column | type | notes |
|---|---|---|
| `id` | str | unique across splits |
| `source_dataset` | str | `tobi_bueck`, `spamassassin`, `synthetic_meetings` |
| `origin` | str | `real` or `synthetic`; report metrics overall and per origin |
| `category` | str | one of the 6 taxonomy names |
| `urgency` | str or null | `low` / `medium` / `high`; **null** for spam and synthetic rows |
| `subject`, `body` | str | cleaned text |
| `text` | str | **the model input**: `subject + "\n\n" + body` (body only if no subject); defined in `ml-service/app/textinput.py`. Train on this column so serving matches |

Rules for training:

- **Deep tier**: train on `train`, select on `val`, never touch `test`.
  Labels are `category`; class order = `config/taxonomy.yaml` order.
  - `train` is capped at 5x the smallest *real* class (`train_cap` in
    `manifest.json`); the rest of the imbalance must be handled with
    **class-weighted cross-entropy**, weights from the `train` class
    frequencies: `w_c = N / (K * n_c)` (sklearn's "balanced"), with N the
    number of train rows, K = 6 classes and n_c the rows of class c.
    No further downsampling or oversampling.
- **Urgency model**: train on `train_urgency`, select on the `val` rows with
  non-null `urgency`, never touch `test`. Labels: `low`, `medium`, `high`
  (ordinal). The English data has **no `critical` rows**; "critical" comes
  from the heuristic/escalation rules in the service, so the urgency model's
  `label_map.json` may list 3 or 4 labels (see 2). `train_urgency` is the
  uncapped train pool, so it shares no rows with val/test. Class-weight it the
  same way if its levels are imbalanced.
- **Checkpoint selection for both models: best val macro-F1** (not accuracy,
  not loss). Record the selected epoch's val macro-F1 and accuracy in
  `model_card.json`.
- The fast tier (trained in this repo, P3.1) uses `class_weight="balanced"`
  and the same macro-F1 selection rule.
- Confirm the test split is the one evaluated here: its `content_sha256` in
  `manifest.json` must match the value in `datasets/README.md` on the Mac.

## 2. Artifact format (PC → Mac)

Copy each model directory into `models/` on the Mac (gitignored):

```
models/deep_tier/                       models/urgency/
├── config.json                         ├── config.json
├── model.safetensors                   ├── model.safetensors
├── tokenizer.json (+ tokenizer files)  ├── tokenizer.json (+ tokenizer files)
├── label_map.json                      ├── label_map.json
└── model_card.json                     └── model_card.json
```

Both are Hugging Face `AutoModelForSequenceClassification` directories
(`save_pretrained` + tokenizer `save_pretrained`).

### `label_map.json`

`{"0": "<label>", "1": "<label>", ...}`: ids are consecutive from 0 and must
match the model's output logits.

- **deep_tier**: exactly the 6 taxonomy names, in any id order:
  `Technical Issue`, `Billing`, `Feature Request`, `General Inquiry`, `Spam`,
  `Meeting Scheduling`.
- **urgency**: ordered by severity: `{"0": "low", "1": "medium", "2": "high"}`,
  optionally with `"3": "critical"` if a future dataset provides it.

### `model_card.json`

```json
{
  "base_model": "microsoft/deberta-v3-base",
  "task": "category" ,
  "trained_on": "groundedtriage-splits-<hash>",
  "trained_at": "2026-10-20T14:03:00Z",
  "max_length": 256,
  "class_weights": {"Technical Issue": 0.0, "...": 0.0},
  "selected_epoch": 3,
  "val_metrics": {"accuracy": 0.0, "macro_f1": 0.0}
}
```

`task` is `"category"` for the deep tier and `"urgency"` for the urgency model.
`max_length` is required (inputs are truncated to it). `val_metrics` holds
whatever was measured on `val` at the selected checkpoint; `macro_f1` is
required (it is the selection metric). For urgency also include ordinal MAE if
available.

### Validation on load (this repo)

The loaders (P3.2 / P3.3) check that every file above exists, that the labels
match the taxonomy / urgency scale, and that `max_length` is present. On any
failure they log what is wrong and fall back (fast tier only / heuristic
urgency). The service never crashes on a bad artifact, and `/health` reports
which models are loaded.

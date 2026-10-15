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
| `processed/train.parquet`, `val.parquet`, `test.parquet` | the splits |
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
- **Urgency model**: use only rows with non-null `urgency` (train/val).
  Labels: `low`, `medium`, `high` (ordinal). The English data has **no
  `critical` rows**; "critical" comes from the heuristic/escalation rules in the
  service, so the urgency model's `label_map.json` may list 3 or 4 labels (see 2).
- `train` is capped (no class above 5x the smallest); `val`/`test` are not.
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
  "val_metrics": {"accuracy": 0.0, "macro_f1": 0.0}
}
```

`task` is `"category"` for the deep tier and `"urgency"` for the urgency model.
`max_length` is required (inputs are truncated to it). `val_metrics` holds
whatever was measured on `val`; for urgency include ordinal MAE if available.

### Validation on load (this repo)

The loaders (P3.2 / P3.3) check that every file above exists, that the labels
match the taxonomy / urgency scale, and that `max_length` is present. On any
failure they log what is wrong and fall back (fast tier only / heuristic
urgency). The service never crashes on a bad artifact, and `/health` reports
which models are loaded.

"""A tiny, randomly initialised sequence classifier saved like a PC artifact
(docs/model_artifacts.md). Nothing is downloaded: the tokenizer is a
word-level `tokenizers` model built in memory."""

from __future__ import annotations

import json
from pathlib import Path

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + (
    "the login error refund invoice please add meeting call spam win prize urgent asap "
    "down today help question"
).split()


def save_tiny_classifier(
    directory: Path,
    labels: list[str],
    task: str,
    max_length: int = 16,
    card_overrides: dict | None = None,
) -> Path:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

    torch.manual_seed(0)
    directory.mkdir(parents=True, exist_ok=True)
    vocab = {w: i for i, w in enumerate(VOCAB)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    fast.save_pretrained(directory)
    config = BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={lab: i for i, lab in enumerate(labels)},
    )
    BertForSequenceClassification(config).save_pretrained(directory)
    (directory / "label_map.json").write_text(
        json.dumps({str(i): lab for i, lab in enumerate(labels)})
    )
    card = {
        "base_model": "tiny-random-bert",
        "task": task,
        "trained_on": "groundedtriage-splits-test",
        "trained_at": "2026-10-16T00:00:00Z",
        "max_length": max_length,
        "val_metrics": {"accuracy": 0.5, "macro_f1": 0.5},
        **(card_overrides or {}),
    }
    (directory / "model_card.json").write_text(json.dumps(card))
    return directory

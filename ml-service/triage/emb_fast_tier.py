"""Embedding fast tier (P3.13, an alternative to the TF-IDF fast tier; no
decision yet, see CLAUDE.md "Deferred decisions").

bge-small-en-v1.5 embeddings (triage.embedder) + logistic regression with
class_weight="balanced", uncalibrated (same choice as the TF-IDF tier in
P3.12). Same interface as FastTier (predict -> TierPrediction, version), so
the cascade and the OOD gate can use either.

Artifact: models/fast_tier_emb[_aug]/{model.joblib (the LR), model_card.json
(embedding_model, max_length, ...)}. The embedder weights come from the
Hugging Face cache, not the artifact.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from app.taxonomy import topic_categories
from triage.embedder import Embedder
from triage.fast_tier import CARD_FILE, MODEL_FILE, TierPrediction

logger = logging.getLogger(__name__)


class EmbFastTier:
    def __init__(self, model, card: dict, embedder: Embedder | None = None):
        self.model = model
        self.card = card
        self.classes: list[str] = [str(c) for c in model.classes_]
        self.embedder = embedder or Embedder(card["embedding_model"], int(card["max_length"]))

    @property
    def version(self) -> str:
        return self.card.get("version", "unknown")

    @classmethod
    def load(cls, directory: Path, embedder: Embedder | None = None) -> EmbFastTier | None:
        directory = Path(directory)
        missing = [f for f in (MODEL_FILE, CARD_FILE) if not (directory / f).exists()]
        if missing:
            logger.warning("embedding fast tier unavailable: %s missing in %s", missing, directory)
            return None
        try:
            import joblib

            card = json.loads((directory / CARD_FILE).read_text())
            model = joblib.load(directory / MODEL_FILE)
            card["embedding_model"], card["max_length"]  # noqa: B018 (required keys)
        except Exception as exc:
            logger.warning("embedding fast tier unavailable: cannot load %s: %s", directory, exc)
            return None
        if {str(c) for c in getattr(model, "classes_", [])} != set(topic_categories()):
            logger.warning("embedding fast tier unavailable: classes != topic categories")
            return None
        return cls(model, card, embedder)

    def predict_proba(self, texts: list[str], vectors: np.ndarray | None = None) -> np.ndarray:
        x = vectors if vectors is not None else self.embedder.encode(texts)
        return self.model.predict_proba(x)

    def predict(self, texts: list[str], vectors: np.ndarray | None = None) -> list[TierPrediction]:
        if not texts:
            return []
        out = []
        for row in self.predict_proba(texts, vectors):
            i = int(np.argmax(row))
            out.append(
                TierPrediction(
                    category=self.classes[i],
                    confidence=float(row[i]),
                    probs={c: float(p) for c, p in zip(self.classes, row, strict=True)},
                )
            )
        return out

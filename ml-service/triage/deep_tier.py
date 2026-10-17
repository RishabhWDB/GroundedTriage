"""Deep tier (P3.2): inference only for the DeBERTa-v3 category classifier
fine-tuned on the GPU PC and copied to models/deep_tier/. No training code
lives in this repo (docs/model_artifacts.md)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.settings import get_settings
from app.taxonomy import topic_categories
from triage.fast_tier import TierPrediction
from triage.hf_model import HFClassifier

TASK = "category"


def check_category_labels(labels: list[str]) -> str | None:
    want = topic_categories()  # Spam is the spam filter's, not a topic class
    if sorted(labels) != sorted(want):
        extra = sorted(set(labels) - set(want))
        missing = sorted(set(want) - set(labels))
        return f"labels must be exactly the 5 topic categories (extra {extra}, missing {missing})"
    return None


class DeepTier(HFClassifier):
    def __init__(self, directory: Path | None = None, **kw):
        s = get_settings()
        kw.setdefault("fp16", s.model_fp16)
        super().__init__(directory or s.models_dir / "deep_tier", TASK, check_category_labels, **kw)

    def predict(self, texts: list[str]) -> list[TierPrediction]:
        probs = self.predict_proba(texts)
        out = []
        for row in probs:
            i = int(np.argmax(row))
            out.append(
                TierPrediction(
                    category=self.labels[i],
                    confidence=float(row[i]),
                    probs={c: float(p) for c, p in zip(self.labels, row, strict=True)},
                )
            )
        return out

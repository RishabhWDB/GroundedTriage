"""SCRIBE's model stages, loaded lazily via transformers pipelines.

- Zero-shot category: facebook/bart-large-mnli over the natural-language
  category descriptions in config/taxonomy.yaml (hypothesis "This email is {}.").
- Sentiment: cardiffnlp/twitter-roberta-base-sentiment-latest, mapped to
  negative -1 / neutral 0 / positive +1.

Needs the optional extra: pip install -e ".[baseline]". Weights (~1.6 GB and
~0.5 GB) are downloaded to the Hugging Face cache on first use.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from app.device import select_device
from app.taxonomy import category_descriptions

logger = logging.getLogger(__name__)

ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
HYPOTHESIS_TEMPLATE = "This email is {}."
SENTIMENT_VALUES = {"negative": -1, "neutral": 0, "positive": 1}


def _pipeline(task: str, model: str, device: str | None):
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            'the SCRIBE baseline needs the optional extra: pip install -e ".[baseline]"'
        ) from exc
    dev = device or select_device()
    logger.info("loading %s on %s", model, dev)
    return pipeline(task, model=model, device=dev)


class ZeroShot:
    def __init__(self, device: str | None = None, batch_size: int = 8):
        self.device, self.batch_size = device, batch_size
        self._pipe = None
        descriptions = category_descriptions()
        self.labels = list(descriptions)
        self._by_description = {d: name for name, d in descriptions.items()}

    def predict(self, texts: Sequence[str]) -> list[tuple[str, float]]:
        """(category, probability) per text; softmax over the 6 descriptions."""
        if not texts:
            return []
        if self._pipe is None:
            self._pipe = _pipeline("zero-shot-classification", ZERO_SHOT_MODEL, self.device)
        out = self._pipe(
            list(texts),
            candidate_labels=list(self._by_description),
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=False,
            batch_size=self.batch_size,
        )
        if isinstance(out, dict):
            out = [out]
        return [(self._by_description[o["labels"][0]], float(o["scores"][0])) for o in out]


class Sentiment:
    def __init__(self, device: str | None = None, batch_size: int = 16):
        self.device, self.batch_size = device, batch_size
        self._pipe = None

    def predict(self, texts: Sequence[str]) -> list[int]:
        """-1 / 0 / +1 per text."""
        if not texts:
            return []
        if self._pipe is None:
            self._pipe = _pipeline("sentiment-analysis", SENTIMENT_MODEL, self.device)
        out = self._pipe(list(texts), truncation=True, max_length=512, batch_size=self.batch_size)
        return [SENTIMENT_VALUES[o["label"].lower()] for o in out]

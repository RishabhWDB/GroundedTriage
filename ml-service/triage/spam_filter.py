"""Binary spam filter (P3.9): the first stage of the cascade.

Trained in this repo (triage.train_spam_filter) on spam_train: SpamAssassin
spam vs SpamAssassin ham + support tickets, so neither writing style nor
source predicts the label on its own. Same features as the fast tier (word
1-2 + char_wb 3-5 TF-IDF) with a balanced logistic regression.

Two thresholds, chosen on spam_val and stored in model_card.json:

- mute (`spam_threshold`): score >= it -> Spam. The lowest score whose false
  positive rate on support tickets is <= 1% (a muted customer is the
  costliest error).
- review (`review_threshold`): review <= score < mute -> "possible spam",
  sent to a human review queue instead of muted.

The card stores two rules (`rules`): "support" (FPR limits on support tickets,
the default) and "support_and_ham" (the same limits on ham too). SPAM_RULE
picks one at serving time.

Artifact: models/spam_filter/{model.joblib, model_card.json}.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from triage.fast_tier import CARD_FILE, MODEL_FILE, build_pipeline

logger = logging.getLogger(__name__)

SPAM, REVIEW, HAM = "spam", "review", "ham"


def build_filter(c: float = 4.0):
    """The fast tier's feature pipeline as a binary classifier (classes False/True)."""
    return build_pipeline(c)


@dataclass
class SpamDecision:
    score: float  # P(spam) from the LR
    decision: str  # spam | review | ham


def decide(score: float, spam_threshold: float, review_threshold: float) -> str:
    if score >= spam_threshold:
        return SPAM
    if score >= review_threshold:
        return REVIEW
    return HAM


class SpamFilter:
    def __init__(self, model, card: dict, rule: str | None = None):
        """rule: a key of card["rules"] (default: card["default_rule"]); cards
        without rules use their top-level thresholds."""
        self.model = model
        self.card = card
        self.rule = rule or card.get("default_rule", "default")
        t = card.get("rules", {}).get(self.rule, card) if "rules" in card else card
        if rule and rule not in card.get("rules", {}):
            raise ValueError(f"unknown spam rule {rule!r}; card has {list(card.get('rules', {}))}")
        self.spam_threshold = float(t["spam_threshold"])
        self.review_threshold = float(t["review_threshold"])
        self._pos = list(model.classes_).index(True)

    @property
    def version(self) -> str:
        return f"{self.card.get('version', 'unknown')}:{self.rule}"

    @classmethod
    def load(cls, directory: Path, rule: str | None = None) -> SpamFilter | None:
        directory = Path(directory)
        missing = [f for f in (MODEL_FILE, CARD_FILE) if not (directory / f).exists()]
        if missing:
            logger.warning("spam filter unavailable: %s missing in %s", missing, directory)
            return None
        try:
            import joblib

            card = json.loads((directory / CARD_FILE).read_text())
            model = joblib.load(directory / MODEL_FILE)
            filt = cls(model, card, rule)
        except Exception as exc:  # corrupt pickle, bad JSON, missing thresholds, unknown rule
            logger.warning("spam filter unavailable: cannot load %s: %s", directory, exc)
            return None
        spam_t, review_t = filt.spam_threshold, filt.review_threshold
        if set(getattr(model, "classes_", [])) != {False, True}:
            logger.warning("spam filter unavailable: not a binary is_spam classifier")
            return None
        if not 0 <= review_t <= spam_t <= 1:
            logger.warning("spam filter unavailable: need 0 <= review <= spam threshold <= 1")
            return None
        logger.info("spam filter %s loaded from %s", filt.version, directory)
        return filt

    def scores(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0)
        # float64: thresholds sit one ulp above a score, which float32 can't represent.
        return self.model.predict_proba(texts)[:, self._pos].astype(np.float64)

    def predict(self, texts: list[str]) -> list[SpamDecision]:
        return [
            SpamDecision(float(s), decide(float(s), self.spam_threshold, self.review_threshold))
            for s in self.scores(texts)
        ]

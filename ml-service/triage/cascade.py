"""Classification cascade (P3.4; spam filter in front since P3.10).

    spam filter -> score >= mute threshold ? Spam (tier=spam_filter); topic models skipped
                -> score >= review threshold ? continue, flagged possible_spam
    fast tier (5 topic classes)
              -> calibrated confidence >= tau ? accept (tier=fast)
              -> else deep tier available ? deep prediction (tier=deep)
              -> else accept the fast prediction with low_confidence=True
    always: urgency (model, else heuristic; critical only from the heuristic rule)

tau lives in config/cascade.yaml and is chosen by triage.tune_threshold on
val once the deep tier exists. Implements app.interfaces.TicketClassifier so
the P7 evaluation harness can swap it with the SCRIBE baseline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

from app.interfaces import CategoryPrediction, TicketText
from app.schemas import Classification, ModelsLoaded
from app.settings import REPO_ROOT
from triage.fast_tier import TierPrediction
from triage.spam_filter import REVIEW, SPAM, SpamDecision
from triage.urgency import UrgencyPredictor, UrgencyResult

logger = logging.getLogger(__name__)

CONFIG_PATH = REPO_ROOT / "config" / "cascade.yaml"


class CategoryModel(Protocol):
    @property
    def version(self) -> str: ...

    def predict(self, texts: list[str]) -> list[TierPrediction]: ...


class OptionalCategoryModel(CategoryModel, Protocol):
    @property
    def available(self) -> bool: ...


class SpamModel(Protocol):
    @property
    def version(self) -> str: ...

    def predict(self, texts: list[str]) -> list[SpamDecision]: ...


def load_tau(path: Path = CONFIG_PATH) -> float:
    tau = float(yaml.safe_load(path.read_text())["tau"])
    if not 0 <= tau <= 1:
        raise ValueError(f"{path}: tau must be in [0, 1], got {tau}")
    return tau


@dataclass
class CascadeResult:
    category: str
    confidence: float
    tier: str  # fast | deep | spam_filter
    low_confidence: bool
    fast: TierPrediction | None  # None when the spam filter muted the ticket
    urgency: UrgencyResult
    spam_score: float | None = None  # None without a spam filter
    possible_spam: bool = False


class Cascade:
    def __init__(
        self,
        fast: CategoryModel,
        deep: OptionalCategoryModel | None,
        urgency: UrgencyPredictor,
        tau: float,
        spam: SpamModel | None = None,
    ):
        self.fast, self.deep, self.urgency, self.tau = fast, deep, urgency, tau
        self.spam = spam

    @property
    def deep_available(self) -> bool:
        return self.deep is not None and self.deep.available

    def run(self, texts: list[str]) -> list[CascadeResult]:
        if not texts:
            return []
        spam: list[SpamDecision | None] = (
            self.spam.predict(texts) if self.spam is not None else [None] * len(texts)
        )
        topic_idx = [i for i, d in enumerate(spam) if d is None or d.decision != SPAM]
        topic = self._topic([texts[i] for i in topic_idx])
        by_index = dict(zip(topic_idx, topic, strict=True))
        urg = self.urgency.predict(texts)
        out = []
        for i, d in enumerate(spam):
            score = d.score if d is not None else None
            if i not in by_index:  # muted by the spam filter
                out.append(
                    CascadeResult("Spam", d.score, "spam_filter", False, None, urg[i], score)
                )
                continue
            cat, conf, tier, low, f = by_index[i]
            review = d is not None and d.decision == REVIEW
            out.append(CascadeResult(cat, conf, tier, low, f, urg[i], score, review))
        return out

    def _topic(self, texts: list[str]) -> list[tuple]:
        """(category, confidence, tier, low_confidence, fast prediction) per text."""
        if not texts:
            return []
        fast = self.fast.predict(texts)
        escalate = [i for i, p in enumerate(fast) if p.confidence < self.tau]
        deep: dict[int, TierPrediction] = {}
        if escalate and self.deep_available:
            try:
                for i, p in zip(
                    escalate, self.deep.predict([texts[i] for i in escalate]), strict=True
                ):
                    deep[i] = p
            except RuntimeError as exc:  # deep tier failed to load: keep fast, flag it
                logger.warning("deep tier failed, accepting fast tier: %s", exc)
        out = []
        for i, f in enumerate(fast):
            if i in deep:
                out.append((deep[i].category, deep[i].confidence, "deep", False, f))
            else:
                out.append((f.category, f.confidence, "fast", f.confidence < self.tau, f))
        return out

    def models_loaded(self) -> ModelsLoaded:
        return ModelsLoaded(deep_tier=self.deep_available, urgency=self.urgency.model_available)

    def model_versions(self) -> dict[str, str]:
        v = {
            "spam_filter": self.spam.version if self.spam is not None else "absent",
            "fast_tier": self.fast.version,
            "tau": f"{self.tau:g}",
        }
        if self.deep_available:
            v["deep_tier"] = self.deep.version
        v["urgency"] = self.urgency.model.version if self.urgency.model_available else "heuristic"
        return v

    def classify_tickets(self, tickets: list[TicketText]) -> list[Classification]:
        results = self.run([t.text for t in tickets])
        loaded, versions = self.models_loaded(), self.model_versions()
        return [
            Classification(
                ticket_id=t.id,
                category=r.category,
                category_confidence=round(r.confidence, 6),
                tier=r.tier,
                low_confidence=r.low_confidence,
                urgency=r.urgency.urgency,
                urgency_confidence=round(r.urgency.confidence, 6),
                urgency_source=r.urgency.source,
                spam_score=None if r.spam_score is None else round(r.spam_score, 6),
                possible_spam=r.possible_spam,
                models_loaded=loaded,
                model_versions=versions,
            )
            for t, r in zip(tickets, results, strict=True)
        ]

    # app.interfaces.TicketClassifier, for the evaluation harness.
    def classify(self, tickets: list[TicketText]) -> list[CategoryPrediction]:
        return [
            CategoryPrediction(
                category=r.category,
                confidence=min(1.0, max(0.0, r.confidence)),
                stage=r.tier,
                urgency=r.urgency.urgency,
                signals={
                    "low_confidence": float(r.low_confidence),
                    "possible_spam": float(r.possible_spam),
                    **({"fast_confidence": r.fast.confidence} if r.fast else {}),
                    **({"spam_score": r.spam_score} if r.spam_score is not None else {}),
                },
            )
            for r in self.run([t.text for t in tickets])
        ]

"""Classification cascade (P3.4).

    fast tier -> calibrated confidence >= tau ? accept (tier=fast)
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


def load_tau(path: Path = CONFIG_PATH) -> float:
    tau = float(yaml.safe_load(path.read_text())["tau"])
    if not 0 <= tau <= 1:
        raise ValueError(f"{path}: tau must be in [0, 1], got {tau}")
    return tau


@dataclass
class CascadeResult:
    category: str
    confidence: float
    tier: str  # fast | deep
    low_confidence: bool
    fast: TierPrediction
    urgency: UrgencyResult


class Cascade:
    def __init__(
        self,
        fast: CategoryModel,
        deep: OptionalCategoryModel | None,
        urgency: UrgencyPredictor,
        tau: float,
    ):
        self.fast, self.deep, self.urgency, self.tau = fast, deep, urgency, tau

    @property
    def deep_available(self) -> bool:
        return self.deep is not None and self.deep.available

    def run(self, texts: list[str]) -> list[CascadeResult]:
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
        urg = self.urgency.predict(texts)
        out = []
        for i, f in enumerate(fast):
            if i in deep:
                d = deep[i]
                out.append(CascadeResult(d.category, d.confidence, "deep", False, f, urg[i]))
            else:
                low = f.confidence < self.tau
                out.append(CascadeResult(f.category, f.confidence, "fast", low, f, urg[i]))
        return out

    def models_loaded(self) -> ModelsLoaded:
        return ModelsLoaded(deep_tier=self.deep_available, urgency=self.urgency.model_available)

    def model_versions(self) -> dict[str, str]:
        v = {"fast_tier": self.fast.version, "tau": f"{self.tau:g}"}
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
                    "fast_confidence": r.fast.confidence,
                },
            )
            for r in self.run([t.text for t in tickets])
        ]

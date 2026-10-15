"""Classifier and drafter interfaces shared by the SCRIBE baseline (P2.2) and
the proposed system (P3.4 cascade, P5.3 generator), so the evaluation harness
(P2.3, P7) can swap one for the other.

These are evaluation-level types. The persisted contracts (Classification,
ReplyDraft in app.schemas) are built from them by the serving pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, Field, field_validator

from app.schemas import Urgency
from app.taxonomy import categories
from app.textinput import model_input


class TicketText(BaseModel):
    id: str = ""
    subject: str = ""
    body: str = ""

    @property
    def text(self) -> str:
        return model_input(self.subject, self.body)


class CategoryPrediction(BaseModel):
    category: str
    confidence: float = Field(ge=0, le=1)
    # Which component decided: "rule" / "zero_shot" (baseline), "fast" / "deep" (cascade).
    stage: str
    urgency: Urgency | None = None
    # Extra per-system signals (e.g. SCRIBE's sentiment in -1/0/+1).
    signals: dict[str, float] = Field(default_factory=dict)

    @field_validator("category")
    @classmethod
    def _known(cls, v: str) -> str:
        if v not in categories():
            raise ValueError(f"unknown category {v!r}; see config/taxonomy.yaml")
        return v


class DraftResult(BaseModel):
    body: str
    # Intermediate versions in order (SCRIBE: [draft, refined]).
    stages: list[str] = Field(default_factory=list)
    # Knowledge-base chunk ids the reply relies on; always empty for the ungrounded baseline.
    citations: list[str] = Field(default_factory=list)
    grounded: bool = False


class TicketClassifier(Protocol):
    name: str

    def classify(self, tickets: Sequence[TicketText]) -> list[CategoryPrediction]: ...


class ReplyDrafter(Protocol):
    name: str

    def draft(self, ticket: TicketText, prediction: CategoryPrediction) -> DraftResult: ...

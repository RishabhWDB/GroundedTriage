"""Shared data contracts. Mirrors internal/model (Go); keep the two in sync."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.taxonomy import categories as taxonomy_categories

SourceName = Literal["email", "zendesk", "csv"]
Tier = Literal["fast", "deep"]
Urgency = Literal["low", "medium", "high", "critical"]
UrgencySource = Literal["model", "heuristic"]
IngestStatus = Literal["accepted", "skipped"]
Action = Literal["mute", "queue", "reply", "schedule"]
DraftStatus = Literal["grounded", "abstained", "approved", "sent", "rejected"]

URGENCY_SCALE: tuple[Urgency, ...] = ("low", "medium", "high", "critical")


class Ticket(BaseModel):
    id: str
    source: SourceName
    external_id: str
    sender: str = ""
    subject: str = ""
    body: str = ""
    received_at: datetime
    thread_id: str | None = None
    raw_ref: str = ""
    content_hash: str
    # "skipped" tickets (e.g. non-English) are stored but never classified.
    ingest_status: IngestStatus = "accepted"
    skip_reason: str = ""


class ModelsLoaded(BaseModel):
    deep_tier: bool = False
    urgency: bool = False


class Classification(BaseModel):
    ticket_id: str
    category: str
    category_confidence: float = Field(ge=0, le=1)
    tier: Tier
    # Extension of the spec contract: set when the fast tier was accepted below tau
    # because the deep tier is unavailable (P3.4); consumed by routing (P4.1).
    low_confidence: bool = False
    urgency: Urgency
    urgency_confidence: float = Field(ge=0, le=1)
    urgency_source: UrgencySource
    models_loaded: ModelsLoaded
    model_versions: dict[str, str] = Field(default_factory=dict)

    @field_validator("category")
    @classmethod
    def _known_category(cls, v: str) -> str:
        if v not in taxonomy_categories():
            raise ValueError(f"unknown category {v!r}; see config/taxonomy.yaml")
        return v


class Route(BaseModel):
    ticket_id: str
    action: Action
    queue: str = ""
    priority: int = 0
    sla_due_at: datetime | None = None
    reason: str = ""


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str = ""
    text: str
    score: float


class ReplyDraft(BaseModel):
    ticket_id: str
    subject: str = ""
    body: str = ""
    citations: list[str] = Field(default_factory=list)
    faithfulness_score: float | None = Field(default=None, ge=0, le=1)
    unsupported_claims: list[str] = Field(default_factory=list)
    status: DraftStatus


class Health(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
    models_loaded: ModelsLoaded
    fast_tier_loaded: bool
    torch_installed: bool

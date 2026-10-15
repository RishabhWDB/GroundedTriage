"""The SCRIBE baseline behind the shared interfaces (app.interfaces).

Classification: regex rules first (accepted at confidence >= 0.95), else
BART-MNLI zero-shot over the taxonomy descriptions. Sentiment (-1/0/+1) is
computed for every ticket and mapped to an urgency the way SCRIBE prioritises
mail: negative -> high, neutral -> medium, positive -> low.

Drafting: two LLM passes, draft then refine, with no retrieval and no
grounding. The LLM is the same gpt-oss-120b drafter the proposed system uses
(documented deviation: SCRIBE used LLaMA-3-70B), so P7 isolates the effect of
grounding.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from app.interfaces import CategoryPrediction, DraftResult, TicketText
from app.llm import LLM
from baseline.rules import apply_rules

SENTIMENT_URGENCY = {-1: "high", 0: "medium", 1: "low"}


class ZeroShotModel(Protocol):
    def predict(self, texts: Sequence[str]) -> list[tuple[str, float]]: ...


class SentimentModel(Protocol):
    def predict(self, texts: Sequence[str]) -> list[int]: ...


@dataclass
class StageStats:
    """Cumulative counts and wall-clock seconds per stage (for latency tables)."""

    tickets: int = 0
    by_rule: int = 0
    by_zero_shot: int = 0
    seconds: dict[str, float] = field(
        default_factory=lambda: {"rules": 0.0, "zero_shot": 0.0, "sentiment": 0.0}
    )


class ScribeClassifier:
    name = "scribe"

    def __init__(
        self,
        zero_shot: ZeroShotModel | None = None,
        sentiment: SentimentModel | None = None,
        *,
        device: str | None = None,
    ):
        if zero_shot is None or sentiment is None:
            from baseline.models import Sentiment, ZeroShot  # needs the [baseline] extra

            zero_shot = zero_shot or ZeroShot(device)
            sentiment = sentiment or Sentiment(device)
        self.zero_shot, self.sentiment = zero_shot, sentiment
        self.stats = StageStats()

    def classify(self, tickets: Sequence[TicketText]) -> list[CategoryPrediction]:
        t0 = time.perf_counter()
        matches = [apply_rules(t.subject, t.body) for t in tickets]
        t1 = time.perf_counter()
        fallback = [i for i, m in enumerate(matches) if m is None or not m.accepted]
        zs = dict(
            zip(
                fallback,
                self.zero_shot.predict([tickets[i].text for i in fallback]),
                strict=True,
            )
        )
        t2 = time.perf_counter()
        sentiments = self.sentiment.predict([t.text for t in tickets])
        t3 = time.perf_counter()

        s = self.stats
        s.tickets += len(tickets)
        s.by_zero_shot += len(fallback)
        s.by_rule += len(tickets) - len(fallback)
        for k, dt in (("rules", t1 - t0), ("zero_shot", t2 - t1), ("sentiment", t3 - t2)):
            s.seconds[k] += dt

        out = []
        for i, sent in enumerate(sentiments):
            if i in zs:
                category, conf = zs[i]
                stage, signals = "zero_shot", {}
            else:
                m = matches[i]
                category, conf, stage = m.category, m.confidence, "rule"
                signals = {f"rule:{r}": 1.0 for r in m.rules}
            out.append(
                CategoryPrediction(
                    category=category,
                    confidence=min(max(conf, 0.0), 1.0),
                    stage=stage,
                    urgency=SENTIMENT_URGENCY[sent],
                    signals={"sentiment": float(sent), **signals},
                )
            )
        return out


DRAFT_SYSTEM = (
    "You are an AI email assistant for a company's customer-support team. Write a reply "
    "to the customer's email on behalf of the support team. Be helpful, polite and "
    "professional. Reply with the email body only."
)
DRAFT_USER = (
    "Email category: {category}\n"
    "Sender sentiment: {sentiment}\n\n"
    "Subject: {subject}\n\n"
    "{body}\n\n"
    "Write a reply to this email."
)
REFINE_SYSTEM = (
    "You are an editor for customer-support emails. Improve the draft reply: make it "
    "clearer, more concise and professional, with an appropriate tone for the "
    "sender's sentiment, and fix any grammar. Reply with the refined email body only."
)
REFINE_USER = "Original email:\nSubject: {subject}\n\n{body}\n\nDraft reply:\n{draft}"
SENTIMENT_WORDS = {-1: "negative", 0: "neutral", 1: "positive"}


class ScribeDrafter:
    name = "scribe"

    def __init__(self, llm: LLM, temperature: float = 0.0):
        self.llm, self.temperature = llm, temperature

    def draft(self, ticket: TicketText, prediction: CategoryPrediction) -> DraftResult:
        sentiment = SENTIMENT_WORDS[int(prediction.signals.get("sentiment", 0))]
        first = self.llm.complete(
            [
                {"role": "system", "content": DRAFT_SYSTEM},
                {
                    "role": "user",
                    "content": DRAFT_USER.format(
                        category=prediction.category,
                        sentiment=sentiment,
                        subject=ticket.subject,
                        body=ticket.body,
                    ),
                },
            ],
            temperature=self.temperature,
        ).strip()
        refined = self.llm.complete(
            [
                {"role": "system", "content": REFINE_SYSTEM},
                {
                    "role": "user",
                    "content": REFINE_USER.format(
                        subject=ticket.subject, body=ticket.body, draft=first
                    ),
                },
            ],
            temperature=self.temperature,
        ).strip()
        return DraftResult(body=refined, stages=[first, refined], citations=[], grounded=False)

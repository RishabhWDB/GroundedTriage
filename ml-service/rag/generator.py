"""Grounded draft generator (P5.3): POST /draft.

The drafter (gpt-oss-120b, Groq JSON mode, temperature 0) answers **only**
from the numbered retrieved chunks and cites them inline as [n]. Tone comes
from urgency and category. It must not promise actions (refunds issued,
accounts changed, ...) that a source doesn't state: that is exactly the
ungrounded SCRIBE failure in results/examples/ungrounded_hallucination_01.md.
When the sources don't answer the ticket, it says so and asks for what is
missing instead of guessing.

SCRIBE's draft -> refine second pass is kept, constrained: the refiner may
improve clarity and tone but must not add, drop or change any fact or
citation. Citations are parsed from the final body: [n] markers outside
1..len(chunks) are removed and reported; `citations` lists the cited chunk
ids in first-cited order. The faithfulness gate (P5.4) decides grounded vs
abstained.
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel, Field

from app.llm import LLM, JSONModeError, LLMError
from app.schemas import RetrievedChunk, Urgency

logger = logging.getLogger(__name__)

SIGN_OFF = "Best regards,\nThe Support Team"

TONE = {
    "critical": "This is critical for the customer: acknowledge the impact first, be direct and "
    "concise, and state the next steps the sources describe.",
    "high": "The customer needs a quick resolution: be direct and empathetic, steps first.",
    "medium": "Be friendly and clear.",
    "low": "Be friendly and brief.",
}
CATEGORY_HINT = {
    "Billing": "Be precise about amounts, dates and policies, and only as the sources state them.",
    "Technical Issue": "Give numbered troubleshooting steps when the sources contain them.",
    "Feature Request": "Thank them for the suggestion; describe only what the sources say exists.",
    "General Inquiry": "Answer the question directly.",
    "Meeting Scheduling": "Answer only the questions you can; scheduling is handled separately.",
}

DRAFT_SYSTEM = (
    "You are a customer-support agent writing an email reply. Use ONLY the numbered sources "
    "below. Every factual statement must end with the [n] citation(s) of the source(s) that "
    "support it. Do not use outside knowledge. Never claim an action was taken (a refund issued, "
    "a setting changed, an escalation made) and never promise one unless a source says it "
    "happens. If the sources do not answer the customer's question, say that plainly and ask "
    "for the information you would need; do not guess. No placeholders such as [Your Name]. "
    f"End the body with:\n{SIGN_OFF}\nOutput JSON only."
)
DRAFT_USER = """Customer email
Subject: {subject}
{body}

Ticket: category {category}, urgency {urgency}. Tone: {tone} {hint}

Sources:
{sources}

Return {{"subject": "Re: ...", "body": "the reply with [n] citations"}}."""

REFINE_SYSTEM = (
    "You edit a customer-support reply for clarity, tone and concision. You must NOT add, "
    "remove or change any fact, number, step, promise or [n] citation, and must not add new "
    "information of any kind. Keep the sign-off. Output JSON only."
)
REFINE_USER = """Reply to edit:
Subject: {subject}
{body}

Return {{"subject": "...", "body": "..."}} with the same facts and the same [n] citations."""

_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_PLACEHOLDER = re.compile(r"\[(?:your|agent|company|customer)[^\]]*\]", re.I)


class DraftRequest(BaseModel):
    ticket_id: str
    subject: str = ""
    body: str = ""
    category: str = "General Inquiry"
    urgency: Urgency = "medium"
    chunks: list[RetrievedChunk]
    refine: bool = True


class GeneratedDraft(BaseModel):
    ticket_id: str
    subject: str
    body: str
    # Cited chunk ids in first-cited order, and their [n] numbers.
    citations: list[str] = Field(default_factory=list)
    cited_numbers: list[int] = Field(default_factory=list)
    # [n] markers that pointed outside the sources (removed from the body).
    invalid_citations: list[int] = Field(default_factory=list)
    refined: bool = False
    notes: list[str] = Field(default_factory=list)


def format_sources(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i}] {c.title}\n{c.text}".strip() for i, c in enumerate(chunks, start=1))


def build_draft_messages(req: DraftRequest) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DRAFT_SYSTEM},
        {
            "role": "user",
            "content": DRAFT_USER.format(
                subject=req.subject or "(none)",
                body=req.body,
                category=req.category,
                urgency=req.urgency,
                tone=TONE.get(req.urgency, TONE["medium"]),
                hint=CATEGORY_HINT.get(req.category, ""),
                sources=format_sources(req.chunks),
            ),
        },
    ]


def build_refine_messages(subject: str, body: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": REFINE_SYSTEM},
        {"role": "user", "content": REFINE_USER.format(subject=subject, body=body)},
    ]


def parse_citations(body: str, n_sources: int) -> tuple[str, list[int], list[int]]:
    """(body with invalid markers removed, valid numbers in first-cited order, invalid numbers)."""
    valid: list[int] = []
    invalid: list[int] = []

    def fix(m: re.Match) -> str:
        nums = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
        keep = [n for n in nums if 1 <= n <= n_sources]
        for n in nums:
            target = valid if 1 <= n <= n_sources else invalid
            if n not in target:
                target.append(n)
        return f"[{', '.join(map(str, keep))}]" if keep else ""

    cleaned = _CITE.sub(fix, body)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)  # space left before punctuation
    return cleaned, valid, invalid


def _json_reply(llm: LLM, messages, attempts: int = 2) -> dict:
    for attempt in range(attempts):
        try:
            text = llm.complete(messages, temperature=0.0, json_mode=True)
        except JSONModeError as exc:
            logger.warning("draft attempt %d: JSON mode failed: %s", attempt + 1, exc)
            continue
        try:
            out = json.loads(text)
            if isinstance(out, dict) and str(out.get("body", "")).strip():
                return {"subject": str(out.get("subject", "")).strip(), "body": str(out["body"])}
            raise ValueError("no body")
        except ValueError as exc:
            logger.warning("draft attempt %d: bad JSON: %s", attempt + 1, exc)
    raise LLMError("draft: the LLM returned invalid JSON twice")


def generate(req: DraftRequest, llm: LLM) -> GeneratedDraft:
    """Raises ValueError without sources (the gate abstains instead) and
    LLMError when the LLM fails."""
    if not req.chunks:
        raise ValueError("no retrieved chunks: nothing to ground a reply on")
    notes: list[str] = []
    draft = _json_reply(llm, build_draft_messages(req))
    refined = False
    if req.refine:
        before = set(parse_citations(draft["body"], len(req.chunks))[1])
        out = _json_reply(llm, build_refine_messages(draft["subject"], draft["body"]))
        after = set(parse_citations(out["body"], len(req.chunks))[1])
        if after == before:
            draft, refined = out, True
        else:  # the refiner changed the citations: keep the unrefined draft
            notes.append(
                f"refine pass dropped: citations changed {sorted(before)} -> {sorted(after)}"
            )
    body, valid, invalid = parse_citations(draft["body"], len(req.chunks))
    if invalid:
        notes.append(f"removed citations to non-existent sources {invalid}")
    if _PLACEHOLDER.search(body):
        notes.append("draft contains a placeholder")
    subject = draft["subject"] or (f"Re: {req.subject}" if req.subject else "Re: your request")
    return GeneratedDraft(
        ticket_id=req.ticket_id,
        subject=subject,
        body=body,
        citations=[req.chunks[n - 1].chunk_id for n in valid],
        cited_numbers=valid,
        invalid_citations=invalid,
        refined=refined,
        notes=notes,
    )

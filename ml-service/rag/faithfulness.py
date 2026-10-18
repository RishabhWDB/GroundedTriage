"""Faithfulness gate + abstain (P5.4): POST /verify.

One judge call (qwen3.8-27b by default, a different family from the drafter;
temperature 0, JSON) splits the draft into atomic factual claims and judges
each against the sources it cites (or all sources when it cites none).
Greetings, sign-offs, empathy and questions to the customer are not claims.

    faithfulness_score = supported claims / all claims     (1.0 with no claims)

Status (thresholds in config/grounding.yaml):
- abstained: no retrieved chunks; or best retrieval score < retrieval_floor;
  or faithfulness_score < min_faithfulness. The reason and the unsupported
  claims are recorded, and the ticket goes to a human queue (P5.5).
- grounded: otherwise.

A draft whose claim list would overflow the judge's max_tokens (qwen's
output limit is 1,000 tokens a minute, so it can't simply be raised) is
judged in two halves against the same sources, recursively up to
MAX_SPLIT_DEPTH; faithfulness is per claim, so the halves' claims are pooled.

Judge calls are cached on disk (sha256 of model + messages) under
datasets/raw/llm_cache/judge/, so re-running an evaluation (P7.2) across the
Groq free tier's daily cap costs nothing for work already done.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from app.fileio import atomic_write_text
from app.llm import LLM, JSONModeError, LLMError, TruncatedError
from app.schemas import ReplyDraft, RetrievedChunk
from app.settings import REPO_ROOT
from rag.generator import format_sources

logger = logging.getLogger(__name__)

CONFIG_PATH = REPO_ROOT / "config" / "grounding.yaml"
CACHE_DIR = REPO_ROOT / "datasets" / "raw" / "llm_cache" / "judge"

JUDGE_SYSTEM = (
    "You check whether a customer-support reply is supported by its sources. Split the reply "
    "into atomic factual claims (one checkable fact each). Ignore greetings, sign-offs, apologies, "
    "expressions of empathy and questions asked to the customer. For each claim, decide whether "
    "the cited sources (or, if it cites none, any of the sources) state or directly imply it. "
    "A claim that a specific action was taken or will be taken (refund issued, account changed) "
    "is supported only if a source says so. Output JSON only."
)
JUDGE_USER = """Sources:
{sources}

Reply:
{draft}

Return {{"claims": [{{"claim": "...", "cited": [n, ...], "supported": true|false,
"reason": "short"}}]}}. Use an empty list when the reply makes no factual claims."""


class GateConfig(BaseModel):
    min_faithfulness: float = Field(default=0.9, ge=0, le=1)
    retrieval_floor: float = Field(default=0.0, ge=0, le=1)


def load_config(path: Path = CONFIG_PATH) -> GateConfig:
    if not path.exists():
        return GateConfig()
    return GateConfig(**(yaml.safe_load(path.read_text()) or {}).get("gate", {}))


class Claim(BaseModel):
    claim: str
    cited: list[int] = Field(default_factory=list)
    supported: bool
    reason: str = ""


class VerifyRequest(BaseModel):
    ticket_id: str
    subject: str = ""
    body: str
    citations: list[str] = Field(default_factory=list)
    chunks: list[RetrievedChunk]


class Verification(BaseModel):
    draft: ReplyDraft
    claims: list[Claim] = Field(default_factory=list)
    abstain_reason: str = ""
    retrieval_max_score: float = 0.0
    thresholds: GateConfig


class JudgeCache:
    def __init__(self, root: Path | None = None):
        self.root = root or CACHE_DIR  # resolved at call time (tests redirect it)

    def key(self, model: str, messages) -> str:
        blob = json.dumps({"model": model, "messages": messages}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> str | None:
        p = self.root / f"{key}.json"
        return json.loads(p.read_text())["response"] if p.exists() else None

    def put(self, key: str, response: str) -> None:
        atomic_write_text(self.root / f"{key}.json", json.dumps({"response": response}))


def judge_messages(body: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER.format(sources=format_sources(chunks), draft=body)},
    ]


def parse_claims(text: str) -> list[Claim]:
    data = json.loads(text)
    claims = data.get("claims") if isinstance(data, dict) else None
    if not isinstance(claims, list):
        raise ValueError("no claims list")
    return [Claim(**c) for c in claims]


MAX_SPLIT_DEPTH = 2


def split_halves(body: str) -> tuple[str, str] | None:
    """Cut a draft near its middle at a paragraph (else sentence) boundary."""
    for sep in ("\n\n", "\n", ". "):
        parts = [p for p in body.split(sep) if p.strip()]
        if len(parts) < 2:
            continue
        total = len(body)
        cut = min(
            range(1, len(parts)), key=lambda i: abs(sum(len(p) for p in parts[:i]) - total / 2)
        )
        return sep.join(parts[:cut]), sep.join(parts[cut:])
    return None


def judge(
    body: str,
    chunks: list[RetrievedChunk],
    llm: LLM,
    model: str = "judge",
    cache: JudgeCache | None = None,
    _depth: int = 0,
) -> list[Claim]:
    try:
        return _judge_once(body, chunks, llm, model, cache)
    except TruncatedError:
        halves = split_halves(body)
        if halves is None or _depth >= MAX_SPLIT_DEPTH:
            raise
        logger.info(
            "judge output truncated; judging the draft in two halves (depth %d)", _depth + 1
        )
        return [c for h in halves for c in judge(h, chunks, llm, model, cache, _depth + 1)]


def _judge_once(
    body: str,
    chunks: list[RetrievedChunk],
    llm: LLM,
    model: str,
    cache: JudgeCache | None,
) -> list[Claim]:
    messages = judge_messages(body, chunks)
    for attempt in range(2):
        msgs = [dict(m) for m in messages]
        if attempt:
            msgs[-1]["content"] += "\n(Attempt 2: return valid JSON as specified.)"
        key = cache.key(model, msgs) if cache else ""
        text = cache.get(key) if cache else None
        if text is None:
            try:
                text = llm.complete(msgs, temperature=0.0, json_mode=True)
            except JSONModeError as exc:
                logger.warning("judge attempt %d: JSON mode failed: %s", attempt + 1, exc)
                continue
        try:
            claims = parse_claims(text)
        except (ValueError, TypeError) as exc:
            logger.warning("judge attempt %d: bad output: %s", attempt + 1, exc)
            continue
        if cache:
            cache.put(key, text)
        return claims
    raise LLMError("judge: invalid JSON twice")


def verify(
    req: VerifyRequest,
    llm: LLM | None,
    cfg: GateConfig,
    model: str = "judge",
    cache: JudgeCache | None = None,
) -> Verification:
    """Raises LLMError when the judge fails (the caller retries later)."""
    max_score = max((c.score for c in req.chunks), default=0.0)

    def result(status, score, claims, reason):
        unsupported = [c.claim for c in claims if not c.supported]
        draft = ReplyDraft(
            ticket_id=req.ticket_id,
            subject=req.subject,
            body=req.body,
            citations=req.citations,
            faithfulness_score=score,
            unsupported_claims=unsupported,
            status=status,
        )
        return Verification(
            draft=draft,
            claims=claims,
            abstain_reason=reason,
            retrieval_max_score=max_score,
            thresholds=cfg,
        )

    if not req.chunks:
        return result("abstained", None, [], "no retrieved sources")
    if max_score < cfg.retrieval_floor:
        return result(
            "abstained",
            None,
            [],
            f"best retrieval score {max_score:.3f} < floor {cfg.retrieval_floor}",
        )
    if llm is None:
        raise LLMError("no judge LLM configured")
    claims = judge(req.body, req.chunks, llm, model, cache)
    score = 1.0 if not claims else sum(c.supported for c in claims) / len(claims)
    score = round(score, 4)
    if score < cfg.min_faithfulness:
        n_bad = sum(not c.supported for c in claims)
        return result(
            "abstained",
            score,
            claims,
            f"faithfulness {score:.2f} < {cfg.min_faithfulness} "
            f"({n_bad} of {len(claims)} claims unsupported)",
        )
    return result("grounded", score, claims, "")

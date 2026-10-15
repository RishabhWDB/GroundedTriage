"""Synthetic "Meeting Scheduling" emails generated with the drafter LLM
(gpt-oss-120b via Groq). Generated once and cached; every row has
origin="synthetic" so metrics can be split by origin.

Reproducibility: the scenario specs come from a seeded RNG over the option
lists below and the prompt is fixed, both committed. Each LLM response is
cached on disk keyed by sha256(model + prompt), so re-runs make no API calls,
and an interrupted run resumes where it stopped. The final rows are written
to datasets/synthetic/meeting_requests.jsonl (committed; see CLAUDE.md).

    python -m data.synthetic_meetings            # generate (resumable)
    python -m data.synthetic_meetings --limit 2  # first 2 batches only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from app.llm import LLM, LLMError, build_llm
from app.settings import REPO_ROOT, get_settings

logger = logging.getLogger(__name__)

SEED = 42
N_ROWS = 400
BATCH_SIZE = 5
OUT_PATH = REPO_ROOT / "datasets" / "synthetic" / "meeting_requests.jsonl"
CACHE_DIR = REPO_ROOT / "datasets" / "raw" / "llm_cache" / "synthetic_meetings"

MEETING_TYPES = [
    "product demo for their team",
    "onboarding session for new users",
    "escalation call about an unresolved support issue",
    "quarterly business review",
    "technical deep-dive on an integration",
    "contract renewal discussion",
    "training session on advanced features",
    "post-incident review after an outage",
    "implementation kickoff call",
    "data migration planning meeting",
    "security and compliance review",
    "pricing and plan upgrade discussion",
]
ACTIONS = ["schedule a new", "reschedule an existing", "confirm a proposed"]
SENDER_ROLES = [
    "IT manager",
    "operations lead",
    "customer success contact",
    "small business owner",
    "procurement officer",
    "head of engineering",
    "project manager",
    "finance director",
    "support team lead",
    "account executive at the vendor",
]
PRODUCTS = [
    "a project-management SaaS platform",
    "a cloud backup service",
    "a CRM system",
    "an analytics dashboard",
    "a payroll and HR tool",
    "a mobile point-of-sale app",
    "a customer-support helpdesk tool",
    "an e-commerce storefront platform",
]
TIME_STYLES = [
    "a specific weekday and time with a timezone",
    "a relative phrase like 'early next week' or 'Thursday afternoon'",
    "two or three alternative time slots to choose from",
    "a date range and asking the recipient to propose a time",
    "a time in a different timezone from the recipient",
]
DURATIONS = [
    "15 minutes",
    "30 minutes",
    "45 minutes",
    "an hour",
    "90 minutes",
    "no stated duration",
]
PLATFORMS = ["Zoom", "Google Meet", "Microsoft Teams", "a phone call", "an on-site visit", "Webex"]
TONES = ["formal", "friendly and casual", "brief and direct", "slightly impatient", "apologetic"]
LENGTHS = ["two or three sentences", "a short paragraph", "two short paragraphs"]

SYSTEM_PROMPT = (
    "You write realistic inbound emails received by a software company's customer-support "
    "inbox. Write natural, varied English as real people do; do not reuse stock phrases such "
    "as 'I hope this message finds you well'. Invent concrete names, company names and "
    "details. Never use bracketed placeholders like [Name] or <date>. Output JSON only."
)

USER_TEMPLATE = (
    "Write {n} different emails, each about a meeting with the support/customer team.\n"
    "Follow each spec exactly:\n\n"
    "{specs}\n\n"
    'Return a JSON object: {{"emails": [{{"subject": "...", "body": "..."}}, ...]}}\n'
    "with exactly {n} emails, in spec order. The body is the email text only "
    "(greeting, request, sign-off), no headers."
)


@dataclass(frozen=True)
class Spec:
    action: str
    meeting: str
    role: str
    product: str
    time_style: str
    duration: str
    platform: str
    tone: str
    length: str

    def describe(self, i: int) -> str:
        return (
            f"{i}. Sender role: {self.role}, at a customer using {self.product}. "
            f"Purpose: {self.action} {self.meeting}. Timing: {self.time_style}; "
            f"duration: {self.duration}; via {self.platform}. "
            f"Tone: {self.tone}. Length: {self.length}."
        )


def make_specs(n: int = N_ROWS, seed: int = SEED) -> list[Spec]:
    rng = random.Random(seed)
    return [
        Spec(
            action=rng.choice(ACTIONS),
            meeting=rng.choice(MEETING_TYPES),
            role=rng.choice(SENDER_ROLES),
            product=rng.choice(PRODUCTS),
            time_style=rng.choice(TIME_STYLES),
            duration=rng.choice(DURATIONS),
            platform=rng.choice(PLATFORMS),
            tone=rng.choice(TONES),
            length=rng.choice(LENGTHS),
        )
        for _ in range(n)
    ]


def build_prompt(batch: list[Spec]) -> list[dict[str, str]]:
    specs = "\n".join(s.describe(i + 1) for i, s in enumerate(batch))
    user = USER_TEMPLATE.format(n=len(batch), specs=specs)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


_PLACEHOLDER = re.compile(r"\[[A-Z][^\]]{0,30}\]|<[a-z_ ]{2,20}>")


def parse_response(text: str, expected: int) -> list[dict[str, str]]:
    """Parse and validate one batch; raises ValueError on a bad response."""
    data = json.loads(text)
    emails = data.get("emails") if isinstance(data, dict) else None
    if not isinstance(emails, list) or len(emails) != expected:
        raise ValueError(f"expected {expected} emails, got {type(emails).__name__}")
    out = []
    for e in emails:
        subject, body = str(e.get("subject", "")).strip(), str(e.get("body", "")).strip()
        if not subject or len(body) < 40:
            raise ValueError("empty subject or too-short body")
        if _PLACEHOLDER.search(subject + body):
            raise ValueError("bracketed placeholder in output")
        out.append({"subject": subject, "body": body})
    return out


class Cache:
    def __init__(self, root: Path):
        self.root = root

    def key(self, model: str, messages: list[dict[str, str]]) -> str:
        blob = json.dumps({"model": model, "messages": messages}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> str | None:
        p = self.root / f"{key}.json"
        return json.loads(p.read_text())["response"] if p.exists() else None

    def put(self, key: str, response: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{key}.json").write_text(json.dumps({"response": response}))


def generate(
    llm: LLM,
    model: str,
    cache: Cache,
    *,
    n: int = N_ROWS,
    batch_size: int = BATCH_SIZE,
    limit_batches: int | None = None,
    attempts: int = 3,
) -> list[dict]:
    specs = make_specs(n)
    rows: list[dict] = []
    batches = [specs[i : i + batch_size] for i in range(0, n, batch_size)]
    for b_idx, batch in enumerate(batches[:limit_batches]):
        emails = None
        for attempt in range(attempts):
            messages = build_prompt(batch)
            if attempt:  # vary the prompt so a retry gets a fresh cache key
                messages[-1]["content"] += f"\n(Attempt {attempt + 1}.)"
            key = cache.key(model, messages)
            text = cache.get(key)
            if text is None:
                text = llm.complete(messages, temperature=0.9, json_mode=True)
                cache.put(key, text)
            try:
                emails = parse_response(text, len(batch))
                break
            except (ValueError, json.JSONDecodeError) as exc:
                logger.warning("batch %d attempt %d rejected: %s", b_idx, attempt + 1, exc)
        if emails is None:
            logger.error("batch %d skipped after %d attempts", b_idx, attempts)
            continue
        for j, (spec, email) in enumerate(zip(batch, emails, strict=True)):
            rows.append(
                {
                    "id": f"syn-meet-{b_idx * batch_size + j:04d}",
                    "source_dataset": "synthetic_meetings",
                    "origin": "synthetic",
                    "subject": email["subject"],
                    "body": email["body"],
                    "category": "Meeting Scheduling",
                    "urgency": None,
                    "spec": asdict(spec),
                }
            )
        logger.info("batch %d/%d ok (%d rows)", b_idx + 1, len(batches), len(rows))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=None, help="only the first N batches")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    llm = build_llm(settings, "draft")
    try:
        rows = generate(llm, settings.groq_model, Cache(CACHE_DIR), limit_batches=args.limit)
    except LLMError as exc:
        logger.error("generation stopped: %s (cached batches are kept; re-run to resume)", exc)
        return 1
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("wrote %d rows to %s; token usage: %s", len(rows), OUT_PATH, llm.usage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

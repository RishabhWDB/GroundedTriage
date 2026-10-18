"""Hard stress-test set for Meeting Scheduling (P2.1i). **Never used for training.**

Two kinds of ~50 emails each, generated with gpt-oss-120b using a different
seed and prompt from data.synthetic_meetings:

- embedded_meeting: the email's main purpose is to set up, move or confirm a
  meeting, but it is wrapped in a technical or billing problem
  (intended label: Meeting Scheduling);
- call_mention: an ordinary technical / billing / feature / general ticket that
  mentions calls, demos, Zoom or meetings in passing, without asking for one
  (intended label: that category).

The labels are *intended* labels and need a human check: every row is written
with needs_review=true, and evaluation code ignores the set until every row
has been reviewed (Arya edits `category`, sets needs_review=false and may add
reviewer_notes). The CSV is committed and is the source of truth once
reviewed, so this script refuses to overwrite it without --force.

    python -m data.stress_meetings            # == make stress-set
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from app.fileio import atomic_path
from app.llm import LLM, LLMError, build_llm
from app.settings import REPO_ROOT, get_settings
from data.synthetic_meetings import Cache, parse_response

logger = logging.getLogger(__name__)

SEED = 1337  # the training generator uses 42
BATCH_SIZE = 5
OUT_PATH = REPO_ROOT / "datasets" / "stress" / "meeting_stress.csv"
CACHE_DIR = REPO_ROOT / "datasets" / "raw" / "llm_cache" / "stress_meetings"
FIELDS = [
    "id",
    "stress_type",
    "intended_category",
    "category",
    "needs_review",
    "reviewer_notes",
    "subject",
    "body",
]

# (stress_type, intended category, context, count)
PLAN = [
    ("embedded_meeting", "Meeting Scheduling", "Technical Issue", 25),
    ("embedded_meeting", "Meeting Scheduling", "Billing", 25),
    ("call_mention", "Technical Issue", "Technical Issue", 15),
    ("call_mention", "Billing", "Billing", 12),
    ("call_mention", "Feature Request", "Feature Request", 12),
    ("call_mention", "General Inquiry", "General Inquiry", 11),
]

TECH_PROBLEMS = [
    "data sync between two systems silently failing",
    "API returning intermittent 502 errors",
    "SSO login loop after an identity-provider change",
    "a mobile app crash on startup after the latest update",
    "slow report generation that times out",
    "webhooks not firing for some events",
    "the video-call integration dropping recordings",
    "a failed data import with encoding errors",
]
BILLING_PROBLEMS = [
    "a double charge on the latest invoice",
    "being billed for seats that were removed",
    "a refund that has not arrived",
    "an unexpected price increase at renewal",
    "VAT applied incorrectly on invoices",
    "a failed card payment that suspended the account",
    "confusion about prorated charges after a plan change",
]
FEATURE_IDEAS = [
    "a native Zoom or Teams integration",
    "calendar sync for task due dates",
    "the ability to record and attach call notes to tickets",
    "bulk export of meeting transcripts",
    "a scheduling widget for their customers",
    "dark mode and keyboard shortcuts",
]
GENERAL_TOPICS = [
    "whether the product is GDPR compliant",
    "what onboarding resources exist after a demo they already attended",
    "data retention for recorded calls",
    "how support hours work across time zones",
    "whether a nonprofit discount exists",
    "the difference between two plans they saw in a webinar",
]
MENTIONS = [
    "mentions a call they had last week with sales",
    "mentions their team uses Zoom daily",
    "mentions a product demo they watched",
    "mentions they will discuss this in an internal meeting",
    "mentions a webinar recording",
    "mentions their calendar is full this month",
    "mentions a phone call with a previous support agent",
]
MEETING_ASKS = [
    "asks to schedule a call to walk through the problem",
    "asks to reschedule an already booked troubleshooting session",
    "asks to confirm the time of a review meeting about it",
    "asks for a screen-share session with an engineer this week",
    "proposes two time slots for a call with their finance team",
]
TONES = ["formal", "casual", "frustrated", "brief", "detailed"]

SYSTEM_PROMPT = (
    "You generate evaluation emails for a customer-support classifier. Each email must be "
    "realistic and specific: concrete product details, numbers, names and dates. Never use "
    "bracketed placeholders. Vary openings and subjects; avoid stock phrases. Output JSON only."
)
USER_TEMPLATE = (
    "Write {n} inbound support emails. Follow each spec exactly; the PRIMARY PURPOSE in each "
    "spec decides what the email is mainly asking for.\n\n{specs}\n\n"
    'Return a JSON object {{"emails": [{{"subject": "...", "body": "..."}}, ...]}} with exactly '
    "{n} emails in spec order. Body = email text only (greeting, content, sign-off)."
)


@dataclass(frozen=True)
class StressSpec:
    stress_type: str
    intended: str
    context: str
    detail: str
    hook: str
    tone: str

    def describe(self, i: int) -> str:
        if self.stress_type == "embedded_meeting":
            purpose = (
                f"PRIMARY PURPOSE: arrange a meeting. The customer has {self.detail} "
                f"({self.context.lower()} context, described in 2-3 sentences) and {self.hook}. "
                "The scheduling request must be the main ask."
            )
        else:
            purpose = (
                f"PRIMARY PURPOSE: a {self.intended.lower()} about {self.detail}. The email "
                f"{self.hook}, but does NOT ask to schedule, move or confirm any meeting or call."
            )
        return f"{i}. {purpose} Tone: {self.tone}."


def make_specs(seed: int = SEED) -> list[StressSpec]:
    rng = random.Random(seed)
    pools = {
        "Technical Issue": TECH_PROBLEMS,
        "Billing": BILLING_PROBLEMS,
        "Feature Request": FEATURE_IDEAS,
        "General Inquiry": GENERAL_TOPICS,
    }
    specs = []
    for stress_type, intended, context, count in PLAN:
        hooks = MEETING_ASKS if stress_type == "embedded_meeting" else MENTIONS
        for _ in range(count):
            specs.append(
                StressSpec(
                    stress_type,
                    intended,
                    context,
                    rng.choice(pools[context]),
                    rng.choice(hooks),
                    rng.choice(TONES),
                )
            )
    rng.shuffle(specs)
    return specs


def build_prompt(batch: list[StressSpec]) -> list[dict[str, str]]:
    specs = "\n".join(s.describe(i + 1) for i, s in enumerate(batch))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(n=len(batch), specs=specs)},
    ]


def generate(llm: LLM, model: str, cache: Cache, attempts: int = 3) -> list[dict]:
    specs = make_specs()
    rows: list[dict] = []
    for b in range(0, len(specs), BATCH_SIZE):
        batch = specs[b : b + BATCH_SIZE]
        emails = None
        for attempt in range(attempts):
            messages = build_prompt(batch)
            if attempt:
                messages[-1]["content"] += f"\n(Attempt {attempt + 1}.)"
            key = cache.key(model, messages)
            text = cache.get(key)
            if text is None:
                text = llm.complete(messages, temperature=0.9, json_mode=True)
                cache.put(key, text)
            try:
                emails = parse_response(text, len(batch))
                break
            except ValueError as exc:  # json.JSONDecodeError is a ValueError
                logger.warning(
                    "batch %d attempt %d rejected: %s", b // BATCH_SIZE, attempt + 1, exc
                )
        if emails is None:
            logger.error("batch %d skipped", b // BATCH_SIZE)
            continue
        for j, (spec, email) in enumerate(zip(batch, emails, strict=True)):
            rows.append(
                {
                    "id": f"stress-{b + j:03d}",
                    "stress_type": spec.stress_type,
                    "intended_category": spec.intended,
                    "category": spec.intended,
                    "needs_review": "true",
                    "reviewer_notes": "",
                    "subject": email["subject"],
                    "body": email["body"],
                }
            )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    with atomic_path(path) as tmp, open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def load_reviewed(path: Path = OUT_PATH) -> list[dict] | None:
    """The stress rows, or None while any row still needs review (or the file is missing)."""
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if any(r["needs_review"].strip().lower() != "false" for r in rows):
        return None
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true", help="overwrite the (possibly reviewed) CSV")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if OUT_PATH.exists() and not args.force:
        logger.error("%s exists (may hold your review); use --force to overwrite", OUT_PATH)
        return 1
    settings = get_settings()
    llm = build_llm(settings, "draft")
    try:
        rows = generate(llm, settings.groq_model, Cache(CACHE_DIR))
    except LLMError as exc:
        logger.error("generation stopped: %s (cached batches are kept; re-run to resume)", exc)
        return 1
    write_csv(rows, OUT_PATH)
    logger.info("wrote %d rows to %s; token usage: %s", len(rows), OUT_PATH, llm.usage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

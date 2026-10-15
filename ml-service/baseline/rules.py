"""SCRIBE's first stage: regex rules with a fixed confidence cutoff.

SCRIBE accepts a rule decision only when its confidence reaches a fixed ~95%
cutoff; everything else falls through to zero-shot classification. Under the
6-category taxonomy, rules cover the two categories that are pattern-like:
Spam and Meeting Scheduling.

Each rule is one cue with a weight. A category's confidence is the noisy-OR of
its matched cues, 1 - prod(1 - w), so a single strong cue (w >= 0.95) is
accepted on its own, while weaker cues must co-occur to reach the cutoff. The
cue lists use generic spam and scheduling vocabulary. They were written before
any evaluation run and are not tuned on val or test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CUTOFF = 0.95
SPAM = "Spam"
MEETING = "Meeting Scheduling"


@dataclass(frozen=True)
class Rule:
    name: str
    category: str
    weight: float
    pattern: re.Pattern[str]


def _r(name: str, category: str, weight: float, pattern: str) -> Rule:
    return Rule(name, category, weight, re.compile(pattern, re.I | re.S))


_MEET_NOUN = r"(?:meeting|call|demo|session|appointment|catch[- ]up|review|kickoff|kick-off)"

RULES: tuple[Rule, ...] = (
    # --- Spam ---
    _r("pharma_casino", SPAM, 0.95, r"\b(?:viagra|cialis|online pharmacy|casino|xanax)\b"),
    _r(
        "advance_fee",
        SPAM,
        0.95,
        r"\b(?:next of kin|beneficiary of|million (?:us )?dollars|wire transfer of)\b",
    ),
    _r(
        "bulk_disclaimer",
        SPAM,
        0.9,
        r"\b(?:this is not spam|you are receiving this (?:e-?mail|message) because"
        r"|to be removed from (?:our|this) (?:mailing )?list)\b",
    ),
    _r(
        "make_money",
        SPAM,
        0.9,
        r"\b(?:make money|work from home|extra income|financial freedom|earn \$?\d)",
    ),
    _r("guarantee", SPAM, 0.85, r"\b100% (?:free|guaranteed|satisfaction)\b"),
    _r("prize", SPAM, 0.8, r"\b(?:lottery|jackpot|you(?:'ve| have) (?:been selected|won))\b"),
    _r("unsubscribe", SPAM, 0.7, r"\b(?:unsubscribe|opt[- ]out)\b"),
    _r("click_here", SPAM, 0.6, r"\bclick here\b"),
    _r(
        "urgency_sales",
        SPAM,
        0.6,
        r"\b(?:act now|limited time|order now|buy now|risk[- ]free|special promotion)\b",
    ),
    _r("loans_rates", SPAM, 0.6, r"\b(?:lowest rates?|mortgage|refinanc\w*|credit card offers?)\b"),
    _r("money_shouting", SPAM, 0.5, r"\$\$\$|!!!|\bfree!"),
    # --- Meeting Scheduling ---
    _r("reschedule", MEETING, 0.85, r"\breschedul\w*"),
    _r("invite", MEETING, 0.85, r"\b(?:calendar|meeting) (?:invite|invitation)\b"),
    _r(
        "schedule_meeting",
        MEETING,
        0.8,
        rf"\b(?:(?:re)?schedule|set up|arrange|book)\b.{{0,40}}\b{_MEET_NOUN}\b",
    ),
    _r(
        "confirm_meeting",
        MEETING,
        0.8,
        rf"\bconfirm\w*\b.{{0,40}}\b{_MEET_NOUN}\b",
    ),
    _r(
        "availability",
        MEETING,
        0.6,
        r"\b(?:availability|time slots?|slot that works|works for you"
        r"|are you (?:free|available))\b",
    ),
    _r("platform", MEETING, 0.5, r"\b(?:zoom|microsoft teams|google meet|webex)\b"),
)


@dataclass(frozen=True)
class RuleMatch:
    category: str
    confidence: float
    rules: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return self.confidence >= CUTOFF


def score(subject: str, body: str, rules: tuple[Rule, ...] = RULES) -> list[RuleMatch]:
    """Per-category noisy-OR confidence of the matched cues, best first."""
    text = f"{subject or ''}\n{body or ''}"
    hits: dict[str, list[Rule]] = {}
    for rule in rules:
        if rule.pattern.search(text):
            hits.setdefault(rule.category, []).append(rule)
    out = []
    for category, matched in hits.items():
        miss = 1.0
        for rule in matched:
            miss *= 1.0 - rule.weight
        out.append(RuleMatch(category, round(1.0 - miss, 6), tuple(r.name for r in matched)))
    return sorted(out, key=lambda m: -m.confidence)


def apply_rules(subject: str, body: str) -> RuleMatch | None:
    """The best rule match, or None when no cue matches. Callers accept it only
    if ``match.accepted`` (confidence >= CUTOFF)."""
    matches = score(subject, body)
    return matches[0] if matches else None

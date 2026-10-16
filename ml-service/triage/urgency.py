"""Urgency (P3.3): the fine-tuned ordinal model from the GPU PC (inference
only) plus a documented keyword heuristic used when the model is absent.

Levels: low < medium < high < critical. The model predicts low/medium/high
(the English data has no critical rows); **critical comes only from the
heuristic's critical rule** (an impact cue plus urgency/escalation), applied
on top of either source.

Heuristic scoring (each group counts once per ticket, however many cues match):

| group | examples | points |
|---|---|---|
| urgency keywords | urgent, ASAP, immediately, emergency, critical | +3 |
| deadline / time expressions | today, tonight, by EOD, within 2 hours, by Friday, deadline | +1 |
| escalation language | escalate, manager, unacceptable, lawyer, chargeback, 3rd time | +1 |
| negative phrasing | frustrated, furious, terrible, worst, extremely disappointed, "!!" | +1 |
| business impact | outage, production down, all users, data loss, security breach | +2 |
| low-urgency cues | no rush, whenever, not urgent, at your convenience, just curious, FYI | -2 |

score <= -1 -> low; 0..2 -> medium; >= 3 -> high (an explicit urgency
keyword alone is enough for high).
critical: business impact AND (urgency keywords OR escalation).
Confidence is a fixed, uncalibrated function of the number of groups that
fired (0.4 with none, +0.1 per group, max 0.8), reported as such.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.settings import get_settings
from triage.hf_model import HFClassifier

TASK = "urgency"
TRAINED_LEVELS = ["low", "medium", "high"]
SCALE = ["low", "medium", "high", "critical"]


def _rx(*patterns: str) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.I)


GROUPS: dict[str, tuple[re.Pattern, int]] = {
    "urgency": (
        _rx(
            r"\burgent(?:ly)?\b",
            r"\basap\b",
            r"\bimmediate(?:ly)?\b",
            r"\bemergency\b",
            r"\bcritical\b",
            r"\bright away\b",
            r"\bas soon as possible\b",
            r"\btime[- ]sensitive\b",
        ),
        3,
    ),
    "deadline": (
        _rx(
            r"\btoday\b",
            r"\btonight\b",
            r"\bby (?:eod|end of (?:the )?day|tomorrow|noon)\b",
            r"\bwithin (?:the next )?\d+ ?(?:minutes?|mins?|hours?|hrs?)\b",
            r"\bby (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            r"\bdeadline\b",
            r"\bbefore \d{1,2}(?::\d{2})? ?(?:am|pm)\b",
            r"\bthis morning\b",
        ),
        1,
    ),
    "escalation": (
        _rx(
            r"\bescalat\w*",
            r"\b(?:your|a) (?:manager|supervisor)\b",
            r"\bunacceptable\b",
            r"\blawyer\b",
            r"\blegal action\b",
            r"\bchargeback\b",
            r"\b(?:third|3rd|fourth|several|multiple) times?\b",
            r"\bstill (?:not|isn't|hasn't|no) (?:been )?(?:fixed|resolved|working|response)\b",
            r"\bcancel (?:my|our) (?:account|subscription|contract)\b",
        ),
        1,
    ),
    "negative": (
        _rx(
            r"\bfrustrat\w*",
            r"\bfurious\b",
            r"\bangry\b",
            r"\bterrible\b",
            r"\bawful\b",
            r"\bworst\b",
            r"\b(?:very|extremely|deeply) disappointed\b",
            r"\bridiculous\b",
            r"!{2,}",
        ),
        1,
    ),
    "impact": (
        _rx(
            r"\boutage\b",
            r"\b(?:production|prod|site|server|system|service|website) (?:is )?"
            r"(?:down|offline|unavailable)\b",
            r"\ball (?:our )?(?:users|customers|employees)\b",
            r"\bdata loss\b",
            r"\b(?:lost|losing) (?:all )?(?:data|revenue|sales|money)\b",
            r"\bsecurity (?:breach|incident)\b",
            r"\b(?:hacked|compromised)\b",
            r"\b(?:can't|cannot|unable to) (?:work|operate|process (?:any )?orders)\b",
        ),
        2,
    ),
    "low": (
        _rx(
            r"\bno rush\b",
            r"\bwhenever\b",
            r"\bnot urgent\b",
            r"\bat your convenience\b",
            r"\bjust (?:curious|wondering)\b",
            r"\bfyi\b",
            r"\bno hurry\b",
            r"\blow priority\b",
        ),
        -2,
    ),
}


@dataclass
class UrgencyResult:
    urgency: str
    confidence: float
    source: str  # "model" | "heuristic"
    signals: list[str] = field(default_factory=list)


def heuristic_urgency(text: str) -> UrgencyResult:
    fired = [name for name, (rx, _) in GROUPS.items() if rx.search(text or "")]
    score = sum(GROUPS[name][1] for name in fired)
    if "impact" in fired and ("urgency" in fired or "escalation" in fired):
        level = "critical"
    elif score >= 3:
        level = "high"
    elif score <= -1:
        level = "low"
    else:
        level = "medium"
    confidence = min(0.8, 0.4 + 0.1 * len(fired))
    return UrgencyResult(level, round(confidence, 2), "heuristic", fired)


def is_critical(text: str) -> bool:
    return heuristic_urgency(text).urgency == "critical"


def check_urgency_labels(labels: list[str]) -> str | None:
    if labels not in (TRAINED_LEVELS, SCALE):
        return (
            f"labels must be {TRAINED_LEVELS} (optionally + 'critical') in id order, got {labels}"
        )
    return None


class UrgencyModel(HFClassifier):
    def __init__(self, directory: Path | None = None, **kw):
        s = get_settings()
        kw.setdefault("fp16", s.model_fp16)
        super().__init__(directory or s.models_dir / "urgency", TASK, check_urgency_labels, **kw)


class UrgencyPredictor:
    """Model when available, else the heuristic; the critical rule applies to both."""

    def __init__(self, model: UrgencyModel | None):
        self.model = model

    @property
    def model_available(self) -> bool:
        return self.model is not None and self.model.available

    def predict(self, texts: list[str]) -> list[UrgencyResult]:
        heur = [heuristic_urgency(t) for t in texts]
        if not texts or not self.model_available:
            return heur
        try:
            probs = self.model.predict_proba(texts)
        except RuntimeError:  # failed to load: fall back for this and later calls
            return heur
        out = []
        for h, row in zip(heur, probs, strict=True):
            i = int(np.argmax(row))
            if h.urgency == "critical":
                out.append(h)
            else:
                out.append(UrgencyResult(self.model.labels[i], float(row[i]), "model"))
        return out

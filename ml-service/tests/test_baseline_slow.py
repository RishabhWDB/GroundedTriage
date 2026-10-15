"""End-to-end SCRIBE baseline on 10 sample emails (downloads ~2 GB of weights
on first run). Opt-in: RUN_SLOW=1. The drafting test also needs GROQ_API_KEY."""

import os

import pytest

from app.settings import REPO_ROOT, get_settings
from baseline.__main__ import load_tickets

SAMPLE_CSV = str(REPO_ROOT / "datasets" / "sample_emails.csv")

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def run():
    pytest.importorskip("transformers")
    from baseline.scribe import ScribeClassifier

    tickets, gold = load_tickets(None, SAMPLE_CSV, 10)
    clf = ScribeClassifier()
    return tickets, gold, clf.classify(tickets), clf


def test_classify_ten_samples_end_to_end(run):
    tickets, gold, preds, clf = run
    assert len(preds) == 10 and len(gold) == 10
    assert {p.stage for p in preds} <= {"rule", "zero_shot"}
    assert all(p.urgency in {"low", "medium", "high"} for p in preds)
    assert all(p.signals["sentiment"] in {-1.0, 0.0, 1.0} for p in preds)
    assert clf.stats.by_zero_shot >= 1
    spam = [p for g, p in zip(gold, preds, strict=True) if g == "Spam"]
    assert spam and all(p.category == "Spam" for p in spam)  # the prize email


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY unset")
def test_draft_ten_samples_end_to_end(run):
    from app.llm import build_llm
    from baseline.scribe import ScribeDrafter

    tickets, _, preds, _ = run
    llm = build_llm(get_settings(), "draft")
    drafter = ScribeDrafter(llm)
    for t, p in zip(tickets, preds, strict=True):
        d = drafter.draft(t, p)
        assert d.body and len(d.stages) == 2 and d.citations == [] and not d.grounded
    assert llm.usage["calls"] == 20

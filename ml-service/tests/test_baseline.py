import pytest

from app.interfaces import CategoryPrediction, TicketText
from app.llm import FakeLLM
from baseline.rules import CUTOFF, RULES, apply_rules, score
from baseline.scribe import ScribeClassifier, ScribeDrafter


@pytest.mark.parametrize(
    "subject,body,category",
    [
        ("You have won!", "Claim your lottery prize: click here, act now!!!", "Spam"),
        ("Cheap meds", "Buy viagra from our online pharmacy today", "Spam"),
        ("Re: inheritance", "I am the next of kin and need a wire transfer of funds", "Spam"),
        ("Make money", "Work from home and earn $5000 a week. To unsubscribe reply STOP", "Spam"),
        (
            "Can we reschedule?",
            "Could we reschedule our onboarding call to Thursday? Let me know a slot that "
            "works for you.",
            "Meeting Scheduling",
        ),
        (
            "Demo next week",
            "I'd like to schedule a product demo on Zoom, please send a calendar invite.",
            "Meeting Scheduling",
        ),
    ],
)
def test_rules_accept_clear_cases(subject, body, category):
    m = apply_rules(subject, body)
    assert m is not None and m.category == category and m.accepted


@pytest.mark.parametrize(
    "subject,body",
    [
        ("Login broken", "I cannot log in after the update; the page shows error 500."),
        ("Invoice question", "We were charged twice for the October invoice, please refund."),
        ("Export to Excel", "Please add an option to export reports to xlsx."),
    ],
)
def test_rules_ignore_ordinary_tickets(subject, body):
    assert apply_rules(subject, body) is None


def test_single_weak_cue_falls_below_cutoff():
    # One meeting cue in a technical ticket: matched, but not accepted.
    m = apply_rules("Sync failing", "The Zoom integration stopped syncing recordings yesterday.")
    assert m is not None and m.category == "Meeting Scheduling"
    assert m.confidence == pytest.approx(0.5) and not m.accepted


def test_noisy_or_combines_cues():
    m = score("", "Can we reschedule? Here is a time slot: Zoom at 3pm")[0]
    expected = 1 - (1 - 0.85) * (1 - 0.6) * (1 - 0.5)
    assert m.confidence == pytest.approx(expected) and m.accepted
    assert set(m.rules) == {"reschedule", "availability", "platform"}


def test_rule_weights_and_cutoff():
    assert CUTOFF == 0.95
    assert all(0 < r.weight <= CUTOFF for r in RULES)
    assert {r.category for r in RULES} == {"Spam", "Meeting Scheduling"}


class FakeZeroShot:
    def __init__(self):
        self.seen = []

    def predict(self, texts):
        self.seen.extend(texts)
        return [("Billing", 0.7)] * len(texts)


class FakeSentiment:
    def predict(self, texts):
        return [-1 if "angry" in t else 1 for t in texts]


def test_classifier_rules_then_zero_shot():
    zs = FakeZeroShot()
    clf = ScribeClassifier(zs, FakeSentiment())
    tickets = [
        TicketText(id="a", subject="Cheap meds", body="viagra online pharmacy"),
        TicketText(id="b", subject="Invoice", body="I am angry about this double charge"),
    ]
    a, b = clf.classify(tickets)
    assert (a.category, a.stage, a.urgency) == ("Spam", "rule", "low")
    assert (b.category, b.stage, b.urgency) == ("Billing", "zero_shot", "high")
    assert b.signals["sentiment"] == -1.0 and "rule:pharma_casino" in a.signals
    assert zs.seen == ["Invoice\n\nI am angry about this double charge"]  # only the fallback
    assert clf.stats.by_rule == 1 and clf.stats.by_zero_shot == 1 and clf.stats.tickets == 2


def test_classifier_empty_batch():
    assert ScribeClassifier(FakeZeroShot(), FakeSentiment()).classify([]) == []


def test_drafter_two_passes_ungrounded():
    llm = FakeLLM(["first draft", "refined reply"])
    t = TicketText(id="x", subject="Refund", body="Please refund my duplicate charge.")
    p = CategoryPrediction(category="Billing", confidence=0.8, stage="zero_shot", signals={})
    d = ScribeDrafter(llm).draft(t, p)
    assert d.body == "refined reply" and d.stages == ["first draft", "refined reply"]
    assert d.citations == [] and d.grounded is False
    assert len(llm.calls) == 2
    assert "Billing" in llm.calls[0]["messages"][1]["content"]
    assert "first draft" in llm.calls[1]["messages"][1]["content"]

import json

from app.llm import FakeLLM
from data import synthetic_meetings
from data.stress_meetings import (
    PLAN,
    SEED,
    SYSTEM_PROMPT,
    Cache,
    build_prompt,
    generate,
    load_reviewed,
    make_specs,
    write_csv,
)


def emails_json(n):
    body = "Our sync has failed since Monday; could we book a call on Tuesday at 3pm CET?"
    return json.dumps({"emails": [{"subject": f"Sync issue {i}", "body": body} for i in range(n)]})


def test_specs_follow_plan_and_differ_from_training_generator():
    specs = make_specs()
    assert len(specs) == sum(p[3] for p in PLAN) == 100
    assert sum(s.stress_type == "embedded_meeting" for s in specs) == 50
    assert make_specs() == specs
    assert SEED != synthetic_meetings.SEED and SYSTEM_PROMPT != synthetic_meetings.SYSTEM_PROMPT
    neg = next(s for s in specs if s.stress_type == "call_mention")
    assert "does NOT ask to schedule" in build_prompt([neg])[-1]["content"]


def test_generate_marks_every_row_for_review(tmp_path):
    rows = generate(FakeLLM(responder=lambda m: emails_json(5)), "m", Cache(tmp_path))
    assert len(rows) == 100 and all(r["needs_review"] == "true" for r in rows)
    assert all(r["category"] == r["intended_category"] for r in rows)


def test_load_reviewed_gates_on_review(tmp_path):
    path = tmp_path / "s.csv"
    assert load_reviewed(path) is None
    row = {
        "id": "stress-000",
        "stress_type": "call_mention",
        "intended_category": "Billing",
        "category": "Billing",
        "needs_review": "true",
        "reviewer_notes": "",
        "subject": "s",
        "body": "b",
    }
    write_csv([row], path)
    assert load_reviewed(path) is None
    write_csv([{**row, "needs_review": "false", "category": "General Inquiry"}], path)
    assert load_reviewed(path)[0]["category"] == "General Inquiry"

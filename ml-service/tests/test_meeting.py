import json
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.llm import FakeLLM, LLMError
from app.main import app
from app.meeting import MeetingRequest, build_messages, extract, validate
from app.models import Registry, get_registry
from app.settings import Settings

RECEIVED = datetime(2026, 10, 13, 9, 30, tzinfo=UTC)  # a Tuesday


def req(body="Can we meet Thursday at 3pm for 30 minutes on Zoom?", tz="Europe/London"):
    return MeetingRequest(ticket_id="t1", subject="Meeting", body=body, received_at=RECEIVED,
                          default_timezone=tz)  # fmt: skip


def test_prompt_states_received_time_and_default_zone():
    user = build_messages(req())[1]["content"]
    assert "2026-10-13T10:30 (Tuesday)" in user  # 09:30 UTC = 10:30 BST
    assert "default timezone Europe/London" in user


def test_extract_happy_path():
    raw = {"is_meeting_request": True, "start": "2026-10-15T15:00", "timezone": None,
           "duration_minutes": 30, "venue": "video", "venue_detail": "Zoom",
           "time_text": "Thursday at 3pm", "action": "schedule"}  # fmt: skip
    llm = FakeLLM([json.dumps(raw)])
    m = extract(req(), llm)
    assert m.is_meeting_request and m.timezone == "Europe/London"
    assert m.start.isoformat() == "2026-10-15T15:00:00+01:00"
    assert (m.duration_minutes, m.venue, m.venue_detail, m.action) == (
        30,
        "video",
        "Zoom",
        "schedule",
    )
    assert m.notes == [] and llm.calls[0]["json_mode"] and llm.calls[0]["temperature"] == 0.0


NY = "2026-10-15T09:00:00-04:00"


@pytest.mark.parametrize(
    "raw,field,expected,note",
    [
        ({"start": "2026-10-01T10:00"}, "start", None, "before the email"),
        ({"start": "2028-01-01T10:00"}, "start", None, "year ahead"),
        ({"start": "next thursday"}, "start", None, "unparseable"),
        ({"duration_minutes": 900}, "duration_minutes", None, "duration"),
        ({"duration_minutes": "half an hour"}, "duration_minutes", None, None),
        ({"timezone": "Mars/Base"}, "timezone", "Europe/London", "unknown timezone"),
        ({"start": "2026-10-15T09:00", "timezone": "America/New_York"}, "start", NY, None),
        ({"venue": "spaceship"}, "venue", "unspecified", None),
        ({"action": "teleport"}, "action", "other", None),
        ({}, "is_meeting_request", False, None),
    ],
)
def test_validate_drops_unusable_fields(raw, field, expected, note):
    m = validate(raw, req())
    got = getattr(m, field)
    assert (got.isoformat() if isinstance(got, datetime) else got) == expected
    assert (note in m.notes[0]) if note else m.notes == []


def test_extract_retries_bad_json_then_fails():
    m = extract(req(), FakeLLM(["not json", json.dumps({"is_meeting_request": True})]))
    assert m.is_meeting_request
    with pytest.raises(LLMError):
        extract(req(), FakeLLM(["nope", "[1, 2]"]))


def test_endpoint(tmp_path):
    reg = Registry(Settings(models_dir=tmp_path, groq_api_key=""))
    app.dependency_overrides[get_registry] = lambda: reg
    try:
        body = {"ticket_id": "t1", "subject": "s", "body": "b", "received_at": RECEIVED.isoformat()}
        client = TestClient(app)
        assert client.post("/extract_meeting", json=body).status_code == 503
        reg._llm = FakeLLM([json.dumps({"is_meeting_request": True, "start": "2026-10-14T11:00"})])
        r = client.post("/extract_meeting", json=body)
        assert r.status_code == 200 and r.json()["start"].startswith("2026-10-14T11:00:00")
        reg._llm = FakeLLM(["x", "y"])
        assert client.post("/extract_meeting", json=body).status_code == 502
    finally:
        app.dependency_overrides.clear()

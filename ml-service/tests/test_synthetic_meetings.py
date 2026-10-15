import json

import pytest

from app.llm import FakeLLM
from data.synthetic_meetings import (
    Cache,
    build_prompt,
    generate,
    make_specs,
    parse_response,
)


def emails_json(
    n, body="Could we set up a 30-minute Zoom call on Tuesday at 3pm CET to go through onboarding?"
):
    return json.dumps(
        {"emails": [{"subject": f"Onboarding call {i}", "body": body} for i in range(n)]}
    )


def test_specs_are_deterministic_and_varied():
    a, b = make_specs(400), make_specs(400)
    assert a == b
    assert len({s.meeting for s in a}) >= 10 and len({s.platform for s in a}) >= 5
    assert make_specs(400, seed=7) != a


def test_prompt_contains_each_spec():
    specs = make_specs(3)
    msgs = build_prompt(specs)
    user = msgs[-1]["content"]
    assert msgs[0]["role"] == "system" and "Write 3 different emails" in user
    for s in specs:
        assert s.meeting in user and s.platform in user


def test_parse_response_validates():
    assert len(parse_response(emails_json(2), 2)) == 2
    with pytest.raises(ValueError, match="expected 3"):
        parse_response(emails_json(2), 3)
    with pytest.raises(ValueError, match="placeholder"):
        parse_response(
            emails_json(
                1, body="Hi team, please book a call with [Your Name] next week, thanks a lot."
            ),
            1,
        )
    with pytest.raises(json.JSONDecodeError):
        parse_response("not json", 1)


def test_generate_uses_cache_and_retries(tmp_path):
    cache = Cache(tmp_path)
    fake = FakeLLM(["garbage", emails_json(5), emails_json(5)])
    rows = generate(fake, "m", cache, n=10, batch_size=5)
    assert len(rows) == 10 and len(fake.calls) == 3  # batch 1 retried once after bad JSON
    assert rows[0]["id"] == "syn-meet-0000" and rows[-1]["id"] == "syn-meet-0009"
    assert {r["origin"] for r in rows} == {"synthetic"}
    assert {r["category"] for r in rows} == {"Meeting Scheduling"}
    assert all(c["json_mode"] for c in fake.calls)

    again = FakeLLM([])  # every batch is now cached: no LLM calls
    assert generate(again, "m", cache, n=10, batch_size=5) == rows
    assert again.calls == []

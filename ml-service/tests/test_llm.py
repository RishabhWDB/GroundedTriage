import json

import httpx
import pytest

from app.llm import FakeLLM, GroqLLM, JSONModeError, LLMError, build_llm
from app.settings import Settings

OK_BODY = {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}


def make_llm(handler, sleeps, max_retries=3):
    return GroqLLM(
        "key",
        "test-model",
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        max_retries=max_retries,
    )


def test_retries_429_then_succeeds_honouring_retry_after():
    calls, sleeps = [], []

    def handler(req):
        calls.append(json.loads(req.content))
        if len(calls) < 3:
            return httpx.Response(429, headers={"retry-after": "2"})
        return httpx.Response(200, json=OK_BODY)

    llm = make_llm(handler, sleeps)
    assert llm.complete([{"role": "user", "content": "hi"}], json_mode=True) == "hello"
    assert len(calls) == 3
    assert sleeps == [2.0, 2.0]
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["model"] == "test-model"
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["max_tokens"] == 4096
    assert "reasoning_effort" not in calls[0]


def test_exponential_backoff_on_5xx_and_transport_errors():
    sleeps, n = [], {"i": 0}

    def handler(req):
        n["i"] += 1
        if n["i"] == 1:
            raise httpx.ConnectError("boom")
        if n["i"] == 2:
            return httpx.Response(503)
        return httpx.Response(200, json=OK_BODY)

    assert make_llm(handler, sleeps).complete([]) == "hello"
    assert len(sleeps) == 2
    assert 0.5 <= sleeps[0] <= 1.0 and 1.0 <= sleeps[1] <= 2.0


def test_gives_up_after_max_retries():
    sleeps = []
    llm = make_llm(lambda req: httpx.Response(429), sleeps, max_retries=2)
    with pytest.raises(LLMError, match="giving up after 3 attempts"):
        llm.complete([])
    assert len(sleeps) == 2


def test_non_retryable_error_raises_immediately():
    sleeps = []
    llm = make_llm(lambda req: httpx.Response(400, text="bad request"), sleeps)
    with pytest.raises(LLMError, match="HTTP 400"):
        llm.complete([])
    assert sleeps == []


def test_json_mode_failure_is_its_own_error():
    body = '{"error":{"code":"json_validate_failed","message":"Failed to generate JSON"}}'
    llm = make_llm(lambda req: httpx.Response(400, text=body), [])
    with pytest.raises(JSONModeError):
        llm.complete([], json_mode=True)


def test_missing_key_raises():
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        GroqLLM("", "m")


def test_fake_llm_scripted_and_responder():
    fake = FakeLLM(["a", "b"])
    assert fake.complete([{"role": "user", "content": "x"}]) == "a"
    assert fake.complete([], json_mode=True) == "b"
    assert fake.calls[1]["json_mode"] is True
    with pytest.raises(LLMError):
        fake.complete([])
    echo = FakeLLM(responder=lambda msgs: msgs[-1]["content"].upper())
    assert echo.complete([{"role": "user", "content": "hi"}]) == "HI"


def test_reasoning_effort_usage_and_truncation():
    sent = []
    bodies = [
        {
            "choices": [
                {"message": {"content": "{}", "reasoning": "..."}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 30,
                "total_tokens": 40,
                "completion_tokens_details": {"reasoning_tokens": 25},
            },
        },
        {
            "choices": [{"message": {"content": '{"a": '}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
        },
    ]

    def handler(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, json=bodies[len(sent) - 1])

    llm = GroqLLM(
        "key", "m", transport=httpx.MockTransport(handler), reasoning_effort="low", max_tokens=50
    )
    assert llm.complete([], json_mode=True) == "{}"
    assert sent[0]["reasoning_effort"] == "low" and sent[0]["max_tokens"] == 50
    with pytest.raises(LLMError, match="truncated at max_tokens=50"):
        llm.complete([])
    assert llm.usage == {
        "calls": 2,
        "prompt_tokens": 20,
        "completion_tokens": 80,
        "reasoning_tokens": 25,
        "total_tokens": 100,
    }


def test_build_llm_roles(monkeypatch):
    for field in Settings.model_fields:  # ignore the developer's .env / environment
        monkeypatch.delenv(field.upper(), raising=False)
    s = Settings(_env_file=None, groq_api_key="gk")
    draft, judge = build_llm(s, "draft"), build_llm(s, "judge")
    assert (draft.model, draft.reasoning_effort) == ("openai/gpt-oss-120b", "low")
    assert (judge.model, judge.reasoning_effort) == ("qwen/qwen3.8-27b", None)
    assert judge.max_tokens == 1000  # Groq's qwen OTPM limit
    assert str(judge._client.base_url).startswith("https://api.groq.com/openai/v1")
    with pytest.raises(ValueError):
        build_llm(s, "other")

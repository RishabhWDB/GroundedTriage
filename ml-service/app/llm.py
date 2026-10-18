"""LLM access: an ``LLM`` protocol, the Groq implementation and a ``FakeLLM`` for tests.

Groq is called via its OpenAI-compatible REST API with httpx (any compatible
provider works via ``base_url``). 429 and 5xx responses and transport errors are
retried with exponential backoff (honouring Retry-After). Reasoning models
(gpt-oss, qwen3) take ``reasoning_effort``; their reasoning comes back in a
separate field, so ``content`` stays clean. A ``finish_reason`` of "length" is an
error rather than a silently truncated answer. Token usage is logged per call and
accumulated in ``usage``. An eval-time response cache comes in P7.2.
"""

import json
import logging
import random
import re
import time
from collections.abc import Callable, Sequence
from typing import Protocol

import httpx

from app.fileio import atomic_write_text

logger = logging.getLogger(__name__)

Message = dict[str, str]


class LLMError(RuntimeError):
    pass


class JSONModeError(LLMError):
    """Groq JSON mode failed server-side (HTTP 400 json_validate_failed), e.g.
    the output was cut off at max_tokens. A bad attempt: callers retry."""


class TruncatedError(LLMError):
    """The output hit max_tokens (finish_reason=length)."""


class LLM(Protocol):
    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str: ...


def _usage_counts(usage: dict) -> dict[str, int]:
    details = usage.get("completion_tokens_details") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "reasoning_tokens": int(details.get("reasoning_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }


RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class GroqLLM:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.groq.com/openai/v1",
        *,
        reasoning_effort: str | None = None,
        max_tokens: int = 4096,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set")
        self.model = model
        self.reasoning_effort = reasoning_effort or None
        self.max_tokens = max_tokens
        self.usage = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._sleep = sleep
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,
        )

    def _delay(self, attempt: int, resp: httpx.Response | None) -> float:
        if resp is not None:
            retry_after = resp.headers.get("retry-after")
            if retry_after:
                try:
                    return min(float(retry_after), self.max_delay)
                except ValueError:
                    pass
        backoff = self.base_delay * (2**attempt)
        return min(backoff, self.max_delay) * (0.5 + random.random() / 2)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        payload: dict = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        last_error = ""
        for attempt in range(self.max_retries + 1):
            resp = None
            try:
                resp = self._client.post("/chat/completions", json=payload)
            except httpx.TransportError as exc:
                last_error = f"transport error: {exc}"
            else:
                if resp.status_code == 200:
                    return self._parse(resp.json(), payload["max_tokens"])
                # 500 chars keep Groq's limit name ("tokens per day (TPD)"), which
                # callers use to tell the daily cap from per-minute limits.
                body = re.sub(r"org_[A-Za-z0-9]+", "org_…", resp.text[:500])
                last_error = f"HTTP {resp.status_code}: {body}"
                if resp.status_code == 400 and "json_validate_failed" in resp.text:
                    raise JSONModeError(last_error)
                if resp.status_code not in RETRYABLE_STATUS:
                    raise LLMError(last_error)
            if attempt < self.max_retries:
                self._sleep(self._delay(attempt, resp))
        raise LLMError(f"giving up after {self.max_retries + 1} attempts; last: {last_error}")

    def _parse(self, body: dict, max_tokens: int) -> str:
        counts = _usage_counts(body.get("usage") or {})
        self.usage["calls"] += 1
        for k, v in counts.items():
            self.usage[k] += v
        logger.info(
            "llm usage model=%s prompt=%d completion=%d reasoning=%d total=%d",
            self.model,
            counts["prompt_tokens"],
            counts["completion_tokens"],
            counts["reasoning_tokens"],
            counts["total_tokens"],
        )
        choice = body["choices"][0]
        if choice.get("finish_reason") == "length":
            raise TruncatedError(
                f"{self.model}: output truncated at max_tokens={max_tokens} "
                f"(reasoning used {counts['reasoning_tokens']}); raise LLM_MAX_TOKENS"
            )
        return choice["message"].get("content") or ""


def build_llm(settings, role: str = "draft") -> GroqLLM:
    """Build the drafter (``role="draft"``) or the faithfulness judge (``"judge"``)."""
    if role == "draft":
        return GroqLLM(
            settings.groq_api_key,
            settings.groq_model,
            settings.groq_base_url,
            reasoning_effort=settings.draft_reasoning_effort,
            max_tokens=settings.llm_max_tokens,
        )
    if role == "judge":
        return GroqLLM(
            settings.judge_api_key or settings.groq_api_key,
            settings.judge_model,
            settings.judge_base_url or settings.groq_base_url,
            reasoning_effort=settings.judge_reasoning_effort,
            max_tokens=settings.judge_max_tokens,
        )
    raise ValueError(f"unknown LLM role {role!r}")


class FakeLLM:
    """Deterministic LLM for tests: returns scripted responses in order, or the
    result of ``responder(messages)``. Records every call in ``calls``."""

    def __init__(
        self,
        responses: Sequence[str] = (),
        responder: Callable[[Sequence[Message]], str] | None = None,
    ):
        self._responses = list(responses)
        self._responder = responder
        self.calls: list[dict] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(
            {"messages": list(messages), "temperature": temperature, "json_mode": json_mode}
        )
        if self._responder is not None:
            return self._responder(messages)
        if not self._responses:
            raise LLMError("FakeLLM: no scripted responses left")
        return self._responses.pop(0)


class CachingLLM:
    """Wraps an LLM and caches complete() on disk by sha256 of (model, messages,
    temperature, json_mode, max_tokens). Used by the evaluations (P7.2) so a
    run resumed across Groq's daily cap never pays twice for the same call.
    Cache hits are counted in ``hits``; misses go to the wrapped LLM."""

    def __init__(self, llm: LLM, root, model: str):
        from pathlib import Path

        self.llm, self.root, self.model = llm, Path(root), model
        self.hits = 0
        self.misses = 0

    def _key(self, messages, temperature, json_mode, max_tokens) -> str:
        import hashlib

        blob = json.dumps(
            {"model": self.model, "messages": list(messages), "temperature": temperature,
             "json_mode": json_mode, "max_tokens": max_tokens},
            sort_keys=True,
        )  # fmt: skip
        return hashlib.sha256(blob.encode()).hexdigest()

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        key = self._key(messages, temperature, json_mode, max_tokens)
        path = self.root / f"{key}.json"
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text())["response"]
        self.misses += 1
        text = self.llm.complete(messages, temperature=temperature, json_mode=json_mode,
                                 max_tokens=max_tokens)  # fmt: skip
        atomic_write_text(path, json.dumps({"response": text}))
        return text

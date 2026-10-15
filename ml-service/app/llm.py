"""LLM access: an ``LLM`` protocol, the Groq implementation and a ``FakeLLM`` for tests.

Groq is called via its OpenAI-compatible REST API with httpx. 429 and 5xx
responses and transport errors are retried with exponential backoff (honouring
Retry-After). An eval-time response cache comes in P7.2.
"""

import random
import time
from collections.abc import Callable, Sequence
from typing import Protocol

import httpx

Message = dict[str, str]


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str: ...


RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class GroqLLM:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.groq.com/openai/v1",
        *,
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
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        last_error = ""
        for attempt in range(self.max_retries + 1):
            resp = None
            try:
                resp = self._client.post("/chat/completions", json=payload)
            except httpx.TransportError as exc:
                last_error = f"transport error: {exc}"
            else:
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code not in RETRYABLE_STATUS:
                    raise LLMError(last_error)
            if attempt < self.max_retries:
                self._sleep(self._delay(attempt, resp))
        raise LLMError(f"giving up after {self.max_retries + 1} attempts; last: {last_error}")


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

"""LLM provider abstraction for the copilot.

Two implementations satisfy the P0 contract:

* :class:`MockProvider` — fully offline and deterministic (same messages →
  same response, no network, no secrets). It is the default and produces a
  fixed, safe narrative; the actual quantified figures always come from the
  tool results rendered by the graph's synthesize step, never from the mock.
* :class:`OpenAICompatibleProvider` — talks to any OpenAI-compatible
  ``/chat/completions`` endpoint over ``httpx``. Base URL, API key, and model
  come from configuration/env; the API key is never logged, raised in an error
  message, or included in ``repr``.

Both implement :class:`LLMProvider`. ``build_provider`` chooses the
OpenAI-compatible provider only when explicitly configured *and* a key is
present; otherwise it falls back to the offline mock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import httpx

DEFAULT_TIMEOUT_SECONDS = 30.0

# Deterministic, secret-free narrative. It contains no invented statistics,
# causality, or actions — quantified values must be supplied by tool evidence.
_MOCK_NARRATIVE = (
    "Here is the grounded answer, derived solely from the deterministic tool "
    "evidence below. No statistics, metrics, causal claims, or actions are "
    "invented beyond what the evidence states."
)


@dataclass(frozen=True)
class LLMResponse:
    """A single provider completion."""

    content: str


class ProviderError(Exception):
    """Stable, secret-free error raised by LLM providers.

    The message is intentionally fixed so no base URL, API key, request body,
    or traceback can ever leak through an exception.
    """


class LLMProvider(Protocol):
    """Minimal provider protocol consumed by the graph's synthesize step."""

    def invoke(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        """Return a completion for ``messages`` (list of role/content dicts)."""
        ...


class MockProvider:
    """Offline, deterministic provider used by default (and in all tests)."""

    def invoke(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        del messages, kwargs  # deterministic: the output never depends on input
        return LLMResponse(content=_MOCK_NARRATIVE)


class OpenAICompatibleProvider:
    """OpenAI-compatible chat-completions provider over ``httpx``.

    ``transport`` is injectable for tests (``httpx.MockTransport``) and defaults
    to a real transport. The API key is used only to build the Authorization
    header and is never stored in a repr or surfaced in any exception.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url or not model:
            raise ValueError("base_url and model are required for the provider")
        if not api_key:
            raise ValueError("api_key is required for the provider")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._transport = transport

    def invoke(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        del kwargs
        url = f"{self._base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [dict(message) for message in messages],
            "temperature": 0.0,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
                response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (
            httpx.HTTPError,
            httpx.InvalidURL,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            raise ProviderError("LLM provider request failed") from exc
        if not isinstance(content, str):
            raise ProviderError("LLM provider returned an invalid response")
        return LLMResponse(content=content)


def build_provider(
    *,
    provider: str,
    api_key: str | None,
    base_url: str | None,
    model: str | None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> LLMProvider:
    """Select the provider from configuration.

    The OpenAI-compatible provider is used only when ``provider`` is
    ``"openai-compatible"`` and a key/base/model are all present; every other
    combination falls back to the offline, deterministic mock (no key ⇒ mock).
    """
    if provider == "openai-compatible" and api_key and base_url and model:
        return OpenAICompatibleProvider(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
        )
    return MockProvider()

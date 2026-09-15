"""Unit tests for the copilot provider abstraction."""

from __future__ import annotations

import json

import httpx
import pytest

from maintai.agent.provider import (
    LLMResponse,
    MockProvider,
    OpenAICompatibleProvider,
    ProviderError,
    build_provider,
)


def test_mock_provider_is_deterministic_and_offline():
    provider = MockProvider()
    messages = [{"role": "user", "content": "hello"}]
    first = provider.invoke(messages)
    second = provider.invoke(messages)
    assert isinstance(first, LLMResponse)
    assert first.content == second.content
    # The mock narrative invents no quantified figures of its own.
    assert not any(char.isdigit() for char in first.content)


def test_build_provider_defaults_to_mock_without_key():
    assert isinstance(
        build_provider(provider="openai-compatible", api_key=None, base_url=None, model=None),
        MockProvider,
    )
    assert isinstance(
        build_provider(provider="mock", api_key="sk-x", base_url="https://x", model="m"),
        MockProvider,
    )


def test_build_provider_selects_openai_when_configured():
    provider = build_provider(
        provider="openai-compatible",
        api_key="sk-x",
        base_url="https://example.com/v1",
        model="gpt-test",
    )
    assert isinstance(provider, OpenAICompatibleProvider)


def test_openai_provider_with_mock_transport():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "grounded answer"}}]})

    provider = OpenAICompatibleProvider(
        base_url="https://example.com/v1",
        api_key="sk-secret-123",
        model="gpt-test",
        transport=httpx.MockTransport(handler),
    )
    response = provider.invoke([{"role": "user", "content": "question"}])
    assert response.content == "grounded answer"
    assert seen["url"] == "https://example.com/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-secret-123"


def test_openai_provider_raises_stable_error_without_leaking_secret():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    provider = OpenAICompatibleProvider(
        base_url="https://example.com/v1",
        api_key="sk-secret-123",
        model="gpt-test",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ProviderError) as excinfo:
        provider.invoke([{"role": "user", "content": "question"}])
    message = str(excinfo.value)
    assert "sk-secret-123" not in message
    assert "LLM provider request failed" in message
    # Chaining is suppressed so no httpx request/response can leak via __cause__.
    assert excinfo.value.__cause__ is None


def test_openai_provider_requires_config():
    with pytest.raises(ValueError):
        OpenAICompatibleProvider(base_url="", api_key="sk", model="m")
    with pytest.raises(ValueError):
        OpenAICompatibleProvider(base_url="https://x", api_key="", model="m")


def test_openai_provider_repr_does_not_leak_secret():
    provider = OpenAICompatibleProvider(
        base_url="https://example.com/v1",
        api_key="sk-secret-123",
        model="gpt-test",
    )
    assert "sk-secret-123" not in repr(provider)
    assert "gpt-test" in repr(provider)


def _payload_for(thinking: str | None) -> dict:
    """Run one offline completion and return the request JSON body."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = OpenAICompatibleProvider(
        base_url="https://example.com/v1",
        api_key="sk-secret-123",
        model="gpt-test",
        thinking=thinking,
        transport=httpx.MockTransport(handler),
    )
    provider.invoke([{"role": "user", "content": "question"}])
    return seen["body"]


def test_payload_adds_thinking_when_mode_set():
    for mode in ("enabled", "disabled"):
        body = _payload_for(mode)
        assert body["thinking"] == {"type": mode}
        assert body["temperature"] == 0.0


def test_payload_preserves_old_shape_when_thinking_none():
    body = _payload_for(None)
    assert "thinking" not in body
    assert body == {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "question"}],
        "temperature": 0.0,
    }


def test_openai_provider_rejects_invalid_thinking_mode():
    with pytest.raises(ValueError):
        OpenAICompatibleProvider(
            base_url="https://example.com/v1",
            api_key="sk",
            model="m",
            thinking="maybe",
        )


def test_build_provider_passes_through_thinking():
    provider = build_provider(
        provider="openai-compatible",
        api_key="sk-x",
        base_url="https://example.com/v1",
        model="gpt-test",
        thinking="disabled",
    )
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider._thinking == "disabled"

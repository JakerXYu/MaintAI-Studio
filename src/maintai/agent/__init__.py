"""MaintAI Studio copilot (LangGraph) package.

Exposes the provider abstraction, the read-only tool allowlist, the compiled
graph, and the :class:`CopilotService` used by the REST API.
"""

from __future__ import annotations

from maintai.agent.graph import build_graph, route_intent
from maintai.agent.provider import (
    LLMProvider,
    LLMResponse,
    MockProvider,
    OpenAICompatibleProvider,
    ProviderError,
    build_provider,
)
from maintai.agent.service import CopilotService
from maintai.agent.tools import CopilotTools, ToolResult

__all__ = [
    "CopilotService",
    "CopilotTools",
    "LLMProvider",
    "LLMResponse",
    "MockProvider",
    "OpenAICompatibleProvider",
    "ProviderError",
    "ToolResult",
    "build_graph",
    "build_provider",
    "route_intent",
]

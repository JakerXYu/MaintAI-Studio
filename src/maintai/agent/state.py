"""LangGraph copilot state.

The state is a plain ``TypedDict`` so it can be fed into and returned from a
compiled ``langgraph.graph.StateGraph`` without any serialization concerns.
Every field is optional at rest (the graph fills them in as it runs) and the
final ``CopilotService.chat`` response only surfaces the public, JSON-safe
subset.
"""

from __future__ import annotations

from typing import Any, TypedDict

from maintai.agent.tools import ToolResult


class CopilotState(TypedDict, total=False):
    """Shared state threaded through the route → tool → synthesize nodes."""

    conversation_id: str
    user_request: str
    dataset_id: str | None
    experiment_id: str | None
    model_id: str | None
    record: dict[str, Any] | None
    intent: str
    tool_results: list[ToolResult]
    evidence: list[dict[str, Any]]
    proposed_action: dict[str, Any] | None
    final_answer: str

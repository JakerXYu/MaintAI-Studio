"""Copilot application service (LangGraph orchestration).

Wraps the compiled route → tool → synthesize graph and turns a user request into
the stable ``answer/intent/evidence/tool_results/proposed_action/conversation_id``
response consumed by ``POST /api/v1/copilot/chat``. The service never touches the
database or registry directly — it only dispatches to the allowlisted read-only
tools bound to the existing application services.
"""

from __future__ import annotations

import uuid
from typing import Any

from maintai.agent.graph import INTENT_UNKNOWN, build_graph
from maintai.agent.provider import LLMProvider
from maintai.agent.state import CopilotState
from maintai.agent.tools import CopilotTools


class CopilotService:
    """Compiled copilot over the deterministic intent router and allowlisted tools."""

    def __init__(self, *, tools: CopilotTools, provider: LLMProvider) -> None:
        self._graph = build_graph(provider=provider, tools=tools)

    def chat(
        self,
        *,
        user_request: str,
        dataset_id: str | None = None,
        experiment_id: str | None = None,
        model_id: str | None = None,
        record: dict[str, Any] | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Run the copilot graph once and return the JSON-safe response."""
        conversation_id = conversation_id or uuid.uuid4().hex
        initial: CopilotState = {
            "conversation_id": conversation_id,
            "user_request": user_request,
            "dataset_id": dataset_id,
            "experiment_id": experiment_id,
            "model_id": model_id,
            "record": record,
        }
        final = self._graph.invoke(initial)
        return {
            "answer": final.get("final_answer") or "",
            "intent": final.get("intent") or INTENT_UNKNOWN,
            "evidence": final.get("evidence") or [],
            "tool_results": [
                result.to_dict() for result in (final.get("tool_results") or [])
            ],
            "proposed_action": final.get("proposed_action"),
            "conversation_id": conversation_id,
        }

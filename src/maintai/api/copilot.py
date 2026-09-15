"""Copilot chat REST API routes.

Exposes ``POST /copilot/chat`` (mounted under ``/api/v1``). The route accepts a
fixed body — ``user_request`` plus optional ``dataset_id``/``experiment_id``/
``model_id``/``record``/``conversation_id`` — and delegates to
:class:`~maintai.agent.service.CopilotService`, which owns the graph + tool
allowlist. The API layer never writes audit events or touches repositories
itself.

The copilot is constructed from the *already-built* application services
(Dataset/Experiment/ModelRegistry/Prediction) — this module reuses them rather
than forking new instances — and a provider selected from configuration
(``mock`` default, ``openai-compatible`` only when a key is present).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from maintai.agent.provider import build_provider
from maintai.agent.service import CopilotService
from maintai.agent.tools import CopilotTools
from maintai.application.datasets import DatasetService
from maintai.application.experiments import ExperimentService
from maintai.application.models import ModelRegistryService
from maintai.application.predictions import PredictionService
from maintai.config import Settings


class CopilotChatRequest(BaseModel):
    """Fixed copilot request body; arbitrary fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    user_request: str = Field(min_length=1)
    dataset_id: str | None = None
    experiment_id: str | None = None
    model_id: str | None = None
    record: dict[str, Any] | None = None
    conversation_id: str | None = None


class CopilotChatResponse(BaseModel):
    """Stable copilot response shape."""

    model_config = ConfigDict(extra="ignore")

    answer: str
    intent: str
    evidence: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    proposed_action: dict[str, Any] | None = None
    conversation_id: str


def build_copilot_service(
    settings: Settings,
    dataset_service: DatasetService,
    experiment_service: ExperimentService,
    model_registry_service: ModelRegistryService,
    prediction_service: PredictionService,
) -> CopilotService:
    """Build the copilot reusing the already-constructed application services."""
    provider = build_provider(
        provider=settings.llm_provider,
        api_key=(
            settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
        ),
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        thinking=settings.llm_thinking,
    )
    tools = CopilotTools(
        dataset_service=dataset_service,
        experiment_service=experiment_service,
        model_registry_service=model_registry_service,
        prediction_service=prediction_service,
    )
    return CopilotService(tools=tools, provider=provider)


def build_copilot_router(service: CopilotService) -> APIRouter:
    """Build the copilot router bound to a single ``CopilotService``."""
    router = APIRouter(prefix="/copilot", tags=["copilot"])

    @router.post("/chat", response_model=CopilotChatResponse)
    def chat(body: CopilotChatRequest) -> CopilotChatResponse:
        return service.chat(
            user_request=body.user_request,
            dataset_id=body.dataset_id,
            experiment_id=body.experiment_id,
            model_id=body.model_id,
            record=body.record,
            conversation_id=body.conversation_id,
        )

    return router

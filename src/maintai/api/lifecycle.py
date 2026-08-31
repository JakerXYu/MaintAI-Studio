"""Champion/challenger lifecycle REST API routes (P1).

Exposes the promotion slice under the ``/models`` namespace:

* ``POST /models/{registered_id}/promotion-request`` — record a challenger and
  propose a ``model_promotion`` approval (system/agent initiated, no human gate).
* ``POST /models/{registered_id}/promote`` — execute an *already approved*
  promotion. This endpoint is gated by the same local-demo human gate as the
  approval decisions (:mod:`maintai.api.human_gate`): bearer token plus a
  non-empty ``X-Human-Actor-ID``.

Approving the underlying approval (``/approvals/{id}/approve``) only records the
decision — it never executes. Execution happens here, explicitly, and is
idempotent via the approval-execution receipt. Every mutation delegates to
:class:`~maintai.application.lifecycle.ModelLifecycleService`, which owns the
audit writes; this layer never writes audit events or touches repositories
directly.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session, sessionmaker

from maintai.api.human_gate import require_human
from maintai.application.lifecycle import (
    LifecycleApprovalNotFoundError,
    LifecycleApprovalNotReadyError,
    LifecycleModelNotFoundError,
    LifecycleNotPromotableError,
    LifecyclePromotionError,
    LifecycleServiceError,
    LifecycleValidationError,
    ModelLifecycleService,
)
from maintai.approvals import (
    ApprovalConflictError,
    ApprovalService,
    ApprovalValidationError,
)
from maintai.audit.service import get_audit_service
from maintai.config import Settings
from maintai.db.approval_execution_repository import ApprovalExecutionRepository
from maintai.db.approval_repository import ApprovalRepository
from maintai.db.lifecycle_repository import ModelLifecycleRepository
from maintai.db.model_repository import ModelRepository
from maintai.mlops.registry import MLflowRegistry

_REQUESTER_TYPE_MAX_LEN = 32
_REQUESTER_ID_MAX_LEN = 64
_APPROVAL_ID_MAX_LEN = 64

# Stable, path-free 500 details (mirrors the other API modules).
_PROMOTION_REQUEST_FAILED_DETAIL = "promotion request failed"
_PROMOTION_FAILED_DETAIL = "promotion execution failed"


# -- request/response models --------------------------------------------------


class PromotionRequestRequest(BaseModel):
    """Body for proposing a promotion; arbitrary fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    requested_by_type: str = Field(default="agent", max_length=_REQUESTER_TYPE_MAX_LEN)
    requested_by_id: str | None = Field(default=None, max_length=_REQUESTER_ID_MAX_LEN)


class PromoteRequest(BaseModel):
    """Body for executing an approved promotion; arbitrary fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    approval_id: str = Field(min_length=1, max_length=_APPROVAL_ID_MAX_LEN)


class PromotionRequestResponse(BaseModel):
    """Stable JSON shape for a promotion request (approval + challenger state)."""

    model_config = ConfigDict(extra="ignore")

    approval_id: str
    approval_status: str
    registered_model_id: str
    lifecycle_status: str
    proposed_payload: dict[str, Any] | None = None


class PromotionResponse(BaseModel):
    """Stable JSON shape for a promotion execution result."""

    model_config = ConfigDict(extra="ignore")

    registered_model_id: str
    name: str | None = None
    version: str | None = None
    lifecycle_status: str | None = None
    approval_id: str
    approval_status: str | None = None
    executed: bool
    receipt_id: str | None = None
    archived_champion_ids: list[str] = []
    mlflow_model_uri: str | None = None


# -- builder ------------------------------------------------------------------


def build_lifecycle_service(
    session_factory: sessionmaker[Session],
    settings: Settings,
) -> ModelLifecycleService:
    """Construct the production lifecycle service from settings/session/repos.

    Wires fresh model/lifecycle/approval/execution repositories, a shared audit
    service, the shared approval service, and an :class:`MLflowRegistry` bound
    to the configured tracking URI. No network or MLflow access happens here.
    """
    audit = get_audit_service(session_factory)
    approval_repository = ApprovalRepository(session_factory)
    return ModelLifecycleService(
        session_factory=session_factory,
        model_repository=ModelRepository(session_factory),
        lifecycle_repository=ModelLifecycleRepository(session_factory),
        approval_repository=approval_repository,
        execution_repository=ApprovalExecutionRepository(session_factory),
        audit=audit,
        registry=MLflowRegistry(settings.mlflow_tracking_uri),
        approval_service=ApprovalService(
            session_factory=session_factory,
            repository=approval_repository,
            audit=audit,
        ),
    )


def build_lifecycle_router(service: ModelLifecycleService, *, token: str | None) -> APIRouter:
    """Build the lifecycle router bound to one ``ModelLifecycleService`` and a token.

    ``token`` is the configured bearer secret (or ``None`` to disable the human
    gate on the promote endpoint).
    """
    router = APIRouter(prefix="/models", tags=["models"])

    @router.post(
        "/{registered_id}/promotion-request",
        response_model=PromotionRequestResponse,
        status_code=201,
    )
    def request_promotion(
        registered_id: str, body: PromotionRequestRequest
    ) -> PromotionRequestResponse:
        try:
            return service.request_promotion(
                registered_id,
                requested_by_type=body.requested_by_type,
                requested_by_id=body.requested_by_id,
            )
        except LifecycleModelNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except LifecycleNotPromotableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ApprovalConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ApprovalValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LifecycleServiceError:
            raise HTTPException(
                status_code=500, detail=_PROMOTION_REQUEST_FAILED_DETAIL
            ) from None

    @router.post("/{registered_id}/promote", response_model=PromotionResponse)
    def promote(registered_id: str, body: PromoteRequest, request: Request) -> PromotionResponse:
        actor_id = require_human(request, token)
        try:
            return service.execute_promotion(
                registered_id, body.approval_id, human_actor_id=actor_id
            )
        except LifecycleModelNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except LifecycleApprovalNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except LifecycleApprovalNotReadyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except LifecycleValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LifecyclePromotionError:
            raise HTTPException(
                status_code=500, detail=_PROMOTION_FAILED_DETAIL
            ) from None
        except LifecycleServiceError:
            raise HTTPException(status_code=500, detail=_PROMOTION_FAILED_DETAIL) from None

    return router

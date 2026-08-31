"""Mock CMMS REST API routes (P1 vertical slice, no live connector).

Exposes ``POST /api/v1/cmms/work-orders/draft`` (execute an already-approved
``cmms_work_order`` approval into one mock work order + execution receipt,
idempotently) and a bounded newest-first list. Every mutation delegates to
:class:`~maintai.cmms.MockCMMSService`, which owns the validation, persistence,
and audit writes; this layer never writes audit events or touches repositories
directly.

The draft endpoint is gated by the shared local-demo bearer token via
:func:`maintai.api.human_gate.require_bearer` (the human decision already
happened at approval time; this endpoint merely executes it as the ``system``
actor). Responses always carry ``mock=true`` and a disclaimer, and there is no
external connector anywhere in this slice.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session, sessionmaker

from maintai.api.human_gate import require_bearer
from maintai.audit.service import get_audit_service
from maintai.cmms import (
    CMMSApprovalNotExecutableError,
    CMMSApprovalNotFoundError,
    CMMSPredictionEventNotFoundError,
    CMMSValidationError,
    MockCMMSService,
    MockCMMSServiceError,
)
from maintai.db.approval_execution_repository import ApprovalExecutionRepository
from maintai.db.approval_repository import ApprovalRepository
from maintai.db.cmms_repository import MockCMMSWorkOrderRepository

# Stable, path-free 500 detail (mirrors the other API modules).
_CMMS_FAILED_DETAIL = "mock CMMS work-order draft failed"


class CMMSDraftRequest(BaseModel):
    """Body for drafting a work order from an approved approval.

    Only ``approval_id`` is accepted; arbitrary fields are rejected outright.
    """

    model_config = ConfigDict(extra="forbid")

    approval_id: str = Field(min_length=1, max_length=64)


class CMMSDraftResponse(BaseModel):
    """Stable JSON shape for a mock work order + execution receipt."""

    model_config = ConfigDict(extra="ignore")

    id: str
    approval_id: str
    asset_id: str
    priority: str
    recommended_action: str
    reason: str | None = None
    risk_score: float | None = None
    evidence: dict[str, Any] | None = None
    source_model_version: str | None = None
    status: str
    created_at: str | None = None
    updated_at: str | None = None
    execution_id: str | None = None
    mock: bool
    disclaimer: str


def build_cmms_service(session_factory: sessionmaker[Session]) -> MockCMMSService:
    """Construct the production mock-CMMS service for a session factory.

    Wires the approval, work-order, and execution repositories plus a shared
    :class:`~maintai.audit.service.AuditService`; no DB access happens until an
    operation runs.
    """
    return MockCMMSService(
        session_factory=session_factory,
        approval_repository=ApprovalRepository(session_factory),
        work_order_repository=MockCMMSWorkOrderRepository(session_factory),
        execution_repository=ApprovalExecutionRepository(session_factory),
        audit=get_audit_service(session_factory),
    )


def build_cmms_router(service: MockCMMSService, *, token: str | None) -> APIRouter:
    """Build the CMMS router bound to one ``MockCMMSService`` and a bearer token.

    ``token`` is the configured local-demo bearer secret (or ``None`` to disable
    drafting).
    """

    router = APIRouter(prefix="/cmms", tags=["cmms"])

    @router.post("/work-orders/draft", response_model=CMMSDraftResponse)
    def create_draft(
        body: CMMSDraftRequest, request: Request, response: Response
    ) -> CMMSDraftResponse:
        require_bearer(request, token)
        try:
            result, created = service.create_draft(body.approval_id)
        except CMMSApprovalNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except CMMSPredictionEventNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except CMMSApprovalNotExecutableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except CMMSValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except MockCMMSServiceError:
            raise HTTPException(status_code=500, detail=_CMMS_FAILED_DETAIL) from None
        response.status_code = 201 if created else 200
        return result

    @router.get("/work-orders", response_model=list[CMMSDraftResponse])
    def list_work_orders(
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> list[CMMSDraftResponse]:
        return service.list(limit=limit, offset=offset)

    return router

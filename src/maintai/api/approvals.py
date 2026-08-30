"""P1 approval REST API routes (human-in-the-loop gate).

Exposes the approval state machine over HTTP: propose a gated operational action
(model promotion, retraining deployment, maintenance action, CMMS work order),
list/get approvals, and let a human decide ``approve``/``reject``/``modify``.
Every mutation delegates to :class:`~maintai.approvals.service.ApprovalService`,
which owns the state machine and the audit writes; this layer never writes audit
events or touches repositories directly.

Human decisions are gated by two inputs that must be present together:

* ``Authorization: Bearer <token>`` — compared in constant time against the
  environment-configured ``approval_api_token``. An unset token returns ``503``;
  a missing or wrong token returns ``401``.
* ``X-Human-Actor-ID`` — a non-empty identifier for the deciding human; missing
  returns ``422``.

This is a **local demo gate**, not an enterprise identity: a shared bearer token
plus a self-asserted actor id is not production authentication. Production must
front these routes with SSO/RBAC (see ``docs/API_CONTRACT.md``). The token is
never logged, echoed, or written to the audit trail, and audit events never
contain proposed/decision payloads (the approval service already redacts them).
"""

from __future__ import annotations

import secrets
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session, sessionmaker

from maintai.approvals import (
    ApprovalConflictError,
    ApprovalNotFoundError,
    ApprovalService,
    ApprovalServiceError,
    ApprovalValidationError,
)
from maintai.audit.service import get_audit_service
from maintai.db.approval_repository import ApprovalRepository

# Length ceilings mirror the ``ApprovalRequest`` columns (maintai.db.models).
_ACTION_MAX_LEN = 64
_ENTITY_MAX_LEN = 64
_ENTITY_ID_MAX_LEN = 64
_REQUESTER_TYPE_MAX_LEN = 32
_REQUESTER_ID_MAX_LEN = 64
_STATUS_MAX_LEN = 32
_REASON_MAX_LEN = 2048

# Stable, path-free 500 details (mirrors the other API modules).
_PROPOSE_FAILED_DETAIL = "approval request failed"
_DECIDE_FAILED_DETAIL = "approval decision failed"


# -- request/response models --------------------------------------------------


class ApprovalProposeRequest(BaseModel):
    """Body for proposing a gated action; arbitrary fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    action_type: str = Field(min_length=1, max_length=_ACTION_MAX_LEN)
    entity_type: str = Field(min_length=1, max_length=_ENTITY_MAX_LEN)
    entity_id: str = Field(min_length=1, max_length=_ENTITY_ID_MAX_LEN)
    requested_by_type: str = Field(min_length=1, max_length=_REQUESTER_TYPE_MAX_LEN)
    requested_by_id: str | None = Field(default=None, max_length=_REQUESTER_ID_MAX_LEN)
    proposed_payload: dict[str, Any] | None = None


class ApprovalDecisionRequest(BaseModel):
    """Body for a human decision; ``expected_version`` is mandatory.

    ``reason`` is required for ``reject``/``modify`` and ``payload`` is required
    for ``modify``; those semantic rules are enforced by the approval service and
    surfaced as ``422``. Arbitrary fields (including any actor-type override) are
    rejected so a client can never change who is deciding.
    """

    model_config = ConfigDict(extra="forbid")

    expected_version: int = Field(ge=1)
    reason: str | None = Field(default=None, max_length=_REASON_MAX_LEN)
    payload: dict[str, Any] | None = None


class ApprovalSummaryResponse(BaseModel):
    """Redacted list summary (never contains proposed/decision payloads)."""

    model_config = ConfigDict(extra="ignore")

    id: str
    action_type: str
    entity_type: str
    entity_id: str
    status: str
    requested_by_type: str
    requested_by_id: str | None = None
    decided_by: str | None = None
    reason: str | None = None
    version: int
    created_at: str | None = None
    decided_at: str | None = None


class ApprovalResponse(ApprovalSummaryResponse):
    """Full detail including the proposed and decision payloads."""

    proposed_payload: dict[str, Any] | None = None
    decision_payload: dict[str, Any] | None = None


# -- builders ----------------------------------------------------------------


def build_approval_service(session_factory: sessionmaker[Session]) -> ApprovalService:
    """Construct the production approval service for a session factory.

    Wires an :class:`~maintai.db.approval_repository.ApprovalRepository` and a
    shared :class:`~maintai.audit.service.AuditService`; no DB or network access
    happens until an approval operation runs.
    """
    return ApprovalService(
        session_factory=session_factory,
        repository=ApprovalRepository(session_factory),
        audit=get_audit_service(session_factory),
    )


# -- router ------------------------------------------------------------------


def build_approvals_router(service: ApprovalService, *, token: str | None) -> APIRouter:
    """Build the approval router bound to one ``ApprovalService`` and a token.

    ``token`` is the configured bearer secret (or ``None`` to disable decisions).
    """

    router = APIRouter(prefix="/approvals", tags=["approvals"])

    def _require_token(request: Request) -> None:
        """Require the configured local-demo bearer token."""
        if token is None:
            raise HTTPException(
                status_code=503,
                detail="approval decisions are disabled: APPROVAL_API_TOKEN is not configured",
            )
        authorization = request.headers.get("authorization")
        if authorization is None:
            raise HTTPException(status_code=401, detail="missing bearer token")
        scheme, _, credentials = authorization.partition(" ")
        if scheme.lower() != "bearer" or not credentials:
            raise HTTPException(status_code=401, detail="invalid authorization header")
        if not secrets.compare_digest(
            credentials.encode("utf-8"), token.encode("utf-8")
        ):
            raise HTTPException(status_code=401, detail="invalid bearer token")

    def _require_human_decision(request: Request) -> str:
        """Enforce the decision gate and return the deciding human actor id."""
        _require_token(request)
        actor_id = request.headers.get("x-human-actor-id")
        if actor_id is None or not actor_id.strip():
            raise HTTPException(
                status_code=422, detail="X-Human-Actor-ID header is required"
            )
        return actor_id.strip()

    def _decide(
        approval_id: str,
        decision: str,
        body: ApprovalDecisionRequest,
        request: Request,
    ) -> ApprovalResponse:
        actor_id = _require_human_decision(request)
        try:
            return service.decide(
                approval_id,
                decision=decision,
                human_actor_id=actor_id,
                human_actor_type="user",
                expected_version=body.expected_version,
                reason=body.reason,
                payload=body.payload,
            )
        except ApprovalNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ApprovalConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ApprovalValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ApprovalServiceError:
            raise HTTPException(status_code=500, detail=_DECIDE_FAILED_DETAIL) from None

    @router.post("", response_model=ApprovalResponse, status_code=201)
    def propose(body: ApprovalProposeRequest) -> ApprovalResponse:
        try:
            return service.propose(
                action_type=body.action_type,
                entity_type=body.entity_type,
                entity_id=body.entity_id,
                requested_by_type=body.requested_by_type,
                requested_by_id=body.requested_by_id,
                proposed_payload=body.proposed_payload,
            )
        except ApprovalValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ApprovalConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ApprovalServiceError:
            raise HTTPException(status_code=500, detail=_PROPOSE_FAILED_DETAIL) from None

    @router.get("", response_model=list[ApprovalSummaryResponse])
    def list_approvals(
        action_type: str | None = Query(default=None, max_length=_ACTION_MAX_LEN),
        entity_type: str | None = Query(default=None, max_length=_ENTITY_MAX_LEN),
        entity_id: str | None = Query(default=None, max_length=_ENTITY_ID_MAX_LEN),
        status: str | None = Query(default=None, max_length=_STATUS_MAX_LEN),
        requested_by_type: str | None = Query(
            default=None, max_length=_REQUESTER_TYPE_MAX_LEN
        ),
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> list[ApprovalSummaryResponse]:
        return service.list(
            action_type=action_type,
            entity_type=entity_type,
            entity_id=entity_id,
            status=status,
            requested_by_type=requested_by_type,
            limit=limit,
            offset=offset,
        )

    @router.get("/{approval_id}", response_model=ApprovalResponse)
    def get_approval(approval_id: str, request: Request) -> ApprovalResponse:
        _require_token(request)
        try:
            return service.get(approval_id)
        except ApprovalNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/{approval_id}/approve", response_model=ApprovalResponse)
    def approve(
        approval_id: str,
        body: ApprovalDecisionRequest,
        request: Request,
    ) -> ApprovalResponse:
        return _decide(approval_id, "approve", body, request)

    @router.post("/{approval_id}/reject", response_model=ApprovalResponse)
    def reject(
        approval_id: str,
        body: ApprovalDecisionRequest,
        request: Request,
    ) -> ApprovalResponse:
        return _decide(approval_id, "reject", body, request)

    @router.post("/{approval_id}/modify", response_model=ApprovalResponse)
    def modify(
        approval_id: str,
        body: ApprovalDecisionRequest,
        request: Request,
    ) -> ApprovalResponse:
        return _decide(approval_id, "modify", body, request)

    return router

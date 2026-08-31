"""Technician feedback REST API routes (P1 append-only capture).

Exposes ``POST /api/v1/predictions/{prediction_id}/feedback`` (append) and
``GET /api/v1/predictions/{prediction_id}/feedback`` (list, newest-first).
Submissions require a non-empty ``X-Human-Actor-ID`` header — a self-asserted
local-demo identity, so **no bearer token** is required (unlike the approval
decision gate). Every mutation delegates to
:class:`~maintai.feedback.service.FeedbackService`, which owns the prediction
existence check and the audit write; this layer never writes audit events or
touches repositories directly.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session, sessionmaker

from maintai.audit.service import get_audit_service
from maintai.db.feedback_repository import TechnicianFeedbackRepository
from maintai.feedback import (
    FeedbackPredictionNotFoundError,
    FeedbackService,
    FeedbackServiceError,
    FeedbackValidationError,
)

# Length ceilings mirror the ``TechnicianFeedback`` columns (maintai.db.models).
_OUTCOME_MAX_LEN = 32
_COMMENT_MAX_LEN = 2048

# Stable, path-free 500 detail (mirrors the other API modules).
_CREATE_FAILED_DETAIL = "feedback submission failed"


def _require_human_actor(request: Request) -> str:
    """Require a non-empty ``X-Human-Actor-ID`` header and return it stripped.

    No bearer token is needed for feedback: this is a self-asserted, local-demo
    identity gate only (see ``docs/API_CONTRACT.md``).
    """
    actor_id = request.headers.get("x-human-actor-id")
    if actor_id is None or not actor_id.strip():
        raise HTTPException(status_code=422, detail="X-Human-Actor-ID header is required")
    return actor_id.strip()


class FeedbackCreateRequest(BaseModel):
    """Body for appending feedback; arbitrary fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    outcome: str = Field(min_length=1, max_length=_OUTCOME_MAX_LEN)
    comment: str | None = Field(default=None, max_length=_COMMENT_MAX_LEN)


class FeedbackResponse(BaseModel):
    """Stable JSON shape for one feedback row (comment is part of the record)."""

    model_config = ConfigDict(extra="ignore")

    id: str
    prediction_event_id: str | None = None
    registered_model_id: str | None = None
    outcome: str
    comment: str | None = None
    technician_id: str | None = None
    created_at: str | None = None


def build_feedback_service(session_factory: sessionmaker[Session]) -> FeedbackService:
    """Construct the production feedback service for a session factory.

    Wires a :class:`~maintai.db.feedback_repository.TechnicianFeedbackRepository`
    and the shared :class:`~maintai.audit.service.AuditService`; no DB or network
    access happens until a feedback operation runs.
    """
    return FeedbackService(
        session_factory=session_factory,
        repository=TechnicianFeedbackRepository(session_factory),
        audit=get_audit_service(session_factory),
    )


def build_feedback_router(service: FeedbackService) -> APIRouter:
    """Build the feedback router bound to one ``FeedbackService``."""
    router = APIRouter(prefix="/predictions", tags=["feedback"])

    @router.post(
        "/{prediction_id}/feedback", response_model=FeedbackResponse, status_code=201
    )
    def create_feedback(
        prediction_id: str, body: FeedbackCreateRequest, request: Request
    ) -> FeedbackResponse:
        technician_id = _require_human_actor(request)
        try:
            return service.create(
                prediction_id,
                outcome=body.outcome,
                technician_id=technician_id,
                comment=body.comment,
            )
        except FeedbackPredictionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FeedbackValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except FeedbackServiceError:
            raise HTTPException(status_code=500, detail=_CREATE_FAILED_DETAIL) from None

    @router.get("/{prediction_id}/feedback", response_model=list[FeedbackResponse])
    def list_feedback(
        prediction_id: str,
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> list[FeedbackResponse]:
        return service.list(prediction_id, limit=limit, offset=offset)

    return router

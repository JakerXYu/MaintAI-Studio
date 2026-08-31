"""Technician feedback application service (P1 append-only capture).

Coordinates the technician-feedback vertical slice: a human technician records
exactly one of four outcomes on a prediction, plus an optional bounded comment.
Rows are append-only — the repository exposes no update/delete path — so a
captured outcome can never be rewritten or removed.

The service verifies the referenced prediction exists before writing, persists
the feedback and a single audit event atomically, and never copies the comment
content into the audit trail (audit payloads carry only ids, outcome, and the
self-asserted technician id, never free-text).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from maintai.audit.service import AuditService
from maintai.db.feedback_repository import TechnicianFeedbackRepository
from maintai.db.models import (
    FEEDBACK_OUTCOME_CONFIRMED_ISSUE,
    FEEDBACK_OUTCOME_DIFFERENT_ISSUE,
    FEEDBACK_OUTCOME_FALSE_ALARM,
    FEEDBACK_OUTCOME_NO_ACTION_NEEDED,
    PredictionEvent,
    TechnicianFeedback,
    new_id,
)

# The four canonical technician outcomes (mirrors maintai.db.models constants).
FEEDBACK_OUTCOMES = frozenset(
    {
        FEEDBACK_OUTCOME_CONFIRMED_ISSUE,
        FEEDBACK_OUTCOME_FALSE_ALARM,
        FEEDBACK_OUTCOME_DIFFERENT_ISSUE,
        FEEDBACK_OUTCOME_NO_ACTION_NEEDED,
    }
)

# Length ceilings mirror the ``TechnicianFeedback`` columns (maintai.db.models).
_COMMENT_MAX_LEN = 2048
_TECHNICIAN_ID_MAX_LEN = 64


class FeedbackServiceError(Exception):
    """Base class for feedback application-service errors."""


class FeedbackPredictionNotFoundError(FeedbackServiceError):
    """Raised when the referenced prediction id does not exist."""


class FeedbackValidationError(FeedbackServiceError):
    """Raised when a feedback submission violates the feedback contract."""


class FeedbackService:
    """Application service for append-only technician feedback capture."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        repository: TechnicianFeedbackRepository,
        audit: AuditService,
    ) -> None:
        self._session_factory = session_factory
        self._repository = repository
        self._audit = audit

    @staticmethod
    def _nonempty(value: str | None) -> bool:
        return isinstance(value, str) and bool(value.strip())

    @staticmethod
    def _to_dict(feedback: TechnicianFeedback) -> dict[str, Any]:
        """Render a feedback row as JSON-safe detail (comment is included here)."""
        return {
            "id": feedback.id,
            "prediction_event_id": feedback.prediction_event_id,
            "registered_model_id": feedback.registered_model_id,
            "outcome": feedback.outcome,
            "comment": feedback.comment,
            "technician_id": feedback.technician_id,
            "created_at": feedback.created_at.isoformat() if feedback.created_at else None,
        }

    # -- public API --------------------------------------------------------

    def create(
        self,
        prediction_id: str,
        *,
        outcome: str,
        technician_id: str,
        comment: str | None = None,
    ) -> dict[str, Any]:
        """Append one technician feedback row (plus its audit event) atomically.

        ``technician_id`` is mandatory and is supplied by the route from the
        ``X-Human-Actor-ID`` header (self-asserted, local-demo identity — no
        bearer token). The referenced prediction must exist; ``outcome`` must be
        one of the four canonical outcomes and ``comment`` is bounded to 2048
        chars. The audit event never contains the comment content.
        """
        if outcome not in FEEDBACK_OUTCOMES:
            raise FeedbackValidationError(
                f"outcome must be one of {sorted(FEEDBACK_OUTCOMES)}"
            )
        if not self._nonempty(technician_id):
            raise FeedbackValidationError("technician_id is required")
        if len(technician_id) > _TECHNICIAN_ID_MAX_LEN:
            raise FeedbackValidationError("technician_id must be at most 64 chars")
        if comment is not None and len(comment) > _COMMENT_MAX_LEN:
            raise FeedbackValidationError("comment must be at most 2048 chars")

        with self._session_factory.begin() as session:
            prediction = session.get(PredictionEvent, prediction_id)
            if prediction is None:
                raise FeedbackPredictionNotFoundError(
                    f"prediction {prediction_id!r} not found"
                )
            feedback = TechnicianFeedback(
                id=new_id(),
                prediction_event_id=prediction_id,
                registered_model_id=prediction.registered_model_id,
                outcome=outcome,
                comment=comment,
                technician_id=technician_id,
            )
            feedback = self._repository.create(feedback, session=session)
            self._audit.record(
                actor_type="user",
                actor_id=technician_id,
                action="feedback.create",
                entity_type="prediction_event",
                entity_id=prediction_id,
                payload={
                    "feedback_id": feedback.id,
                    "prediction_event_id": prediction_id,
                    "registered_model_id": prediction.registered_model_id,
                    "outcome": outcome,
                    "technician_id": technician_id,
                },
                session=session,
            )
        return self._to_dict(feedback)

    def list(
        self, prediction_id: str, *, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Return feedback for a prediction, newest-first."""
        rows = self._repository.list(
            prediction_event_id=prediction_id, limit=limit, offset=offset
        )
        return [self._to_dict(row) for row in rows]

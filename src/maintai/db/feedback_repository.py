"""TechnicianFeedback repository (SQLAlchemy 2, session-factory based).

Mirrors the other repositories in this codebase: every method opens its own
short-lived session unless an external ``session`` is supplied. Feedback is
append-only — the repository exposes no ``update``/``delete`` path, so captured
technician outcomes can never be rewritten or removed.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from maintai.db.models import TechnicianFeedback


class TechnicianFeedbackRepository:
    """Persistent access to :class:`maintai.db.models.TechnicianFeedback` rows."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create(
        self, feedback: TechnicianFeedback, *, session: Session | None = None
    ) -> TechnicianFeedback:
        """Insert a new feedback row and return the flushed instance."""
        if session is not None:
            session.add(feedback)
            session.flush()
            return feedback
        with self._session_factory.begin() as session:
            session.add(feedback)
            session.flush()
            return feedback

    def get(
        self, feedback_id: str, *, session: Session | None = None
    ) -> TechnicianFeedback | None:
        """Return a feedback row by primary key, or ``None`` when absent."""
        if session is not None:
            return session.get(TechnicianFeedback, feedback_id)
        with self._session_factory() as session:
            return session.get(TechnicianFeedback, feedback_id)

    def list(
        self,
        *,
        prediction_event_id: str | None = None,
        registered_model_id: str | None = None,
        outcome: str | None = None,
        limit: int = 100,
        offset: int = 0,
        session: Session | None = None,
    ) -> list[TechnicianFeedback]:
        """Return feedback newest-first with clamped pagination and filters."""
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        stmt = (
            select(TechnicianFeedback)
            .order_by(TechnicianFeedback.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if prediction_event_id is not None:
            stmt = stmt.where(
                TechnicianFeedback.prediction_event_id == prediction_event_id
            )
        if registered_model_id is not None:
            stmt = stmt.where(
                TechnicianFeedback.registered_model_id == registered_model_id
            )
        if outcome is not None:
            stmt = stmt.where(TechnicianFeedback.outcome == outcome)
        if session is not None:
            return list(session.scalars(stmt))
        with self._session_factory() as session:
            return list(session.scalars(stmt))

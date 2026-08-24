"""PredictionEvent repository (SQLAlchemy 2, session-factory based).

Mirrors the other repositories in this codebase. ``bulk_create`` accepts an
external ``session`` so the prediction service can persist all per-record
``PredictionEvent`` rows and its single request-level audit event in one
transaction. Prediction events are immutable once written (no update/delete).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from maintai.db.models import PredictionEvent


class PredictionRepository:
    """Persistent access to :class:`maintai.db.models.PredictionEvent` rows."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def bulk_create(
        self, events: list[PredictionEvent], *, session: Session | None = None
    ) -> list[PredictionEvent]:
        """Insert a batch of prediction events and return the flushed instances."""
        if not events:
            return []
        if session is not None:
            session.add_all(events)
            session.flush()
            return events
        with self._session_factory.begin() as session:
            session.add_all(events)
            session.flush()
            return events

    def list(
        self,
        *,
        registered_model_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[PredictionEvent]:
        """Return prediction events ordered by newest-first with clamped pagination."""
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        stmt = (
            select(PredictionEvent)
            .order_by(PredictionEvent.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if registered_model_id is not None:
            stmt = stmt.where(PredictionEvent.registered_model_id == registered_model_id)
        with self._session_factory() as session:
            return list(session.scalars(stmt))

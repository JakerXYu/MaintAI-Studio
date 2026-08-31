"""ModelLifecycleState repository (SQLAlchemy 2, session-factory based).

Mirrors the other repositories in this codebase: every method opens its own
short-lived session unless an external ``session`` is supplied, so the P1
registry service can persist a lifecycle change and its audit event in one
transaction. One row exists per registered model, so ``create`` (and the
``registered_model_id`` unique constraint) enforce the one-per-model invariant.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from maintai.db.models import ModelLifecycleState


class ModelLifecycleRepository:
    """Persistent access to :class:`maintai.db.models.ModelLifecycleState` rows."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create(
        self, lifecycle: ModelLifecycleState, *, session: Session | None = None
    ) -> ModelLifecycleState:
        """Insert a new lifecycle row and return the flushed instance."""
        if session is not None:
            session.add(lifecycle)
            session.flush()
            return lifecycle
        with self._session_factory.begin() as session:
            session.add(lifecycle)
            session.flush()
            return lifecycle

    def get(
        self, lifecycle_id: str, *, session: Session | None = None
    ) -> ModelLifecycleState | None:
        """Return a lifecycle row by primary key, or ``None`` when absent."""
        if session is not None:
            return session.get(ModelLifecycleState, lifecycle_id)
        with self._session_factory() as session:
            return session.get(ModelLifecycleState, lifecycle_id)

    def get_by_registered_model(
        self, registered_model_id: str, *, session: Session | None = None
    ) -> ModelLifecycleState | None:
        """Return the single lifecycle row for a registered model, if any."""
        stmt = select(ModelLifecycleState).where(
            ModelLifecycleState.registered_model_id == registered_model_id
        )
        if session is not None:
            return session.scalar(stmt)
        with self._session_factory() as session:
            return session.scalar(stmt)

    def list(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
        session: Session | None = None,
    ) -> list[ModelLifecycleState]:
        """Return lifecycle rows newest-first with clamped pagination and a status filter."""
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        stmt = (
            select(ModelLifecycleState)
            .order_by(ModelLifecycleState.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if status is not None:
            stmt = stmt.where(ModelLifecycleState.status == status)
        if session is not None:
            return list(session.scalars(stmt))
        with self._session_factory() as session:
            return list(session.scalars(stmt))

    def update(
        self, lifecycle: ModelLifecycleState, *, session: Session | None = None
    ) -> ModelLifecycleState:
        """Persist a status transition (challenger/champion/archived) and return the merged row."""
        if session is not None:
            merged = session.merge(lifecycle)
            session.flush()
            return merged
        with self._session_factory.begin() as session:
            merged = session.merge(lifecycle)
            session.flush()
            return merged

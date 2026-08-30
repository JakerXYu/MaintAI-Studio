"""MonitoringRun repository (SQLAlchemy 2, session-factory based).

Mirrors the other repositories in this codebase: every method opens its own
short-lived session unless an external ``session`` is supplied, so the monitoring
application service can persist a monitoring run and its audit event in a single
transaction. Runs are immutable once written except through ``update``, which is
provided for service-managed lifecycle changes (completed status / results).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from maintai.db.models import MonitoringRun


class MonitoringRepository:
    """Persistent access to :class:`maintai.db.models.MonitoringRun` rows."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create(
        self, run: MonitoringRun, *, session: Session | None = None
    ) -> MonitoringRun:
        """Insert a new monitoring run and return the flushed instance."""
        if session is not None:
            session.add(run)
            session.flush()
            return run
        with self._session_factory.begin() as session:
            session.add(run)
            session.flush()
            return run

    def get(self, run_id: str, *, session: Session | None = None) -> MonitoringRun | None:
        """Return a monitoring run by primary key, or ``None`` when absent."""
        if session is not None:
            return session.get(MonitoringRun, run_id)
        with self._session_factory() as session:
            return session.get(MonitoringRun, run_id)

    def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        registered_model_id: str | None = None,
        session: Session | None = None,
    ) -> list[MonitoringRun]:
        """Return monitoring runs ordered by newest-first with clamped pagination."""
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        stmt = (
            select(MonitoringRun)
            .order_by(MonitoringRun.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if registered_model_id is not None:
            stmt = stmt.where(MonitoringRun.registered_model_id == registered_model_id)
        if session is not None:
            return list(session.scalars(stmt))
        with self._session_factory() as session:
            return list(session.scalars(stmt))

    def update(
        self, run: MonitoringRun, *, session: Session | None = None
    ) -> MonitoringRun:
        """Persist changes to an existing monitoring run and return the merged row."""
        if session is not None:
            merged = session.merge(run)
            session.flush()
            return merged
        with self._session_factory.begin() as session:
            merged = session.merge(run)
            session.flush()
            return merged

"""Persistent audit repository.

All writes go through SQLAlchemy; no raw SQL. Audit events are immutable once
written (no update/delete helpers are provided).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from maintai.db.models import AuditEvent


class AuditRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def add(self, event: AuditEvent, *, session: Session | None = None) -> AuditEvent:
        if session is not None:
            session.add(event)
            session.flush()
            return event
        with self._session_factory() as session:
            session.add(event)
            session.commit()
            session.refresh(event)
            return event

    def list(
        self,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        limit = max(1, min(limit, 500))
        stmt = select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(limit)
        if entity_type is not None:
            stmt = stmt.where(AuditEvent.entity_type == entity_type)
        if entity_id is not None:
            stmt = stmt.where(AuditEvent.entity_id == entity_id)
        with self._session_factory() as session:
            return list(session.scalars(stmt))

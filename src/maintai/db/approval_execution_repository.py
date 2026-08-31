"""ApprovalExecution repository (SQLAlchemy 2, session-factory based).

Mirrors the other repositories in this codebase: every method opens its own
short-lived session unless an external ``session`` is supplied, so the owning P1
executor can persist an execution receipt and its audit event in one
transaction. Receipts are immutable once written (no update/delete) and each
``approval_id`` can be executed at most once (unique constraint).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from maintai.db.models import ApprovalExecution


class ApprovalExecutionRepository:
    """Persistent access to :class:`maintai.db.models.ApprovalExecution` rows."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create(
        self, execution: ApprovalExecution, *, session: Session | None = None
    ) -> ApprovalExecution:
        """Insert a new execution receipt and return the flushed instance."""
        if session is not None:
            session.add(execution)
            session.flush()
            return execution
        with self._session_factory.begin() as session:
            session.add(execution)
            session.flush()
            return execution

    def get(
        self, execution_id: str, *, session: Session | None = None
    ) -> ApprovalExecution | None:
        """Return an execution receipt by primary key, or ``None`` when absent."""
        if session is not None:
            return session.get(ApprovalExecution, execution_id)
        with self._session_factory() as session:
            return session.get(ApprovalExecution, execution_id)

    def get_by_approval(
        self, approval_id: str, *, session: Session | None = None
    ) -> ApprovalExecution | None:
        """Return the receipt for an approval, if the action has been executed."""
        stmt = select(ApprovalExecution).where(
            ApprovalExecution.approval_id == approval_id
        )
        if session is not None:
            return session.scalar(stmt)
        with self._session_factory() as session:
            return session.scalar(stmt)

    def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        session: Session | None = None,
    ) -> list[ApprovalExecution]:
        """Return execution receipts newest-first with clamped pagination."""
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        stmt = (
            select(ApprovalExecution)
            .order_by(ApprovalExecution.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if session is not None:
            return list(session.scalars(stmt))
        with self._session_factory() as session:
            return list(session.scalars(stmt))

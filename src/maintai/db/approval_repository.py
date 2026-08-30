"""ApprovalRequest repository (SQLAlchemy 2, session-factory based).

Mirrors the other repositories in this codebase: every method opens its own
short-lived session unless an external ``session`` is supplied, so callers can
share a transaction with an audit record (a proposal or a decision plus its
audit event commit atomically).

The ``transition`` helper is a compare-and-swap: it updates a row only while
``id`` matches, ``version`` equals the expected value, and ``status`` is still
``pending``. Any other outcome (missing row, stale version, already decided)
raises :class:`ApprovalConflictError` so a concurrent or duplicate decision can
never silently win.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from maintai.db.models import APPROVAL_STATUS_PENDING, ApprovalRequest, utcnow


class ApprovalConflictError(Exception):
    """Raised when a compare-and-swap transition fails.

    Covers a missing approval id, a stale ``version`` (concurrent writer), and a
    request that is no longer ``pending`` (already decided).
    """


class ApprovalRepository:
    """Persistent access to :class:`maintai.db.models.ApprovalRequest` rows."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create(
        self, approval: ApprovalRequest, *, session: Session | None = None
    ) -> ApprovalRequest:
        """Insert a new approval request and return the flushed instance."""
        if session is not None:
            session.add(approval)
            session.flush()
            return approval
        with self._session_factory.begin() as session:
            session.add(approval)
            session.flush()
            return approval

    def get(
        self, approval_id: str, *, session: Session | None = None
    ) -> ApprovalRequest | None:
        """Return an approval request by primary key, or ``None`` when absent."""
        if session is not None:
            return session.get(ApprovalRequest, approval_id)
        with self._session_factory() as session:
            return session.get(ApprovalRequest, approval_id)

    def find_pending(
        self,
        *,
        action_type: str,
        entity_type: str,
        entity_id: str,
        session: Session | None = None,
    ) -> ApprovalRequest | None:
        """Find an equivalent pending request, if one exists."""
        stmt = select(ApprovalRequest).where(
            ApprovalRequest.action_type == action_type,
            ApprovalRequest.entity_type == entity_type,
            ApprovalRequest.entity_id == entity_id,
            ApprovalRequest.status == APPROVAL_STATUS_PENDING,
        )
        if session is not None:
            return session.scalar(stmt)
        with self._session_factory() as session:
            return session.scalar(stmt)

    def list(
        self,
        *,
        action_type: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        status: str | None = None,
        requested_by_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
        session: Session | None = None,
    ) -> list[ApprovalRequest]:
        """Return approvals newest-first with clamped pagination and filters."""
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        stmt = (
            select(ApprovalRequest)
            .order_by(ApprovalRequest.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if action_type is not None:
            stmt = stmt.where(ApprovalRequest.action_type == action_type)
        if entity_type is not None:
            stmt = stmt.where(ApprovalRequest.entity_type == entity_type)
        if entity_id is not None:
            stmt = stmt.where(ApprovalRequest.entity_id == entity_id)
        if status is not None:
            stmt = stmt.where(ApprovalRequest.status == status)
        if requested_by_type is not None:
            stmt = stmt.where(ApprovalRequest.requested_by_type == requested_by_type)
        if session is not None:
            return list(session.scalars(stmt))
        with self._session_factory() as session:
            return list(session.scalars(stmt))

    def transition(
        self,
        approval_id: str,
        expected_version: int,
        *,
        status: str,
        decision_payload: dict[str, Any] | None = None,
        decided_by: str | None = None,
        reason: str | None = None,
        decided_at: datetime | None = None,
        session: Session | None = None,
    ) -> ApprovalRequest:
        """Compare-and-swap a pending approval into a terminal status.

        The ``UPDATE`` is guarded by ``id``, ``version == expected_version`` and
        ``status == pending``; ``version`` is bumped atomically. Raises
        :class:`ApprovalConflictError` when the guard matches zero rows (missing
        id, stale version, or already decided).
        """
        values: dict[str, Any] = {
            "status": status,
            "decision_payload": decision_payload,
            "decided_by": decided_by,
            "reason": reason,
            "decided_at": decided_at if decided_at is not None else utcnow(),
            "version": ApprovalRequest.version + 1,
        }

        def _run(s: Session) -> ApprovalRequest:
            result = s.execute(
                update(ApprovalRequest)
                .where(
                    ApprovalRequest.id == approval_id,
                    ApprovalRequest.version == expected_version,
                    ApprovalRequest.status == APPROVAL_STATUS_PENDING,
                )
                .values(**values)
            )
            if result.rowcount != 1:
                raise ApprovalConflictError(
                    f"approval {approval_id!r} transition failed: "
                    "it was already decided or its version changed"
                )
            refreshed = s.get(ApprovalRequest, approval_id)
            if refreshed is None:  # pragma: no cover - guarded by rowcount
                raise ApprovalConflictError(
                    f"approval {approval_id!r} transition failed: not found"
                )
            return refreshed

        if session is not None:
            return _run(session)
        with self._session_factory.begin() as session:
            return _run(session)

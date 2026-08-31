"""MockCMMSWorkOrder repository (SQLAlchemy 2, session-factory based).

Mirrors the other repositories in this codebase: every method opens its own
short-lived session unless an external ``session`` is supplied, so the owning P1
service can persist a draft and its audit event in one transaction. Each draft
is tied to exactly one approval request (``approval_id`` unique); ``update`` is
provided for the service-managed ``draft -> approved`` transition.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from maintai.db.models import MockCMMSWorkOrder


class MockCMMSWorkOrderRepository:
    """Persistent access to :class:`maintai.db.models.MockCMMSWorkOrder` rows."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create(
        self, work_order: MockCMMSWorkOrder, *, session: Session | None = None
    ) -> MockCMMSWorkOrder:
        """Insert a new mock CMMS work order and return the flushed instance."""
        if session is not None:
            session.add(work_order)
            session.flush()
            return work_order
        with self._session_factory.begin() as session:
            session.add(work_order)
            session.flush()
            return work_order

    def get(
        self, work_order_id: str, *, session: Session | None = None
    ) -> MockCMMSWorkOrder | None:
        """Return a work order by primary key, or ``None`` when absent."""
        if session is not None:
            return session.get(MockCMMSWorkOrder, work_order_id)
        with self._session_factory() as session:
            return session.get(MockCMMSWorkOrder, work_order_id)

    def get_by_approval(
        self, approval_id: str, *, session: Session | None = None
    ) -> MockCMMSWorkOrder | None:
        """Return the work order for an approval, if one has been drafted."""
        stmt = select(MockCMMSWorkOrder).where(
            MockCMMSWorkOrder.approval_id == approval_id
        )
        if session is not None:
            return session.scalar(stmt)
        with self._session_factory() as session:
            return session.scalar(stmt)

    def list(
        self,
        *,
        status: str | None = None,
        asset_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
        session: Session | None = None,
    ) -> list[MockCMMSWorkOrder]:
        """Return work orders newest-first with clamped pagination and filters."""
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        stmt = (
            select(MockCMMSWorkOrder)
            .order_by(MockCMMSWorkOrder.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if status is not None:
            stmt = stmt.where(MockCMMSWorkOrder.status == status)
        if asset_id is not None:
            stmt = stmt.where(MockCMMSWorkOrder.asset_id == asset_id)
        if session is not None:
            return list(session.scalars(stmt))
        with self._session_factory() as session:
            return list(session.scalars(stmt))

    def update(
        self, work_order: MockCMMSWorkOrder, *, session: Session | None = None
    ) -> MockCMMSWorkOrder:
        """Persist a status transition (draft -> approved) and return the merged row."""
        if session is not None:
            merged = session.merge(work_order)
            session.flush()
            return merged
        with self._session_factory.begin() as session:
            merged = session.merge(work_order)
            session.flush()
            return merged

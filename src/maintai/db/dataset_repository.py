"""Dataset repository (SQLAlchemy 2, session-factory based)."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from maintai.db.models import Dataset


class DatasetRepository:
    """Persistent access to :class:`maintai.db.models.Dataset` rows.

    Every method opens its own short-lived session from the injected factory,
    matching the existing repository style in this codebase.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create(self, dataset: Dataset, *, session: Session | None = None) -> Dataset:
        """Insert a new dataset row and return the refreshed instance."""
        if session is not None:
            session.add(dataset)
            session.flush()
            return dataset
        with self._session_factory() as session:
            session.add(dataset)
            session.commit()
            session.refresh(dataset)
            return dataset

    def get(self, dataset_id: str) -> Dataset | None:
        """Return a dataset by primary key, or ``None`` when absent."""
        with self._session_factory() as session:
            return session.get(Dataset, dataset_id)

    def find_by_hash(self, file_hash: str) -> Dataset | None:
        """Return the first dataset whose content hash matches, if any."""
        stmt = select(Dataset).where(Dataset.file_hash == file_hash)
        with self._session_factory() as session:
            return session.scalars(stmt).first()

    def list(self, *, limit: int = 100, offset: int = 0) -> list[Dataset]:
        """Return datasets ordered by newest-first with clamped pagination."""
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        stmt = (
            select(Dataset)
            .order_by(Dataset.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        with self._session_factory() as session:
            return list(session.scalars(stmt))

    def update(self, dataset: Dataset, *, session: Session | None = None) -> Dataset:
        """Persist changes to an existing dataset and return the merged row."""
        if session is not None:
            merged = session.merge(dataset)
            session.flush()
            return merged
        with self._session_factory() as session:
            merged = session.merge(dataset)
            session.commit()
            session.refresh(merged)
            return merged

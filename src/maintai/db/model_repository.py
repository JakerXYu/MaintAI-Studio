"""RegisteredModel / ModelRun repository (SQLAlchemy 2, session-factory based).

Mirrors :class:`maintai.db.dataset_repository.DatasetRepository` and
:class:`maintai.db.experiment_repository.ExperimentRepository`. Every method
opens its own short-lived session unless an external ``session`` is supplied, so
callers can share a transaction with an audit record (e.g. a model register plus
its audit event, or a demo deploy plus its sibling demotion, commit atomically).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from maintai.db.models import ModelRun, RegisteredModel


class ModelRepository:
    """Persistent access to :class:`RegisteredModel` and :class:`ModelRun` rows."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create(
        self, model: RegisteredModel, *, session: Session | None = None
    ) -> RegisteredModel:
        """Insert a new registered-model row and return the flushed instance."""
        if session is not None:
            session.add(model)
            session.flush()
            return model
        with self._session_factory.begin() as session:
            session.add(model)
            session.flush()
            return model

    def get(
        self, registered_id: str, *, session: Session | None = None
    ) -> RegisteredModel | None:
        """Return a registered model by primary key, or ``None`` when absent."""
        if session is not None:
            return session.get(RegisteredModel, registered_id)
        with self._session_factory() as session:
            return session.get(RegisteredModel, registered_id)

    def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        session: Session | None = None,
    ) -> list[RegisteredModel]:
        """Return registered models ordered by newest-first with clamped pagination."""
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        stmt = (
            select(RegisteredModel)
            .order_by(RegisteredModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if session is not None:
            return list(session.scalars(stmt))
        with self._session_factory() as session:
            return list(session.scalars(stmt))

    def update(
        self, model: RegisteredModel, *, session: Session | None = None
    ) -> RegisteredModel:
        """Persist changes to an existing registered model and return the merged row."""
        if session is not None:
            merged = session.merge(model)
            session.flush()
            return merged
        with self._session_factory.begin() as session:
            merged = session.merge(model)
            session.flush()
            return merged

    def find_by_model_run(
        self, model_run_id: str, *, session: Session | None = None
    ) -> list[RegisteredModel]:
        """Return every registered model derived from a single ``ModelRun``."""
        stmt = select(RegisteredModel).where(RegisteredModel.model_run_id == model_run_id)
        if session is not None:
            return list(session.scalars(stmt))
        with self._session_factory() as session:
            return list(session.scalars(stmt))

    def list_by_name(
        self, name: str, *, session: Session | None = None
    ) -> list[RegisteredModel]:
        """Return all versions of a registered model name in creation order."""
        stmt = (
            select(RegisteredModel)
            .where(RegisteredModel.name == name)
            .order_by(RegisteredModel.created_at.asc())
        )
        if session is not None:
            stmt = stmt.with_for_update()
            return list(session.scalars(stmt))
        with self._session_factory() as session:
            return list(session.scalars(stmt))

    def get_model_run(self, model_run_id: str) -> ModelRun | None:
        """Return a model run by primary key, or ``None`` when absent."""
        with self._session_factory() as session:
            return session.get(ModelRun, model_run_id)

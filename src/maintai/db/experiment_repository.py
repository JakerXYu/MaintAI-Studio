"""Experiment / ModelRun repository (SQLAlchemy 2, session-factory based).

Mirrors :class:`maintai.db.dataset_repository.DatasetRepository`. Every method
opens its own short-lived session unless an external ``session`` is supplied, so
callers can share a transaction with an audit record (e.g. an experiment create
plus its audit event commit atomically).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from maintai.db.models import Experiment, ModelRun


class ExperimentRepository:
    """Persistent access to :class:`Experiment` and :class:`ModelRun` rows."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create(self, experiment: Experiment, *, session: Session | None = None) -> Experiment:
        """Insert a new experiment row and return the flushed instance."""
        if session is not None:
            session.add(experiment)
            session.flush()
            return experiment
        with self._session_factory() as session:
            session.add(experiment)
            session.commit()
            session.refresh(experiment)
            return experiment

    def get(self, experiment_id: str) -> Experiment | None:
        """Return an experiment by primary key, or ``None`` when absent."""
        with self._session_factory() as session:
            return session.get(Experiment, experiment_id)

    def list(self, *, limit: int = 100, offset: int = 0) -> list[Experiment]:
        """Return experiments ordered by newest-first with clamped pagination."""
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        stmt = (
            select(Experiment)
            .order_by(Experiment.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        with self._session_factory() as session:
            return list(session.scalars(stmt))

    def update(self, experiment: Experiment, *, session: Session | None = None) -> Experiment:
        """Persist changes to an existing experiment and return the merged row."""
        if session is not None:
            merged = session.merge(experiment)
            session.flush()
            return merged
        with self._session_factory() as session:
            merged = session.merge(experiment)
            session.commit()
            session.refresh(merged)
            return merged

    def create_model_run(self, model_run: ModelRun, *, session: Session | None = None) -> ModelRun:
        """Insert a new model-run row and return the flushed instance."""
        if session is not None:
            session.add(model_run)
            session.flush()
            return model_run
        with self._session_factory() as session:
            session.add(model_run)
            session.commit()
            session.refresh(model_run)
            return model_run

    def list_model_runs(self, experiment_id: str) -> list[ModelRun]:
        """Return the model runs of one experiment ordered by creation time."""
        stmt = (
            select(ModelRun)
            .where(ModelRun.experiment_id == experiment_id)
            .order_by(ModelRun.created_at.asc())
        )
        with self._session_factory() as session:
            return list(session.scalars(stmt))

"""Experiment REST API routes.

FastAPI is the only business entry point. This module exposes the P0 experiment
lifecycle endpoints over HTTP and maps application-service errors to stable HTTP
status codes. The POST endpoint accepts only a fixed request body (``dataset_id``,
``model_names``, ``minimum_recall``) — never arbitrary run ids, shell commands, or
filesystem paths — and schedules the single in-process training worker via
FastAPI ``BackgroundTasks``.

Every mutation (create) delegates to
:class:`~maintai.application.experiments.ExperimentService`, which owns audit
recording; this layer never writes audit events itself. Response models expose
only ``runs:/`` MLflow URIs and never raw filesystem paths.
"""

from __future__ import annotations

import threading
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.orm import Session, sessionmaker

from maintai.application.datasets import DatasetNotFoundError, DatasetService, StorageError
from maintai.application.experiments import (
    DatasetNotReadyError,
    ExperimentNotFoundError,
    ExperimentService,
    ExperimentServiceError,
)
from maintai.audit.service import get_audit_service
from maintai.config import Settings
from maintai.data.ingest import DataIngestError
from maintai.data.split import SplitConfig, SplitError
from maintai.db.experiment_repository import ExperimentRepository
from maintai.ml.schemas import MLSettings
from maintai.mlops.tracker import MLflowTracker

_MLFLOW_RUNS_PREFIX = "runs:/"


def _runs_uri_only(value: object) -> str | None:
    """Return an MLflow URI only when it is a ``runs:/`` reference, else ``None``.

    This guarantees a raw filesystem path or a ``file://``/``s3://`` URI can never
    leak into a response model even if a bug upstream ever produced one.
    """
    if not isinstance(value, str):
        return None
    if not value.startswith(_MLFLOW_RUNS_PREFIX):
        return None
    return value


class ExperimentCreateRequest(BaseModel):
    """Request body for creating a queued experiment."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)
    model_names: list[str] | None = Field(default=None, max_length=3)
    minimum_recall: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("model_names")
    @classmethod
    def _model_names_nonempty(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        if any(not isinstance(name, str) or not name.strip() for name in value):
            raise ValueError("model_names entries must be non-empty strings")
        return value


class ExperimentSummary(BaseModel):
    """Stable JSON shape for an experiment summary (no JSON blobs, no runs)."""

    model_config = ConfigDict(extra="ignore")

    id: str
    dataset_id: str | None = None
    name: str
    task_type: str | None = None
    status: str
    recommended_model: str | None = None
    recommended_run_id: str | None = None
    config_hash: str | None = None
    primary_metric: str | None = None
    value: float | None = None
    error_message: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


class ModelRunResponse(BaseModel):
    """Stable JSON shape for a persisted ``ModelRun`` row (runs:/ URIs only)."""

    model_config = ConfigDict(extra="ignore")

    id: str
    experiment_id: str | None = None
    mlflow_run_id: str | None = None
    model_name: str
    config_hash: str | None = None
    status: str
    primary_metric: str | None = None
    primary_metric_value: float | None = None
    metrics: dict[str, Any] | None = None
    confusion_matrix: Any | None = None
    artifact_uri: str | None = None
    model_uri: str | None = None
    feature_names: list[str] | None = None
    training_time_seconds: float | None = None
    created_at: str | None = None

    @field_validator("artifact_uri", "model_uri", mode="before")
    @classmethod
    def _only_runs_uris(cls, value: object) -> str | None:
        return _runs_uri_only(value)


class ExperimentResponse(ExperimentSummary):
    """Full experiment detail including the training-plan snapshot and runs."""

    training_plan: dict[str, Any] | None = None
    model_runs: list[ModelRunResponse] | None = None


class ComparisonResponse(BaseModel):
    """Stable JSON shape for an experiment's best-model comparison."""

    model_config = ConfigDict(extra="allow")

    experiment_id: str
    task: str | None = None
    primary_metric: str | None = None
    best_model: str | None = None
    ranking: list[str] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    recommended_run_id: str | None = None
    value: float | None = None


class SingleTrainingCoordinator:
    """Serialize CPU-heavy experiment runs inside one API process.

    This is intentionally not a durable queue. It enforces the documented P0
    capacity of one active training run without introducing Celery/Redis.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def run(self, service: ExperimentService, experiment_id: str) -> dict[str, Any]:
        with self._lock:
            return service.run(experiment_id)


def build_experiment_service(
    session_factory: sessionmaker[Session],
    settings: Settings,
    dataset_service: DatasetService,
) -> ExperimentService:
    """Construct the production ``ExperimentService`` from settings/session/repos.

    Maps the flat ``Settings`` split/ml rule models onto the deterministic
    ``SplitConfig`` / ``MLSettings`` contracts and wires the MLflow tracker to the
    configured tracking URI and experiment name.
    """
    return ExperimentService(
        session_factory=session_factory,
        dataset_service=dataset_service,
        experiment_repository=ExperimentRepository(session_factory),
        audit=get_audit_service(session_factory),
        tracker=MLflowTracker(
            settings.mlflow_tracking_uri,
            settings.mlflow_experiment_name,
        ),
        artifact_root=settings.artifact_storage_path,
        ml_settings=MLSettings(**settings.ml.model_dump()),
        split_settings=SplitConfig(**settings.split.model_dump()),
    )


def build_experiments_router(
    service: ExperimentService,
    coordinator: SingleTrainingCoordinator | None = None,
) -> APIRouter:
    """Build the experiment router bound to a single ``ExperimentService``."""
    router = APIRouter(prefix="/experiments", tags=["experiments"])
    coordinator = coordinator or SingleTrainingCoordinator()

    @router.post(
        "",
        response_model=ExperimentResponse,
        response_model_exclude_none=True,
        status_code=202,
    )
    def create_experiment(
        body: ExperimentCreateRequest,
        background_tasks: BackgroundTasks,
    ) -> Any:
        try:
            snapshot = service.create(body.dataset_id, body.model_names, body.minimum_recall)
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except DatasetNotReadyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ExperimentServiceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SplitError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except StorageError:
            raise HTTPException(
                status_code=500, detail="dataset storage is unavailable"
            ) from None
        except DataIngestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        background_tasks.add_task(coordinator.run, service, snapshot["id"])
        return snapshot

    @router.get(
        "",
        response_model=list[ExperimentSummary],
        response_model_exclude_none=True,
    )
    def list_experiments(
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> list[ExperimentSummary]:
        return service.list(limit=limit, offset=offset)

    @router.get(
        "/{experiment_id}",
        response_model=ExperimentResponse,
        response_model_exclude_none=True,
    )
    def get_experiment(experiment_id: str) -> ExperimentResponse:
        try:
            return service.get(experiment_id)
        except ExperimentNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/{experiment_id}/comparison", response_model=ComparisonResponse)
    def get_comparison(experiment_id: str) -> ComparisonResponse:
        try:
            return service.comparison(experiment_id)
        except ExperimentNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return router

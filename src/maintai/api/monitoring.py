"""Monitoring REST API routes (P1 deterministic monitoring).

Exposes ``POST /api/v1/monitoring/runs`` (synchronous, demo scale), a list
endpoint, and a detail endpoint. Every mutation delegates to
:class:`~maintai.application.monitoring.MonitoringService`, which owns the
compute and the audit writes; this layer never writes audit events or touches
repositories directly. Errors map to stable 404/409/422/500 with path-free
details.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.orm import Session, sessionmaker

from maintai.application.datasets import DatasetService
from maintai.application.monitoring import (
    MonitoringDatasetNotFoundError,
    MonitoringModelNotFoundError,
    MonitoringNotDeployableError,
    MonitoringRunNotFoundError,
    MonitoringService,
    MonitoringServiceError,
    MonitoringSourceError,
)
from maintai.audit.service import get_audit_service
from maintai.config import Settings
from maintai.db.experiment_repository import ExperimentRepository
from maintai.db.model_repository import ModelRepository
from maintai.db.monitoring_repository import MonitoringRepository
from maintai.monitoring.config import anomaly_config, drift_config, recommend_config
from maintai.monitoring.contracts import ReplayKind

_MONITORING_FAILED_DETAIL = "monitoring run failed"


class MonitoringRunRequest(BaseModel):
    """Body for a monitoring run: ``model_id`` + exactly one production source.

    ``performance_drop`` / ``baseline_anomaly_rate`` / ``schedule`` / ``manual``
    are optional recommendation inputs. ``manual`` only adds a manual trigger to
    the recommendation — it never forces training, deployment, or an approval.
    Arbitrary fields are rejected outright.
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1)
    production_dataset_id: str | None = Field(default=None, min_length=1)
    replay_kind: ReplayKind | None = None
    performance_drop: float | None = Field(default=None, ge=0.0)
    baseline_anomaly_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    schedule: bool | None = None
    manual: bool | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> MonitoringRunRequest:
        has_production = self.production_dataset_id is not None
        has_replay = self.replay_kind is not None
        if has_production == has_replay:
            raise ValueError(
                "exactly one of production_dataset_id or replay_kind is required"
            )
        return self


class MonitoringRunResponse(BaseModel):
    """Stable JSON shape for a monitoring run (no raw rows, no filesystem paths)."""

    model_config = ConfigDict(extra="ignore")

    id: str
    registered_model_id: str
    dataset_id: str
    production_dataset_id: str | None = None
    replay_kind: str | None = None
    status: str
    drift: dict[str, Any] | None = None
    anomaly: dict[str, Any] | None = None
    recommendation: dict[str, Any] | None = None
    input_summary: dict[str, Any] | None = None
    error_message: str | None = None
    created_at: str | None = None
    completed_at: str | None = None


def build_monitoring_service(
    session_factory: sessionmaker[Session],
    settings: Settings,
    dataset_service: DatasetService,
) -> MonitoringService:
    """Construct the production ``MonitoringService`` from settings/session/repos.

    Maps the ``Settings`` monitoring block onto the deterministic monitoring
    configs via :mod:`maintai.monitoring.config`, and wires the shared dataset
    service plus fresh model/experiment/monitoring repositories and the audit
    service. No network or MLflow access happens here.
    """
    return MonitoringService(
        session_factory=session_factory,
        dataset_service=dataset_service,
        model_repository=ModelRepository(session_factory),
        experiment_repository=ExperimentRepository(session_factory),
        repository=MonitoringRepository(session_factory),
        audit=get_audit_service(session_factory),
        anomaly_config=anomaly_config(settings),
        drift_config=drift_config(settings),
        recommend_config=recommend_config(settings),
    )


def build_monitoring_router(service: MonitoringService) -> APIRouter:
    """Build the monitoring router bound to one ``MonitoringService``."""
    router = APIRouter(prefix="/monitoring", tags=["monitoring"])

    @router.post("/runs", response_model=MonitoringRunResponse, status_code=201)
    def create_run(body: MonitoringRunRequest) -> MonitoringRunResponse:
        try:
            return service.run(
                body.model_id,
                production_dataset_id=body.production_dataset_id,
                replay_kind=body.replay_kind,
                performance_drop=body.performance_drop,
                baseline_anomaly_rate=body.baseline_anomaly_rate,
                schedule_due=bool(body.schedule),
                manual=bool(body.manual),
            )
        except MonitoringModelNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except MonitoringNotDeployableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except MonitoringSourceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except MonitoringDatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except MonitoringServiceError:
            raise HTTPException(status_code=500, detail=_MONITORING_FAILED_DETAIL) from None

    @router.get("/runs", response_model=list[MonitoringRunResponse])
    def list_runs(
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> list[MonitoringRunResponse]:
        return service.list(limit=limit, offset=offset)

    @router.get("/runs/{run_id}", response_model=MonitoringRunResponse)
    def get_run(run_id: str) -> MonitoringRunResponse:
        try:
            return service.get(run_id)
        except MonitoringRunNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return router

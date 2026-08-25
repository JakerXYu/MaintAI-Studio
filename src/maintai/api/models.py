"""Model registry + prediction REST API routes (P0 candidate → demo deploy).

FastAPI is the only business entry point. This module exposes the model registry
lifecycle (list/get/register/demo-deploy) and the deployed-model inference
endpoints (single + batch) over HTTP, and maps application-service errors to
stable HTTP status codes without leaking filesystem paths or internal details.

Design constraints (mirrors :mod:`maintai.api.datasets` and
:mod:`maintai.api.experiments`):

* The API layer never writes audit events itself — every mutation delegates to
  :class:`~maintai.application.models.ModelRegistryService` or
  :class:`~maintai.application.predictions.PredictionService`, which own audit
  recording.
* Response models expose only ``models:/`` MLflow URIs (never a ``file://`` URI
  or a raw filesystem path); a defensive ``field_validator`` filters anything
  that does not carry the ``models:/`` prefix.
* P0 never promotes to ``champion`` or ``Production``. The only deployment
  mutation is ``deploy-demo``, which flips a registered model to
  ``demo_deployed`` — human-approval promotion belongs to P1 and has no route.
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.orm import Session, sessionmaker

from maintai.application.models import (
    ExperimentNotEligibleError,
    ModelRegistryService,
    ModelRegistryServiceError,
    ModelRunNotFoundError,
    RegisterConflictError,
    RegisteredModelNotFoundError,
)
from maintai.application.predictions import (
    InvalidRecordsError,
    ModelNotDeployedError,
    PredictionError,
    PredictionModelNotFoundError,
    PredictionService,
)
from maintai.audit.service import get_audit_service
from maintai.config import Settings
from maintai.db.experiment_repository import ExperimentRepository
from maintai.db.model_repository import ModelRepository
from maintai.db.prediction_repository import PredictionRepository
from maintai.mlops.registry import MLflowRegistry

_MODELS_PREFIX = "models:/"

# Stable, path-free 500 details: the application services already render
# path-free messages, but the API layer returns a fixed string so a backend or
# artifact failure can never leak a filesystem path, URI, or traceback.
_REGISTRATION_FAILED_DETAIL = "model registration failed"
_PREDICTION_FAILED_DETAIL = "prediction failed"

# Registered model names mirror the MLflow registry rule: readable, filesystem
# safe characters only. Path separators and control characters are rejected so a
# malformed ``name`` is a client (422) error rather than a registry failure.
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_NAME_MAX_LEN = 255


def _models_uri_only(value: object) -> str | None:
    """Return an MLflow URI only when it is a ``models:/`` reference, else ``None``.

    Guarantees a raw filesystem path, an absolute ``file://``/``s3://`` URI, or
    any other non-``models:/`` value can never leak into a response model.
    """
    if not isinstance(value, str):
        return None
    if not value.startswith(_MODELS_PREFIX):
        return None
    return value


# -- request/response models --------------------------------------------------


class RegisteredModelResponse(BaseModel):
    """Stable JSON shape for a registered model (``models:/`` URIs only)."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    version: str
    model_run_id: str | None = None
    experiment_id: str | None = None
    mlflow_model_uri: str | None = None
    alias: str | None = None
    deployment_status: str
    created_at: str | None = None
    updated_at: str | None = None

    @field_validator("mlflow_model_uri", mode="before")
    @classmethod
    def _only_models_uris(cls, value: object) -> str | None:
        return _models_uri_only(value)


class RegisterModelRequest(BaseModel):
    """Request body for registering a model run as a ``candidate``.

    ``name`` is optional (defaults to the model run's own name) and, when
    provided, must be a filesystem-safe registry name. Arbitrary fields are
    rejected outright.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=_NAME_MAX_LEN)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not _NAME_RE.match(value):
            raise ValueError("name may only contain letters, digits, '.', '_' and '-'")
        return value


class PredictRequest(BaseModel):
    """Base request body for inference (``model_id`` + a bounded record list)."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1)
    records: list[dict[str, Any]]


class PredictSingleRequest(PredictRequest):
    """Single inference accepts exactly one record."""

    records: list[dict[str, Any]] = Field(min_length=1, max_length=1)


class PredictBatchRequest(PredictRequest):
    """Batch inference accepts between 1 and 100 records."""

    records: list[dict[str, Any]] = Field(min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    """Stable JSON shape for an inference response (JSON-safe per-record results)."""

    model_config = ConfigDict(extra="ignore")

    model_id: str
    model_version: str
    count: int
    records: list[dict[str, Any]]


# -- builders ----------------------------------------------------------------


def build_model_services(
    session_factory: sessionmaker[Session],
    settings: Settings,
) -> tuple[ModelRegistryService, PredictionService]:
    """Construct the production registry + prediction services.

    Both services share a single :class:`~maintai.db.model_repository.ModelRepository`
    (and a single :class:`~maintai.audit.service.AuditService`) so they always
    read the same registered-model rows. The registry service also wires an
    :class:`~maintai.db.experiment_repository.ExperimentRepository` for
    eligibility checks and an :class:`~maintai.mlops.registry.MLflowRegistry`
    bound to the configured tracking URI; the prediction service wires a
    :class:`~maintai.db.prediction_repository.PredictionRepository` and the
    configured random seed. No network is touched here — the MLflow client is
    only created when a registry operation runs.
    """
    model_repository = ModelRepository(session_factory)
    audit = get_audit_service(session_factory)
    registry_service = ModelRegistryService(
        session_factory=session_factory,
        model_repository=model_repository,
        experiment_repository=ExperimentRepository(session_factory),
        audit=audit,
        registry=MLflowRegistry(settings.mlflow_tracking_uri),
        artifact_root=settings.artifact_storage_path,
    )
    prediction_service = PredictionService(
        session_factory=session_factory,
        model_repository=model_repository,
        prediction_repository=PredictionRepository(session_factory),
        audit=audit,
        artifact_root=settings.artifact_storage_path,
        seed=settings.random_seed,
    )
    return registry_service, prediction_service


# -- routers ------------------------------------------------------------------


def build_models_router(service: ModelRegistryService) -> APIRouter:
    """Build the model registry router bound to one ``ModelRegistryService``."""
    router = APIRouter(prefix="/models", tags=["models"])

    @router.get("", response_model=list[RegisteredModelResponse])
    def list_models(
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> list[RegisteredModelResponse]:
        return service.list(limit=limit, offset=offset)

    @router.get("/{registered_id}", response_model=RegisteredModelResponse)
    def get_model(registered_id: str) -> RegisteredModelResponse:
        try:
            return service.get(registered_id)
        except RegisteredModelNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/{model_run_id}/register", response_model=RegisteredModelResponse)
    def register_model(
        model_run_id: str,
        body: RegisterModelRequest,
    ) -> RegisteredModelResponse:
        try:
            return service.register(model_run_id, body.name)
        except ModelRunNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ExperimentNotEligibleError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RegisterConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ModelRegistryServiceError:
            raise HTTPException(
                status_code=500, detail=_REGISTRATION_FAILED_DETAIL
            ) from None

    @router.post(
        "/{registered_id}/deploy-demo",
        response_model=RegisteredModelResponse,
    )
    def deploy_demo(registered_id: str) -> RegisteredModelResponse:
        try:
            return service.deploy_demo(registered_id)
        except RegisteredModelNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return router


def _predict(
    service: PredictionService,
    model_id: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Shared error mapping for the single and batch inference endpoints."""
    try:
        return service.predict(model_id, records)
    except PredictionModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelNotDeployedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidRecordsError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PredictionError:
        raise HTTPException(
            status_code=500, detail=_PREDICTION_FAILED_DETAIL
        ) from None


def build_predict_router(service: PredictionService) -> APIRouter:
    """Build the inference router bound to one ``PredictionService``."""
    router = APIRouter(prefix="/predict", tags=["predict"])

    @router.post("", response_model=PredictionResponse)
    def predict_single(body: PredictSingleRequest) -> PredictionResponse:
        return _predict(service, body.model_id, body.records)

    @router.post("/batch", response_model=PredictionResponse)
    def predict_batch(body: PredictBatchRequest) -> PredictionResponse:
        return _predict(service, body.model_id, body.records)

    return router

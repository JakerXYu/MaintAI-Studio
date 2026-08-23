"""Dataset REST API routes.

FastAPI is the only business entry point. This module exposes the dataset
lifecycle endpoints over HTTP and maps application-service errors to stable
HTTP status codes without leaking storage paths or internal details. Every
mutation (upload / profile / task-recommendation) delegates to
:class:`~maintai.application.datasets.DatasetService`, which owns audit
recording; this layer never writes audit events itself.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session, sessionmaker

from maintai.application.datasets import (
    DatasetNotFoundError,
    DatasetService,
    DuplicateDatasetError,
    StorageError,
)
from maintai.audit.service import get_audit_service
from maintai.config import Settings
from maintai.data.ingest import BYTES_PER_MB, DataIngestError, FileTooLargeError
from maintai.data.leakage import LeakageError
from maintai.data.quality import QualityConfig
from maintai.db.dataset_repository import DatasetRepository


class DatasetResponse(BaseModel):
    """Stable JSON shape for a dataset resource (summary or full detail)."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    name: str
    original_filename: str | None = None
    source_type: str
    file_hash: str | None = None
    size_bytes: int | None = None
    status: str
    row_count: int | None = None
    column_count: int | None = None
    quality_score: float | None = None
    target_column: str | None = None
    asset_id_column: str | None = None
    timestamp_column: str | None = None
    task_type: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    profile: dict[str, Any] | None = None


class TaskRecommendationRequest(BaseModel):
    """Request body for deterministic task recommendation."""

    target_column: str = Field(min_length=1)
    asset_id_column: str | None = None
    timestamp_column: str | None = None


class QualityResponse(BaseModel):
    """Stable JSON shape for a stored data-quality report."""

    model_config = ConfigDict(extra="allow")

    dataset_id: str
    score: float | None = None
    report: dict[str, Any] | None = None


class TaskRecommendationResponse(BaseModel):
    """Stable JSON shape for a task recommendation result."""

    model_config = ConfigDict(extra="allow")

    dataset_id: str
    target_column: str
    task: dict[str, Any]
    leakage: dict[str, Any]
    quality_score: float | None = None


def build_dataset_service(
    session_factory: sessionmaker[Session],
    settings: Settings,
) -> DatasetService:
    """Construct the production ``DatasetService`` from settings/session/audit/storage."""
    return DatasetService(
        repository=DatasetRepository(session_factory),
        session_factory=session_factory,
        audit=get_audit_service(session_factory),
        storage_root=settings.dataset_storage_path,
        max_upload_bytes=settings.max_upload_mb * BYTES_PER_MB,
        quality_config=QualityConfig(**settings.data.model_dump()),
    )


def build_datasets_router(service: DatasetService) -> APIRouter:
    """Build the dataset router bound to a single ``DatasetService`` instance."""
    router = APIRouter(prefix="/datasets", tags=["datasets"])
    # DatasetService keeps its configured upload cap private; read it once here
    # so multipart uploads are bounded to ``max_bytes + 1`` bytes.
    max_upload_bytes = service.max_upload_bytes

    @router.post(
        "/upload",
        response_model=DatasetResponse,
        response_model_exclude_none=True,
        status_code=201,
    )
    async def upload_dataset(file: UploadFile = File(...)) -> Any:
        # Read at most max_upload_bytes + 1 bytes so an oversize payload is
        # detected without buffering unbounded content into memory.
        content = await file.read(max_upload_bytes + 1)
        if len(content) > max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"upload exceeds the {max_upload_bytes} byte limit",
            )
        filename = file.filename or ""
        try:
            return service.upload(content, filename)
        except DuplicateDatasetError as exc:
            return JSONResponse(
                status_code=409,
                content={
                    "detail": str(exc),
                    "existing_dataset_id": exc.existing_dataset_id,
                },
            )
        except FileTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except StorageError:
            raise HTTPException(
                status_code=500, detail="dataset storage is unavailable"
            ) from None
        except DataIngestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("", response_model=list[DatasetResponse], response_model_exclude_none=True)
    def list_datasets(
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> list[DatasetResponse]:
        return service.list(limit=limit, offset=offset)

    @router.get(
        "/{dataset_id}",
        response_model=DatasetResponse,
        response_model_exclude_none=True,
    )
    def get_dataset(dataset_id: str) -> DatasetResponse:
        try:
            return service.get(dataset_id)
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post(
        "/{dataset_id}/profile",
        response_model=DatasetResponse,
        response_model_exclude_none=True,
    )
    def profile_dataset(dataset_id: str) -> DatasetResponse:
        try:
            return service.profile(dataset_id)
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except StorageError:
            raise HTTPException(
                status_code=500, detail="dataset storage is unavailable"
            ) from None
        except DataIngestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/{dataset_id}/quality", response_model=QualityResponse)
    def get_quality(dataset_id: str) -> QualityResponse:
        try:
            return service.get_quality(dataset_id)
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post(
        "/{dataset_id}/task-recommendation",
        response_model=TaskRecommendationResponse,
    )
    def recommend_task(
        dataset_id: str,
        body: TaskRecommendationRequest,
    ) -> TaskRecommendationResponse:
        try:
            return service.recommend_task(
                dataset_id,
                body.target_column,
                body.asset_id_column,
                body.timestamp_column,
            )
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except StorageError:
            raise HTTPException(
                status_code=500, detail="dataset storage is unavailable"
            ) from None
        except DataIngestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LeakageError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return router

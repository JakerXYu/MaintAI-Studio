"""Dataset application service (use-case orchestration).

Coordinates deterministic ingestion, profiling, task recommendation, leakage
detection, and quality assessment. The ORM ``file_path`` column is a *controlled
storage key* (``{dataset_id}.{extension}``) — never a user-supplied filename and
never an absolute path. Files are written atomically inside ``storage_root`` and
audit events record allowlisted metadata only.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from maintai.audit.service import AuditService
from maintai.data.ingest import (
    BYTES_PER_MB,
    DEFAULT_MAX_COLUMNS,
    DEFAULT_MAX_MEMORY_MB,
    DEFAULT_MAX_ROWS,
    DEFAULT_MAX_UPLOAD_MB,
    ingest_bytes,
)
from maintai.data.leakage import detect as detect_leakage
from maintai.data.profile import profile as build_profile
from maintai.data.quality import QualityConfig
from maintai.data.quality import assess as assess_quality
from maintai.data.schema import infer_schema
from maintai.db.dataset_repository import DatasetRepository
from maintai.db.models import (
    DATASET_STATUS_PROFILED,
    DATASET_STATUS_TASK_RECOMMENDED,
    DATASET_STATUS_UPLOADED,
    Dataset,
    new_id,
)
from maintai.tasks.infer import recommend_task as recommend_task_engine


class DatasetServiceError(Exception):
    """Base class for dataset application-service errors."""


class DuplicateDatasetError(DatasetServiceError):
    """Raised when an upload's content hash already exists."""

    def __init__(self, existing_dataset_id: str) -> None:
        super().__init__("a dataset with identical content already exists")
        self.existing_dataset_id = existing_dataset_id


class DatasetNotFoundError(DatasetServiceError):
    """Raised when a dataset id does not exist."""


class StorageError(DatasetServiceError):
    """Raised when stored bytes cannot be read or written safely.

    Messages never expose the resolved absolute storage path.
    """


class DatasetService:
    """Application service for dataset lifecycle operations."""

    def __init__(
        self,
        *,
        repository: DatasetRepository,
        session_factory: sessionmaker[Session],
        audit: AuditService,
        storage_root: str | Path,
        max_upload_bytes: int | None = None,
        max_rows: int | None = None,
        max_columns: int | None = None,
        max_memory_bytes: int | None = None,
        quality_config: QualityConfig | None = None,
    ) -> None:
        self._repository = repository
        self._session_factory = session_factory
        self._audit = audit
        self._storage_root = Path(storage_root).resolve()
        self._max_upload_bytes = (
            max_upload_bytes
            if max_upload_bytes is not None
            else DEFAULT_MAX_UPLOAD_MB * BYTES_PER_MB
        )
        self._max_rows = max_rows if max_rows is not None else DEFAULT_MAX_ROWS
        self._max_columns = max_columns if max_columns is not None else DEFAULT_MAX_COLUMNS
        self._max_memory_bytes = (
            max_memory_bytes
            if max_memory_bytes is not None
            else DEFAULT_MAX_MEMORY_MB * BYTES_PER_MB
        )
        self._quality_config = quality_config or QualityConfig()

    @property
    def max_upload_bytes(self) -> int:
        return self._max_upload_bytes

    # -- storage helpers ---------------------------------------------------

    def _ensure_storage_root(self) -> Path:
        try:
            self._storage_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError("storage root is not writable") from exc
        return self._storage_root

    def _resolve_storage_key(self, storage_key: str | None) -> Path:
        """Resolve a controlled storage key to a path inside ``storage_root``.

        The key must be a plain filename; any separator, traversal, or null byte
        is rejected so the resolved path can never escape the storage root.
        """
        if not storage_key:
            raise StorageError("dataset has no storage key")
        if (
            storage_key in {".", ".."}
            or "/" in storage_key
            or "\\" in storage_key
            or "\x00" in storage_key
        ):
            raise StorageError("invalid storage key")
        root = self._ensure_storage_root()
        target = (root / storage_key).resolve()
        if target.parent != root:
            raise StorageError("storage key escapes the storage root")
        return target

    def _atomic_write(self, storage_key: str, data: bytes) -> Path:
        """Write bytes to a temp file in-place, then atomically ``os.replace``."""
        target = self._resolve_storage_key(storage_key)
        tmp_path: Path | None = None
        try:
            fd, tmp_name = tempfile.mkstemp(
                dir=str(target.parent), prefix=".tmp-", suffix=target.suffix
            )
            tmp_path = Path(tmp_name)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp_path, target)
            return target
        except OSError as exc:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            raise StorageError("failed to persist dataset bytes") from exc

    def _remove_storage(self, storage_key: str | None) -> None:
        try:
            target = self._resolve_storage_key(storage_key)
        except StorageError:
            return
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass

    def _read_storage(self, storage_key: str | None) -> bytes:
        target = self._resolve_storage_key(storage_key)
        try:
            return target.read_bytes()
        except FileNotFoundError as exc:
            raise StorageError("dataset file is missing from storage") from exc
        except OSError as exc:
            raise StorageError("failed to read dataset bytes") from exc

    # -- internal helpers --------------------------------------------------

    def _get_or_raise(self, dataset_id: str) -> Dataset:
        dataset = self._repository.get(dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(f"dataset {dataset_id!r} not found")
        return dataset

    def _dataset_to_dict(self, dataset: Dataset, *, include_details: bool) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": dataset.id,
            "name": dataset.name,
            "original_filename": dataset.original_filename,
            "source_type": dataset.source_type,
            "file_path": dataset.file_path,
            "file_hash": dataset.file_hash,
            "size_bytes": dataset.size_bytes,
            "status": dataset.status,
            "row_count": dataset.row_count,
            "column_count": dataset.column_count,
            "quality_score": dataset.quality_score,
            "target_column": dataset.target_column,
            "asset_id_column": dataset.asset_id_column,
            "timestamp_column": dataset.timestamp_column,
            "task_type": dataset.task_type,
            "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
            "updated_at": dataset.updated_at.isoformat() if dataset.updated_at else None,
        }
        if include_details:
            result["schema"] = dataset.schema_json
            result["profile"] = dataset.profile_json
        return result

    @staticmethod
    def _upload_audit_payload(dataset: Dataset) -> dict[str, Any]:
        """Allowlisted metadata only — no paths, blobs, or file contents."""
        return {
            "dataset_id": dataset.id,
            "filename": dataset.original_filename,
            "source_type": dataset.source_type,
            "size_bytes": dataset.size_bytes,
            "row_count": dataset.row_count,
            "column_count": dataset.column_count,
        }

    # -- public API --------------------------------------------------------

    def upload(self, data: bytes, filename: str) -> dict[str, Any]:
        """Validate, dedupe, store, and register a new dataset."""
        ingested = ingest_bytes(
            data,
            filename,
            max_bytes=self._max_upload_bytes,
            max_rows=self._max_rows,
            max_columns=self._max_columns,
            max_memory_bytes=self._max_memory_bytes,
        )

        existing = self._repository.find_by_hash(ingested.sha256)
        if existing is not None:
            raise DuplicateDatasetError(existing.id)

        schema = infer_schema(ingested.frame)
        dataset_id = new_id()
        storage_key = f"{dataset_id}.{ingested.source_type}"
        self._atomic_write(storage_key, data)

        dataset = Dataset(
            id=dataset_id,
            name=ingested.name,
            original_filename=ingested.name,
            source_type=ingested.source_type,
            file_path=storage_key,
            file_hash=ingested.sha256,
            size_bytes=ingested.size_bytes,
            status=DATASET_STATUS_UPLOADED,
            row_count=int(ingested.frame.shape[0]),
            column_count=int(ingested.frame.shape[1]),
            schema_json=schema.model_dump(),
        )
        try:
            with self._session_factory.begin() as session:
                dataset = self._repository.create(dataset, session=session)
                self._audit.record(
                    actor_type="system",
                    action="dataset.upload",
                    entity_type="dataset",
                    entity_id=dataset_id,
                    payload=self._upload_audit_payload(dataset),
                    session=session,
                )
        except IntegrityError as exc:
            self._remove_storage(storage_key)
            concurrent = self._repository.find_by_hash(ingested.sha256)
            if concurrent is not None:
                raise DuplicateDatasetError(concurrent.id) from exc
            raise
        except Exception:
            self._remove_storage(storage_key)
            raise
        return self._dataset_to_dict(dataset, include_details=True)

    def profile(self, dataset_id: str) -> dict[str, Any]:
        """Re-read stored bytes and compute schema, profile, and quality."""
        dataset = self._get_or_raise(dataset_id)
        storage_key = dataset.file_path
        data = self._read_storage(storage_key)
        assert storage_key is not None  # _read_storage raises StorageError for None
        ingested = ingest_bytes(
            data,
            storage_key,
            max_bytes=self._max_upload_bytes,
            max_rows=self._max_rows,
            max_columns=self._max_columns,
            max_memory_bytes=self._max_memory_bytes,
        )
        frame = ingested.frame

        schema = infer_schema(frame)
        report = build_profile(frame, schema)
        quality = assess_quality(frame, schema, config=self._quality_config)

        dataset.schema_json = schema.model_dump()
        profile_json = dict(dataset.profile_json or {})
        profile_json.update(
            {
                "profile": report.model_dump(),
                "quality": quality.model_dump(),
            }
        )
        dataset.profile_json = profile_json
        dataset.quality_score = quality.score
        dataset.row_count = int(frame.shape[0])
        dataset.column_count = int(frame.shape[1])
        dataset.status = DATASET_STATUS_PROFILED
        with self._session_factory.begin() as session:
            dataset = self._repository.update(dataset, session=session)
            self._audit.record(
                actor_type="system",
                action="dataset.profile",
                entity_type="dataset",
                entity_id=dataset.id,
                payload={
                    "dataset_id": dataset.id,
                    "row_count": dataset.row_count,
                    "column_count": dataset.column_count,
                    "quality_score": dataset.quality_score,
                    "status": dataset.status,
                },
                session=session,
            )
        return self._dataset_to_dict(dataset, include_details=True)

    def recommend_task(
        self,
        dataset_id: str,
        target_column: str,
        asset_id_column: str | None = None,
        timestamp_column: str | None = None,
    ) -> dict[str, Any]:
        """Recommend a task, detect leakage, and assess quality for a dataset."""
        dataset = self._get_or_raise(dataset_id)
        storage_key = dataset.file_path
        data = self._read_storage(storage_key)
        assert storage_key is not None  # _read_storage raises StorageError for None
        frame = ingest_bytes(
            data,
            storage_key,
            max_bytes=self._max_upload_bytes,
            max_rows=self._max_rows,
            max_columns=self._max_columns,
            max_memory_bytes=self._max_memory_bytes,
        ).frame

        schema = infer_schema(frame)
        recommendation = recommend_task_engine(frame, target_column)
        leakage = detect_leakage(
            frame,
            target_column,
            timestamp_column,
            asset_id_column,
            schema,
        )
        quality = assess_quality(
            frame,
            schema,
            timestamp_col=timestamp_column,
            asset_id_col=asset_id_column,
            target_col=target_column,
            config=self._quality_config,
        )

        dataset.target_column = target_column
        dataset.asset_id_column = asset_id_column
        dataset.timestamp_column = timestamp_column
        dataset.task_type = recommendation.recommended_task
        dataset.quality_score = quality.score

        profile_json = dict(dataset.profile_json or {})
        profile_json["task_recommendation"] = recommendation.model_dump()
        profile_json["leakage_report"] = leakage.model_dump()
        profile_json["quality"] = quality.model_dump()
        dataset.profile_json = profile_json
        dataset.status = DATASET_STATUS_TASK_RECOMMENDED
        with self._session_factory.begin() as session:
            dataset = self._repository.update(dataset, session=session)
            self._audit.record(
                actor_type="system",
                action="dataset.task_recommended",
                entity_type="dataset",
                entity_id=dataset.id,
                payload={
                    "dataset_id": dataset.id,
                    "target_column": target_column,
                    "task_type": recommendation.recommended_task,
                    "leakage_verdict": leakage.verdict,
                    "excluded_features": leakage.excluded_features,
                },
                session=session,
            )
        return {
            "dataset_id": dataset.id,
            "target_column": target_column,
            "task": recommendation.model_dump(),
            "leakage": leakage.model_dump(),
            "quality_score": quality.score,
        }

    def get_quality(self, dataset_id: str) -> dict[str, Any]:
        """Return the stored quality score and report for a dataset."""
        dataset = self._get_or_raise(dataset_id)
        profile_json = dataset.profile_json
        report = profile_json.get("quality") if isinstance(profile_json, dict) else None
        return {
            "dataset_id": dataset.id,
            "score": dataset.quality_score,
            "report": report,
        }

    def load_frame(self, dataset_id: str) -> pd.DataFrame:
        """Re-read stored bytes, re-apply ingest limits, and return the frame.

        This is the only sanctioned way for other application services to obtain
        a dataset's rows. It reuses the controlled storage read (no arbitrary
        paths) and the configured ingest limits, and never exposes the resolved
        storage path to the caller.
        """
        dataset = self._get_or_raise(dataset_id)
        storage_key = dataset.file_path
        data = self._read_storage(storage_key)
        assert storage_key is not None  # _read_storage raises StorageError for None
        return ingest_bytes(
            data,
            storage_key,
            max_bytes=self._max_upload_bytes,
            max_rows=self._max_rows,
            max_columns=self._max_columns,
            max_memory_bytes=self._max_memory_bytes,
        ).frame

    def get(self, dataset_id: str) -> dict[str, Any]:
        """Return full dataset metadata including schema and profile."""
        return self._dataset_to_dict(self._get_or_raise(dataset_id), include_details=True)

    def list(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """Return dataset metadata summaries (no JSON blobs)."""
        datasets = self._repository.list(limit=limit, offset=offset)
        return [self._dataset_to_dict(d, include_details=False) for d in datasets]

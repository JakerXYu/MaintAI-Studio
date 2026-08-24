"""ORM models for MaintAI Studio (Phase A).

All column types are portable across SQLite (tests) and Postgres (runtime).
Primary keys are generated UUID hex strings; timestamps are timezone-aware,
set via Python-side defaults so both databases behave identically.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, CheckConstraint, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from maintai.db.base import Base


def utcnow() -> datetime:
    """Current UTC time with timezone information."""
    return datetime.now(UTC)


def new_id() -> str:
    """Generate a compact UUID hex primary key."""
    return uuid.uuid4().hex


# Dataset lifecycle states (set by the application service layer).
DATASET_STATUS_UPLOADED = "uploaded"
DATASET_STATUS_PROFILED = "profiled"
DATASET_STATUS_TASK_RECOMMENDED = "task_recommended"

# Experiment lifecycle states (set by the experiment application service).
EXPERIMENT_STATUS_QUEUED = "queued"
EXPERIMENT_STATUS_RUNNING = "running"
EXPERIMENT_STATUS_SUCCEEDED = "succeeded"
EXPERIMENT_STATUS_FAILED = "failed"

# ModelRun outcome states (set by the experiment application service).
MODEL_RUN_STATUS_SUCCESS = "success"
MODEL_RUN_STATUS_FAILED = "failed"

# RegisteredModel deployment states (set by the registry application service).
DEPLOYMENT_STATUS_CANDIDATE = "candidate"
DEPLOYMENT_STATUS_DEMO_DEPLOYED = "demo_deployed"


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, default="csv")
    file_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    file_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=DATASET_STATUS_UPLOADED
    )
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    column_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    schema_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    profile_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_column: Mapped[str | None] = mapped_column(String(255), nullable=True)
    asset_id_column: Mapped[str | None] = mapped_column(String(255), nullable=True)
    timestamp_column: Mapped[str | None] = mapped_column(String(255), nullable=True)
    task_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    dataset_id: Mapped[str | None] = mapped_column(
        ForeignKey("datasets.id", ondelete="RESTRICT"), index=True, nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    task_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=EXPERIMENT_STATUS_QUEUED
    )
    # Immutable training-plan snapshot plus run context (plan, minimum_recall,
    # scale_numeric, n_jobs) serialized as portable JSON. The application layer
    # also appends the deterministic result/comparison/explanation here after a
    # run completes so every claim stays evidence-backed in one JSON column.
    training_plan_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    recommended_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    recommended_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    config_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    primary_metric: Mapped[str | None] = mapped_column(String(64), nullable=True)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    experiment_id: Mapped[str | None] = mapped_column(
        ForeignKey("experiments.id", ondelete="RESTRICT"), index=True, nullable=True
    )
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    config_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    git_commit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="created")
    primary_metric: Mapped[str | None] = mapped_column(String(64), nullable=True)
    primary_metric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    confusion_matrix_json: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    artifact_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    model_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    feature_names_json: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    training_time_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class RegisteredModel(Base):
    __tablename__ = "registered_models"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    model_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_runs.id", ondelete="RESTRICT"), index=True, nullable=True
    )
    experiment_id: Mapped[str | None] = mapped_column(
        ForeignKey("experiments.id", ondelete="RESTRICT"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    # Internal-only controlled package artifact basename (never exposed in JSON;
    # the artifact lives in the app-private ``artifact_root`` directory).
    artifact_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # MLflow registry reference: always a ``models:/{name}/{version}`` URI.
    mlflow_model_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    alias: Mapped[str | None] = mapped_column(String(32), nullable=True)
    approval_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending"
    )
    deployment_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=DEPLOYMENT_STATUS_CANDIDATE
    )
    deployed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class PredictionEvent(Base):
    """Immutable record of a single prediction made by a deployed registered model."""

    __tablename__ = "prediction_events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    registered_model_id: Mapped[str | None] = mapped_column(
        ForeignKey("registered_models.id", ondelete="RESTRICT"), index=True, nullable=True
    )
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    asset_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    event_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    prediction_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    input_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )


class AuditEvent(Base):
    __tablename__ = "audit_events"
    __table_args__ = (
        CheckConstraint(
            "actor_type IN ('user', 'agent', 'system')",
            name="ck_audit_events_actor_type",
        ),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    actor_type: Mapped[str] = mapped_column(String(32), nullable=False)  # user|agent|system
    actor_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    entity_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    entity_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )

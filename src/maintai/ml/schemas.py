"""Pydantic contracts for the deterministic ML training/evaluation core.

These models are pure data contracts: they depend only on Pydantic (v2) and on
``maintai.data.schemas.SplitResult``. They are JSON-safe and serialise cleanly
for agent tool results and for persisting summaries without ML internals.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from maintai.data.schemas import SplitResult

TaskType = Literal["binary_classification", "multiclass_classification", "regression"]

CLASSIFICATION_TASKS: tuple[str, ...] = (
    "binary_classification",
    "multiclass_classification",
)

_KNOWN_METRICS: frozenset[str] = frozenset(
    {"precision", "recall", "f1", "roc_auc", "pr_auc", "mae", "rmse", "r2"}
)


class MLSettings(BaseModel):
    """Deterministic ML defaults (mirrors ``configs/default.yaml`` ``ml:``)."""

    seed: int = 42
    n_jobs: int = 1
    scale_numeric: bool = True
    minimum_recall: float = 0.80

    @field_validator("n_jobs")
    @classmethod
    def _n_jobs_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("n_jobs must be >= 1")
        return value

    @field_validator("minimum_recall")
    @classmethod
    def _recall_in_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("minimum_recall must be in [0, 1]")
        return value


class TrainingPlan(BaseModel):
    """Deterministic training request: what to train, on what, and how to split."""

    target: str
    task: TaskType
    features: list[str] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list)
    split: SplitResult
    model_names: list[str] = Field(default_factory=list)
    primary_metric: str | None = None
    seed: int = 42

    @field_validator("seed")
    @classmethod
    def _seed_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("seed must be >= 0")
        return value

    @field_validator("primary_metric")
    @classmethod
    def _known_metric(cls, value: str | None) -> str | None:
        if value is not None and value not in _KNOWN_METRICS:
            raise ValueError(f"unknown primary metric {value!r}")
        return value

    @model_validator(mode="after")
    def _metric_matches_task(self) -> TrainingPlan:
        if self.primary_metric is None:
            return self
        classification_metrics = {"precision", "recall", "f1", "roc_auc", "pr_auc"}
        regression_metrics = {"mae", "rmse", "r2"}
        allowed = regression_metrics if self.task == "regression" else classification_metrics
        if self.primary_metric not in allowed:
            raise ValueError(
                f"metric {self.primary_metric!r} is not valid for task {self.task!r}"
            )
        return self


class ModelEvaluation(BaseModel):
    """JSON-safe summary of a single model's training/evaluation."""

    model_name: str
    task: TaskType
    status: Literal["success", "error"] = "success"
    metrics: dict[str, float | None] = Field(default_factory=dict)
    confusion_matrix: list[list[int]] | None = None
    labels: list[str | int | float | bool | None] = Field(default_factory=list)
    positive_label: str | int | float | bool | None = None
    inference_latency_ms: float | None = None
    training_time_seconds: float | None = None
    error: str | None = None
    notes: list[str] = Field(default_factory=list)


class TrainingResult(BaseModel):
    """JSON-safe summary of a full training run (pipelines live separately)."""

    task: TaskType
    target: str
    seed: int
    primary_metric: str | None = None
    evaluations: list[ModelEvaluation] = Field(default_factory=list)
    best_model: str | None = None
    errors: list[str] = Field(default_factory=list)


class Comparison(BaseModel):
    """Deterministic best-model recommendation with explainable notes."""

    task: TaskType
    primary_metric: str
    best_model: str | None = None
    ranking: list[str] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

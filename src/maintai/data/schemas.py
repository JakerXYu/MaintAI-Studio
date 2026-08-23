"""Pydantic contracts for the deterministic data-intelligence layer.

These models are pure data contracts: they depend only on Pydantic (v2) and
never on FastAPI, SQLAlchemy, or MLflow. They are safe to serialize to JSON for
``datasets.schema_json`` / ``datasets.profile_json`` and for agent tool results.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ColumnSchema(BaseModel):
    """Inferred schema for a single column."""

    name: str
    dtype: str
    semantic_type: str
    missing_rate: float
    nunique: int
    unique_ratio: float


class SchemaInference(BaseModel):
    """Deterministic physical/semantic schema inference result."""

    columns: list[ColumnSchema]
    candidate_targets: list[str] = Field(default_factory=list)
    candidate_asset_ids: list[str] = Field(default_factory=list)
    candidate_timestamps: list[str] = Field(default_factory=list)


class NumericStats(BaseModel):
    """Basic numeric summary statistics."""

    count: int
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    q1: float | None = None
    q3: float | None = None
    iqr: float | None = None


class CategoryFrequency(BaseModel):
    """A single top category with its frequency."""

    value: str
    count: int
    ratio: float


class ColumnProfile(BaseModel):
    """Per-column profile."""

    name: str
    dtype: str
    semantic_type: str | None = None
    missing_count: int
    missing_rate: float
    unique_count: int
    cardinality: int
    constant: bool
    numeric: NumericStats | None = None
    top_categories: list[CategoryFrequency] = Field(default_factory=list)


class ProfileReport(BaseModel):
    """Whole-dataset profile."""

    row_count: int
    column_count: int
    cell_count: int
    missing_cells: int
    missing_rate: float
    duplicate_rows: int
    duplicate_row_rate: float
    type_distribution: dict[str, int] = Field(default_factory=dict)
    columns: list[ColumnProfile] = Field(default_factory=list)


class QualityFinding(BaseModel):
    """A single explainable data-quality finding."""

    check: str
    severity: Literal["info", "warning", "error"]
    column: str | None = None
    message: str
    value: float | int | str | None = None
    penalty: float = 0.0


class Penalty(BaseModel):
    """A point deduction contributing to the health score."""

    check: str
    column: str | None = None
    points: float
    reason: str


class QualityReport(BaseModel):
    """Explainable data-quality report."""

    label: str = "MaintAI heuristic data health score"
    max_score: float = 100.0
    score: float
    penalties: list[Penalty] = Field(default_factory=list)
    findings: list[QualityFinding] = Field(default_factory=list)
    checks: dict[str, Any] = Field(default_factory=dict)
    trainable_sample_count: int = 0


class LeakageIssue(BaseModel):
    """A single target-leakage finding."""

    feature: str
    kind: str
    severity: Literal["safe", "warning", "block"]
    reason: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class LeakageReport(BaseModel):
    """Target-leakage detection result."""

    verdict: Literal["safe", "warning", "block"]
    issues: list[LeakageIssue] = Field(default_factory=list)
    excluded_features: list[str] = Field(default_factory=list)


class SplitResult(BaseModel):
    """Deterministic data-split result with evidence."""

    strategy: str
    train_indices: list[int] = Field(default_factory=list)
    test_indices: list[int] = Field(default_factory=list)
    validation_indices: list[int] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)

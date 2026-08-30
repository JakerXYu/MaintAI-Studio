"""Pydantic v2 JSON-safe contracts for the P1 deterministic monitoring core.

These models are pure data contracts: they depend only on Pydantic and never on
pandas, scikit-learn, FastAPI, or SQLAlchemy. They serialise cleanly to JSON for
audit events and agent tool results, and carry no ML internals.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Severity = Literal["LOW", "MEDIUM", "HIGH"]

SEVERITY_ORDER: dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

ReplayKind = Literal["normal", "mild", "severe", "increased_failure_risk"]

REPLAY_KINDS: tuple[str, ...] = ("normal", "mild", "severe", "increased_failure_risk")


class DeviatingFeature(BaseModel):
    """A single feature contributing most to a row's anomaly score."""

    feature: str
    value: float | None = None
    robust_z: float | None = None


class AnomalyRow(BaseModel):
    """Per-row anomaly result: scores, flag, and top deviating features."""

    row_index: int
    robust_score: float
    iforest_score: float
    combined_score: float
    flag: bool
    top_deviating_features: list[DeviatingFeature] = Field(default_factory=list)


class AnomalyResult(BaseModel):
    """Deterministic anomaly-detection report for a production window."""

    rows: list[AnomalyRow] = Field(default_factory=list)
    threshold: float = 3.0
    flagged_count: int = 0
    anomaly_rate: float = 0.0
    contamination: float | str = "auto"
    numeric_features: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class FeatureDrift(BaseModel):
    """Per-feature drift evidence (numeric and/or categorical)."""

    feature: str
    kind: Literal["numeric", "categorical"]
    severity: Severity = "LOW"
    psi: float | None = None
    ks_stat: float | None = None
    ks_pvalue: float | None = None
    mean_shift: float | None = None
    std_shift: float | None = None
    total_variation: float | None = None
    missingness_shift: float | None = None
    new_categories: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class DriftReport(BaseModel):
    """Aggregate drift report with heuristic severity (NOT an industry standard)."""

    features: list[FeatureDrift] = Field(default_factory=list)
    overall_severity: Severity = "LOW"
    thresholds: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class RetrainingEvidence(BaseModel):
    """Quantified evidence behind a retraining trigger."""

    trigger: str
    value: float | int | str | bool | None = None
    threshold: float | int | str | None = None
    detail: str


class RetrainingRecommendation(BaseModel):
    """Propose retraining only; never train or deploy automatically."""

    recommended: bool = False
    auto_deploy: bool = False
    triggers: list[str] = Field(default_factory=list)
    evidence: list[RetrainingEvidence] = Field(default_factory=list)
    proposed_action: str | None = None

    @field_validator("auto_deploy")
    @classmethod
    def _never_auto_deploy(cls, value: bool) -> bool:
        # Auto-deploy is permanently disabled in P1; human approval is mandatory.
        return False


class ReplayBatch(BaseModel):
    """Metadata for one synthetic production-replay batch (NOT real production)."""

    kind: ReplayKind
    row_count: int
    seed: int
    transformations: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)

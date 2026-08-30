"""Retraining recommendation (rule engine; proposes only, never trains/deploys).

A retraining recommendation is raised when ANY trigger fires: HIGH drift,
primary-metric performance drop, anomaly-rate jump, schedule due, or a manual
request. The result only proposes retraining — ``auto_deploy`` is always False
and no training or deployment is performed here.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from maintai.monitoring.contracts import (
    DriftReport,
    RetrainingEvidence,
    RetrainingRecommendation,
)

_NO_RETRAIN_ACTION = "No retraining needed: no trigger fired. Continue monitoring."
_PROPOSE_ACTION = (
    "Propose retraining with human approval. No automatic training or deployment."
)


class RecommendConfig(BaseModel):
    """Demo-default recommendation thresholds (mirrors ``configs/default.yaml``)."""

    performance_drop_threshold: float = 0.05
    anomaly_rate_jump_threshold: float = 0.10
    schedule_days_threshold: int = 30

    @field_validator("performance_drop_threshold", "anomaly_rate_jump_threshold")
    @classmethod
    def _non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("threshold must be >= 0")
        return value

    @field_validator("schedule_days_threshold")
    @classmethod
    def _positive_days(cls, value: int) -> int:
        if value < 1:
            raise ValueError("schedule_days_threshold must be >= 1")
        return value


def recommend(
    drift_report: DriftReport,
    *,
    performance_drop: float | None = None,
    anomaly_rate: float | None = None,
    baseline_anomaly_rate: float = 0.0,
    schedule_due: bool = False,
    days_since_last_retrain: int | None = None,
    manual: bool = False,
    config: RecommendConfig | None = None,
) -> RetrainingRecommendation:
    """Evaluate every trigger and return a proposal (never train/deploy)."""
    config = config or RecommendConfig()
    triggers: list[str] = []
    evidence: list[RetrainingEvidence] = []

    # 1. HIGH drift.
    if drift_report.overall_severity == "HIGH":
        high_features = [f.feature for f in drift_report.features if f.severity == "HIGH"]
        triggers.append("high_drift")
        evidence.append(
            RetrainingEvidence(
                trigger="high_drift",
                value=drift_report.overall_severity,
                threshold="HIGH",
                detail=f"overall drift severity HIGH (features: {high_features})",
            )
        )

    # 2. Primary-metric performance drop.
    if performance_drop is not None and performance_drop >= config.performance_drop_threshold:
        triggers.append("performance_drop")
        evidence.append(
            RetrainingEvidence(
                trigger="performance_drop",
                value=performance_drop,
                threshold=config.performance_drop_threshold,
                detail="primary metric dropped below the accepted threshold",
            )
        )

    # 3. Anomaly-rate jump.
    if anomaly_rate is not None:
        jump = anomaly_rate - baseline_anomaly_rate
        if jump >= config.anomaly_rate_jump_threshold:
            triggers.append("anomaly_rate_jump")
            evidence.append(
                RetrainingEvidence(
                    trigger="anomaly_rate_jump",
                    value=jump,
                    threshold=config.anomaly_rate_jump_threshold,
                    detail=f"anomaly rate rose from {baseline_anomaly_rate} to {anomaly_rate}",
                )
            )

    # 4. Schedule due.
    schedule_triggered = schedule_due or (
        days_since_last_retrain is not None
        and days_since_last_retrain >= config.schedule_days_threshold
    )
    if schedule_triggered:
        triggers.append("schedule")
        evidence.append(
            RetrainingEvidence(
                trigger="schedule",
                value=days_since_last_retrain,
                threshold=config.schedule_days_threshold,
                detail="retraining schedule is due",
            )
        )

    # 5. Manual request.
    if manual:
        triggers.append("manual")
        evidence.append(
            RetrainingEvidence(
                trigger="manual",
                value=True,
                threshold=None,
                detail="manually requested retraining review",
            )
        )

    recommended = bool(triggers)
    return RetrainingRecommendation(
        recommended=recommended,
        auto_deploy=False,
        triggers=triggers,
        evidence=evidence,
        proposed_action=_PROPOSE_ACTION if recommended else _NO_RETRAIN_ACTION,
    )

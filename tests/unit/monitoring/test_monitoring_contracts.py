"""Contract tests for the monitoring schemas (JSON-safe Pydantic v2)."""

import json

import pytest
from pydantic import ValidationError

from maintai.monitoring.contracts import (
    AnomalyResult,
    AnomalyRow,
    DeviatingFeature,
    DriftReport,
    FeatureDrift,
    ReplayBatch,
    RetrainingRecommendation,
)


def test_anomaly_contract_json_roundtrip():
    obj = AnomalyResult(
        rows=[
            AnomalyRow(
                row_index=0,
                robust_score=1.0,
                iforest_score=0.5,
                combined_score=1.2,
                flag=True,
                top_deviating_features=[DeviatingFeature(feature="a", value=3.0, robust_z=2.0)],
            )
        ],
        threshold=3.0,
        flagged_count=1,
    )
    data = obj.model_dump()
    assert AnomalyResult.model_validate(data) == obj
    json.loads(obj.model_dump_json())


def test_drift_contract_json_roundtrip():
    obj = DriftReport(
        features=[FeatureDrift(feature="a", kind="numeric", severity="HIGH", psi=0.3)],
        overall_severity="HIGH",
    )
    json.loads(obj.model_dump_json())
    assert DriftReport.model_validate(obj.model_dump()) == obj


def test_retraining_recommendation_auto_deploy_forced_false():
    rec = RetrainingRecommendation(recommended=True, auto_deploy=True, triggers=["manual"])
    assert rec.auto_deploy is False


def test_replay_batch_metadata_json():
    batch = ReplayBatch(kind="normal", row_count=10, seed=42, transformations=["x"])
    json.loads(batch.model_dump_json())
    assert ReplayBatch.model_validate(batch.model_dump()) == batch


def test_severity_literal_rejects_invalid():
    with pytest.raises(ValidationError):
        FeatureDrift(feature="a", kind="numeric", severity="HUGE")

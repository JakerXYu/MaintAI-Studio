"""Unit tests for the read-only tool allowlist."""

from __future__ import annotations

import pytest
from fakes import (
    FakeDatasetService,
    FakeExperimentService,
    FakeModelRegistryService,
    FakePredictionService,
)

from maintai.agent.tools import CopilotTools

# The full public allowlist surface — exactly these seven read methods.
_ALLOWLISTED_METHODS = {
    "dataset_profile",
    "dataset_quality",
    "dataset_task",
    "experiment_results",
    "experiment_comparison",
    "model_deployment_status",
    "prediction_explain",
}


def test_tool_allowlist_surface(fakes):
    public = {name for name in dir(fakes.tools) if not name.startswith("_")}
    assert _ALLOWLISTED_METHODS.issubset(public)
    unexpected = [
        name for name in public if name not in _ALLOWLISTED_METHODS
        and callable(getattr(fakes.tools, name))
    ]
    assert not unexpected


def test_dataset_profile_strips_internal_fields(fakes):
    result = fakes.tools.dataset_profile("d1")
    assert result.ok is True
    assert result.evidence["source"] == "dataset.profile"
    assert "file_path" not in result.data
    assert "file_hash" not in result.data
    assert result.data["quality_score"] == 82.5


def test_dataset_quality_returns_evidence(fakes):
    result = fakes.tools.dataset_quality("d1")
    assert result.ok is True
    assert result.data["score"] == 82.5
    assert result.evidence == {"source": "dataset.quality", "dataset_id": "d1"}
    assert fakes.dataset.get_quality_calls == ["d1"]


def test_dataset_task_reads_recommendation(fakes):
    result = fakes.tools.dataset_task("d1")
    assert result.ok is True
    assert result.data["task_type"] == "binary_classification"
    assert result.evidence["source"] == "dataset.task"


def test_experiment_comparison_and_results(fakes):
    comparison = fakes.tools.experiment_comparison("e1")
    assert comparison.ok is True
    assert comparison.data["best_model"] == "xgboost"

    results = fakes.tools.experiment_results("e1")
    assert results.ok is True
    assert results.data["primary_metric"] == "recall"
    assert results.data["model_runs"][0]["metrics"]["recall"] == 0.91


def test_model_deployment_status(fakes):
    result = fakes.tools.model_deployment_status("m1")
    assert result.ok is True
    assert result.data["deployment_status"] == "demo_deployed"
    assert result.evidence["source"] == "model.deployment_status"


def test_prediction_explain_requires_record(fakes):
    result = fakes.tools.prediction_explain("m1", None)
    assert result.ok is False
    assert "record is required" in result.error
    assert fakes.prediction.predict_calls == []


def test_prediction_explain_returns_result(fakes):
    result = fakes.tools.prediction_explain("m1", {"sensor": 0.5})
    assert result.ok is True
    assert result.data["records"][0]["confidence"] == 0.93
    assert fakes.prediction.predict_calls == [("m1", [{"sensor": 0.5}])]
    assert fakes.prediction.persist_values == [False]


def test_prediction_not_deployed_is_rejected():
    prediction = FakePredictionService(deployed=False)
    tools = CopilotTools(
        dataset_service=FakeDatasetService(),
        experiment_service=FakeExperimentService(),
        model_registry_service=FakeModelRegistryService(),
        prediction_service=prediction,
    )
    result = tools.prediction_explain("m1", {"sensor": 0.5})
    assert result.ok is False
    assert "not demo-deployed" in result.error


def test_not_found_is_graceful(fakes):
    result = fakes.tools.dataset_quality("does-not-exist")
    assert result.ok is False
    assert "not found" in result.error


def test_tools_reject_arbitrary_arguments(fakes):
    with pytest.raises(TypeError):
        fakes.tools.dataset_quality(dataset_id="d1", evil=True)
    with pytest.raises(TypeError):
        fakes.tools.dataset_quality()  # missing required arg

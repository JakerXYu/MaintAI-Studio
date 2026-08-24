"""Integration tests for the P0 Experiment application service.

These exercise the full upload → profile → task → create → run loop against a
real in-memory SQLite database, a real MLflow local file backend, and a real
artifact directory (all under ``tmp_path``), with a small synthetic dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from maintai.application.datasets import DatasetService
from maintai.application.experiments import (
    DatasetNotReadyError,
    ExperimentNotFoundError,
    ExperimentService,
    ExperimentStateError,
)
from maintai.audit.repository import AuditRepository
from maintai.audit.service import AuditService
from maintai.db.dataset_repository import DatasetRepository
from maintai.db.experiment_repository import ExperimentRepository
from maintai.ml import package
from maintai.mlops import MLflowTracker, TrackingError

EXPERIMENT_NAME = "p0_experiment_test"
CLASSIFICATION_MODELS = {"logistic_regression", "random_forest", "xgboost"}


def _classification_csv(n: int = 240, n_pos: int = 120, seed: int = 0) -> bytes:
    """Small, balanced, synthetic binary-classification CSV with a leakage column."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    rng.shuffle(y)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
            "asset_id": [f"M{i % 12:02d}" for i in range(n)],
            "serial_no": [f"s{i:05d}" for i in range(n)],
            "sensor": np.round(rng.normal(0.0, 1.0, n) + y * 2.0, 6),
            "cat": rng.choice(["a", "b", "c"], n),
            "flag": rng.choice([True, False], n),
            "failure": y,
        }
    )
    return frame.to_csv(index=False).encode("utf-8")


def _prepared_dataset(dataset_service: DatasetService) -> tuple[str, dict]:
    """Upload → profile → task-recommend and return (dataset_id, uploaded)."""
    uploaded = dataset_service.upload(_classification_csv(), "sensor_data.csv")
    dataset_id = uploaded["id"]
    dataset_service.profile(dataset_id)
    dataset_service.recommend_task(
        dataset_id,
        target_column="failure",
        asset_id_column="asset_id",
        timestamp_column="timestamp",
    )
    return dataset_id, uploaded


# -- fixtures ----------------------------------------------------------------


@pytest.fixture()
def storage_root(tmp_path):
    return tmp_path / "storage"


@pytest.fixture()
def artifact_root(tmp_path):
    return tmp_path / "artifacts"


@pytest.fixture()
def tracker(tmp_path):
    return MLflowTracker((tmp_path / "mlruns").as_uri(), EXPERIMENT_NAME)


@pytest.fixture()
def audit_repository(session_factory):
    return AuditRepository(session_factory)


@pytest.fixture()
def dataset_service(session_factory, storage_root):
    return DatasetService(
        repository=DatasetRepository(session_factory),
        session_factory=session_factory,
        audit=AuditService(AuditRepository(session_factory)),
        storage_root=storage_root,
    )


@pytest.fixture()
def experiment_service(session_factory, dataset_service, tracker, artifact_root):
    return ExperimentService(
        session_factory=session_factory,
        dataset_service=dataset_service,
        experiment_repository=ExperimentRepository(session_factory),
        audit=AuditService(AuditRepository(session_factory)),
        tracker=tracker,
        artifact_root=artifact_root,
    )


def _experiment_events(audit_repository, experiment_id):
    return audit_repository.list(entity_type="experiment", entity_id=experiment_id)


# -- happy path --------------------------------------------------------------


def test_full_experiment_lifecycle(
    dataset_service,
    experiment_service,
    tracker,
    audit_repository,
    artifact_root,
):
    dataset_id, _ = _prepared_dataset(dataset_service)

    created = experiment_service.create(dataset_id)
    assert created["status"] == "queued"
    assert created["config_hash"]
    assert created["training_plan"]["plan"]["task"] == "binary_classification"
    assert set(created["training_plan"]["plan"]["excluded"]) == {
        "asset_id",
        "serial_no",
        "timestamp",
    }

    result = experiment_service.run(created["id"])
    assert result["status"] == "succeeded"
    assert result["recommended_model"] in CLASSIFICATION_MODELS
    assert result["recommended_run_id"]
    assert result["primary_metric"] == "f1"
    assert result["value"] is not None

    got = experiment_service.get(created["id"])
    model_runs = got["model_runs"]
    assert len(model_runs) == 3
    assert {run["model_name"] for run in model_runs} == CLASSIFICATION_MODELS
    for run in model_runs:
        assert run["status"] == "success"
        assert run["mlflow_run_id"]
        assert run["model_uri"]
        assert run["artifact_uri"]
        assert run["metrics"]["f1"] is not None
        assert run["confusion_matrix"] is not None
        assert run["feature_names"]
        assert run["training_time_seconds"] is not None

    # MLflow runs exist with the required params and artifacts.
    client = MlflowClient(tracker.tracking_uri)
    mlflow_experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs([mlflow_experiment.experiment_id], max_results=100)
    assert len(runs) == 3
    for run in runs:
        assert run.data.params["dataset_id"] == dataset_id
        assert run.data.params["config_hash"] == created["config_hash"]
        assert run.data.params["target"] == "failure"
        assert run.data.params["task"] == "binary_classification"
        paths = {info.path for info in client.list_artifacts(run.info.run_id, path="artifacts")}
        assert {
            "artifacts/training_plan.json",
            "artifacts/evaluation.json",
            "artifacts/schema.json",
            "artifacts/quality.json",
        } <= paths

    # The recommended comparison is JSON-safe and consistent.
    comp = experiment_service.comparison(created["id"])
    assert comp["best_model"] == result["recommended_model"]
    assert set(comp["ranking"]) == CLASSIFICATION_MODELS

    # The saved package manifest loads and predicts (no raw path returned).
    snapshot = got["training_plan"]
    package_name = snapshot["package_artifact"]
    assert "/" not in package_name and "\\" not in package_name
    pipeline, manifest = package.load_artifact(artifact_root, package_name)
    assert manifest.model_name == result["recommended_model"]
    assert manifest.task == "binary_classification"
    preds = pipeline.predict(
        pd.DataFrame(
            {
                "sensor": [0.0, 2.0, -1.0],
                "cat": ["a", "b", "c"],
                "flag": [True, False, True],
            }
        )
    )
    assert len(preds) == 3

    # The global explanation artifact is present and JSON-safe.
    explanation = snapshot["explanation"]
    assert explanation["method"] in {"shap_tree", "shap_linear", "permutation_importance"}
    assert explanation["top_features"]
    assert explanation["disclaimer"]

    # Audit trail records the lifecycle.
    actions = [event.action for event in _experiment_events(audit_repository, created["id"])]
    assert "experiment.create" in actions
    assert "experiment.run_started" in actions
    assert "experiment.run_succeeded" in actions

    # list() returns JSON-safe summaries without blobs.
    listed = experiment_service.list()
    assert [item["id"] for item in listed] == [created["id"]]
    assert "training_plan" not in listed[0]


def test_fixed_seed_repeated_recommendation(
    dataset_service, experiment_service
):
    dataset_id, _ = _prepared_dataset(dataset_service)

    first = experiment_service.run(experiment_service.create(dataset_id)["id"])
    second = experiment_service.run(experiment_service.create(dataset_id)["id"])

    assert first["status"] == "succeeded"
    assert second["status"] == "succeeded"
    assert first["recommended_model"] == second["recommended_model"]
    assert first["primary_metric"] == second["primary_metric"]
    assert first["value"] == pytest.approx(second["value"])


# -- failure paths -----------------------------------------------------------


def test_create_requires_task(dataset_service, experiment_service):
    uploaded = dataset_service.upload(_classification_csv(), "data.csv")
    dataset_service.profile(uploaded["id"])

    with pytest.raises(DatasetNotReadyError):
        experiment_service.create(uploaded["id"])


def test_run_missing_file_marks_failed(
    dataset_service,
    experiment_service,
    audit_repository,
    storage_root,
):
    dataset_id, uploaded = _prepared_dataset(dataset_service)
    created = experiment_service.create(dataset_id)

    (storage_root / uploaded["file_path"]).unlink()

    result = experiment_service.run(created["id"])
    assert result["status"] == "failed"
    assert "missing" in result["error_message"]
    assert str(storage_root) not in result["error_message"]

    actions = [event.action for event in _experiment_events(audit_repository, created["id"])]
    assert "experiment.run_started" in actions
    assert "experiment.run_failed" in actions


def test_run_mlflow_failure_marks_failed(
    dataset_service,
    experiment_service,
    audit_repository,
    monkeypatch,
):
    dataset_id, _ = _prepared_dataset(dataset_service)
    created = experiment_service.create(dataset_id)

    def boom(**kwargs):
        raise TrackingError("MLflow tracking failed: ConnectionError")

    monkeypatch.setattr(experiment_service._tracker, "log_model_run", boom)

    result = experiment_service.run(created["id"])
    assert result["status"] == "failed"
    assert "MLflow" in result["error_message"]

    actions = [event.action for event in _experiment_events(audit_repository, created["id"])]
    assert "experiment.run_started" in actions
    assert "experiment.run_failed" in actions


def test_partial_mlflow_failure_persists_reconciliation_evidence(
    dataset_service,
    experiment_service,
    monkeypatch,
):
    dataset_id, _ = _prepared_dataset(dataset_service)
    created = experiment_service.create(dataset_id)
    original = experiment_service._tracker.log_model_run
    calls = 0

    def fail_after_first(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TrackingError("MLflow tracking failed: ConnectionError")
        return original(**kwargs)

    monkeypatch.setattr(experiment_service._tracker, "log_model_run", fail_after_first)
    result = experiment_service.run(created["id"])
    assert result["status"] == "failed"

    stored = experiment_service.get(created["id"])
    assert len(stored["model_runs"]) == 1
    assert len(stored["training_plan"]["tracked_runs"]) == 1
    assert any(
        "not fully tracked" in note
        for note in stored["training_plan"]["reconciliation_notes"]
    )


# -- read guards -------------------------------------------------------------


def test_get_and_run_not_found(dataset_service, experiment_service):
    with pytest.raises(ExperimentNotFoundError):
        experiment_service.get("does-not-exist")
    with pytest.raises(ExperimentNotFoundError):
        experiment_service.run("does-not-exist")
    with pytest.raises(ExperimentNotFoundError):
        experiment_service.comparison("does-not-exist")


def test_run_requires_queued_state(dataset_service, experiment_service):
    dataset_id, _ = _prepared_dataset(dataset_service)
    created = experiment_service.create(dataset_id)
    experiment_service.run(created["id"])

    with pytest.raises(ExperimentStateError):
        experiment_service.run(created["id"])

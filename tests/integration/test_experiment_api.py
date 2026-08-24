"""Integration tests for the P0 Experiment REST API.

These exercise the full upload → profile → task → POST experiment → GET flow over
HTTP against an in-memory SQLite database, a real MLflow local file backend, and
real tmp artifact storage, with the dataset/experiment services injected into the
app factory. Because ``TestClient`` runs ``BackgroundTasks`` before returning, the
``POST`` response body still reflects the queued snapshot while a subsequent
``GET`` asserts the final persisted state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from maintai.api.main import create_app
from maintai.application.datasets import DatasetService
from maintai.application.experiments import ExperimentService
from maintai.audit.repository import AuditRepository
from maintai.audit.service import AuditService
from maintai.db.dataset_repository import DatasetRepository
from maintai.db.experiment_repository import ExperimentRepository
from maintai.mlops import MLflowTracker, TrackingError

EXPERIMENT_NAME = "p0_experiment_api_test"
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


def _prepared_dataset(dataset_service: DatasetService) -> str:
    """Upload → profile → task-recommend and return the dataset id."""
    uploaded = dataset_service.upload(_classification_csv(), "sensor_data.csv")
    dataset_id = uploaded["id"]
    dataset_service.profile(dataset_id)
    dataset_service.recommend_task(
        dataset_id,
        target_column="failure",
        asset_id_column="asset_id",
        timestamp_column="timestamp",
    )
    return dataset_id


def _upload_and_profile(dataset_service: DatasetService) -> str:
    """Upload → profile (no task recommendation) and return the dataset id."""
    uploaded = dataset_service.upload(_classification_csv(), "sensor_data.csv")
    dataset_id = uploaded["id"]
    dataset_service.profile(dataset_id)
    return dataset_id


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


@pytest.fixture()
def api_client(session_factory, dataset_service, experiment_service):
    app = create_app(
        session_factory=session_factory,
        dataset_service=dataset_service,
        experiment_service=experiment_service,
        mlflow_healthcheck=lambda: None,
    )
    with TestClient(app) as client:
        yield client


# -- happy path --------------------------------------------------------------


def test_experiment_api_full_flow(api_client, dataset_service):
    dataset_id = _prepared_dataset(dataset_service)

    created = api_client.post("/api/v1/experiments", json={"dataset_id": dataset_id})
    assert created.status_code == 202
    body = created.json()
    assert body["status"] == "queued"
    assert body["dataset_id"] == dataset_id
    assert body["config_hash"]
    assert body["training_plan"]["plan"]["task"] == "binary_classification"

    experiment_id = body["id"]

    # BackgroundTasks complete before TestClient returns, so the persisted state
    # is final by the time we GET.
    got = api_client.get(f"/api/v1/experiments/{experiment_id}")
    assert got.status_code == 200
    detail = got.json()
    assert detail["status"] == "succeeded"
    assert detail["recommended_model"] in CLASSIFICATION_MODELS
    assert detail["recommended_run_id"]
    assert detail["primary_metric"] == "f1"
    assert detail["value"] is not None

    model_runs = detail["model_runs"]
    assert len(model_runs) == 3
    assert {run["model_name"] for run in model_runs} == CLASSIFICATION_MODELS
    for run in model_runs:
        assert run["status"] == "success"
        assert run["model_uri"].startswith("runs:/")
        assert run["artifact_uri"].startswith("runs:/")
        assert run["metrics"]["f1"] is not None

    comparison = api_client.get(f"/api/v1/experiments/{experiment_id}/comparison")
    assert comparison.status_code == 200
    comp = comparison.json()
    assert comp["experiment_id"] == experiment_id
    assert comp["best_model"] == detail["recommended_model"]
    assert set(comp["ranking"]) == CLASSIFICATION_MODELS


def test_experiment_api_list_returns_summaries(api_client, dataset_service):
    dataset_id = _prepared_dataset(dataset_service)
    created = api_client.post("/api/v1/experiments", json={"dataset_id": dataset_id})
    experiment_id = created.json()["id"]

    listed = api_client.get("/api/v1/experiments")
    assert listed.status_code == 200
    summaries = listed.json()
    assert [item["id"] for item in summaries] == [experiment_id]
    assert "training_plan" not in summaries[0]
    assert "model_runs" not in summaries[0]


# -- validation / not-ready / not-found --------------------------------------


def test_experiment_api_rejects_missing_dataset_id(api_client):
    r = api_client.post("/api/v1/experiments", json={})
    assert r.status_code == 422


def test_experiment_api_rejects_arbitrary_params(api_client, dataset_service):
    dataset_id = _prepared_dataset(dataset_service)
    r = api_client.post(
        "/api/v1/experiments",
        json={"dataset_id": dataset_id, "shell": "rm -rf /", "run_id": "x"},
    )
    assert r.status_code == 422


def test_experiment_api_rejects_invalid_minimum_recall(api_client, dataset_service):
    dataset_id = _prepared_dataset(dataset_service)
    for value in (-0.1, 1.5):
        r = api_client.post(
            "/api/v1/experiments",
            json={"dataset_id": dataset_id, "minimum_recall": value},
        )
        assert r.status_code == 422


def test_experiment_api_dataset_not_found(api_client):
    r = api_client.post("/api/v1/experiments", json={"dataset_id": "does-not-exist"})
    assert r.status_code == 404


def test_experiment_api_dataset_not_ready(api_client, dataset_service):
    dataset_id = _upload_and_profile(dataset_service)
    r = api_client.post("/api/v1/experiments", json={"dataset_id": dataset_id})
    assert r.status_code == 409


def test_experiment_api_experiment_not_found(api_client):
    assert api_client.get("/api/v1/experiments/does-not-exist").status_code == 404
    assert (
        api_client.get("/api/v1/experiments/does-not-exist/comparison").status_code == 404
    )


# -- background failure ------------------------------------------------------


def test_experiment_api_background_mlflow_failure(
    api_client,
    dataset_service,
    experiment_service,
    monkeypatch,
):
    dataset_id = _prepared_dataset(dataset_service)

    def boom(**kwargs):
        raise TrackingError("MLflow tracking failed: ConnectionError")

    monkeypatch.setattr(experiment_service._tracker, "log_model_run", boom)

    created = api_client.post("/api/v1/experiments", json={"dataset_id": dataset_id})
    assert created.status_code == 202
    experiment_id = created.json()["id"]

    got = api_client.get(f"/api/v1/experiments/{experiment_id}")
    assert got.status_code == 200
    detail = got.json()
    assert detail["status"] == "failed"
    assert "MLflow" in detail["error_message"]


# -- request-id / audit ------------------------------------------------------


def test_experiment_api_request_id_and_audit(api_client, dataset_service, session_factory):
    dataset_id = _prepared_dataset(dataset_service)

    created = api_client.post(
        "/api/v1/experiments",
        json={"dataset_id": dataset_id},
        headers={"X-Request-ID": "exp-abc123"},
    )
    assert created.status_code == 202
    assert created.headers["x-request-id"] == "exp-abc123"
    experiment_id = created.json()["id"]

    got = api_client.get(
        f"/api/v1/experiments/{experiment_id}",
        headers={"X-Request-ID": "exp-abc123"},
    )
    assert got.status_code == 200
    assert got.headers["x-request-id"] == "exp-abc123"

    audit = AuditRepository(session_factory)
    events = audit.list(entity_type="experiment", entity_id=experiment_id)
    actions = {event.action for event in events}
    assert "experiment.create" in actions
    assert "experiment.run_started" in actions
    assert "experiment.run_succeeded" in actions

"""Integration tests for the P1 cost-aware comparison REST API.

These exercise ``POST /api/v1/experiments/{id}/cost-comparison`` end-to-end over
HTTP against an in-memory SQLite database and a real local MLflow backend. The
happy path reuses the persisted ``TrainingResult`` snapshot written by a
completed binary-classification run; the error paths assert stable 404 (missing
experiment), 409 (not yet run / incomplete), and 422 (non-binary result /
invalid request) responses.
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
from maintai.mlops import MLflowTracker

EXPERIMENT_NAME = "p1_cost_comparison_api_test"
CLASSIFICATION_MODELS = {"logistic_regression", "random_forest", "xgboost"}

_COST_BODY = {"fn_cost": 10.0, "fp_cost": 2.0, "minimum_recall": 0.0}


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


def _regression_csv(n: int = 240, seed: int = 0) -> bytes:
    """Synthetic regression CSV whose target is continuous float (many unique values)."""
    rng = np.random.default_rng(seed)
    sensor = np.round(rng.normal(0.0, 1.0, n), 6)
    remaining_life = np.round(1000.0 + sensor * 50.0 + rng.normal(0.0, 20.0, n), 6)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
            "asset_id": [f"M{i % 12:02d}" for i in range(n)],
            "serial_no": [f"s{i:05d}" for i in range(n)],
            "sensor": sensor,
            "cat": rng.choice(["a", "b", "c"], n),
            "flag": rng.choice([True, False], n),
            "remaining_life": remaining_life,
        }
    )
    return frame.to_csv(index=False).encode("utf-8")


def _prepare(dataset_service: DatasetService, csv: bytes, target_column: str) -> str:
    """Upload → profile → task-recommend and return the dataset id."""
    uploaded = dataset_service.upload(csv, "sensor_data.csv")
    dataset_id = uploaded["id"]
    dataset_service.profile(dataset_id)
    dataset_service.recommend_task(
        dataset_id,
        target_column=target_column,
        asset_id_column="asset_id",
        timestamp_column="timestamp",
    )
    return dataset_id


def _run_experiment(api_client, dataset_service: DatasetService, target_column: str) -> str:
    """Create + run an experiment over HTTP and return the experiment id."""
    dataset_id = _prepare(dataset_service, _classification_csv(), target_column)
    created = api_client.post("/api/v1/experiments", json={"dataset_id": dataset_id})
    assert created.status_code == 202
    return created.json()["id"]


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


def test_cost_comparison_happy_path(api_client, dataset_service):
    experiment_id = _run_experiment(api_client, dataset_service, "failure")

    r = api_client.post(
        f"/api/v1/experiments/{experiment_id}/cost-comparison", json=_COST_BODY
    )
    assert r.status_code == 200
    body = r.json()
    assert body["experiment_id"] == experiment_id
    assert body["minimum_recall"] == 0.0
    assert body["assumptions"] == {
        "fn_cost": 10.0,
        "fp_cost": 2.0,
        "currency_label": "demo units",
    }
    assert body["disclaimer"]
    assert body["metric_best"] is not None
    assert body["cost_best"] is not None
    assert {row["model_name"] for row in body["rows"]} == CLASSIFICATION_MODELS
    for row in body["rows"]:
        assert row["status"] == "ok"
        assert row["fn"] is not None
        assert row["fp"] is not None
        assert row["expected_error_cost"] is not None


def test_cost_comparison_is_deterministic(api_client, dataset_service):
    experiment_id = _run_experiment(api_client, dataset_service, "failure")
    url = f"/api/v1/experiments/{experiment_id}/cost-comparison"
    payload = {"fn_cost": 50.0, "fp_cost": 5.0, "minimum_recall": 0.8}

    first = api_client.post(url, json=payload)
    second = api_client.post(url, json=payload)
    assert first.status_code == 200
    assert first.json() == second.json()


# -- error paths -------------------------------------------------------------


def test_cost_comparison_experiment_not_found(api_client):
    r = api_client.post(
        "/api/v1/experiments/does-not-exist/cost-comparison", json=_COST_BODY
    )
    assert r.status_code == 404


def test_cost_comparison_queued_experiment_conflict(
    api_client, dataset_service, experiment_service
):
    dataset_id = _prepare(dataset_service, _classification_csv(), "failure")
    queued = experiment_service.create(dataset_id)  # created but never run

    r = api_client.post(
        f"/api/v1/experiments/{queued['id']}/cost-comparison", json=_COST_BODY
    )
    assert r.status_code == 409


def test_cost_comparison_non_binary_result_unprocessable(api_client, dataset_service):
    dataset_id = _prepare(dataset_service, _regression_csv(), "remaining_life")
    created = api_client.post("/api/v1/experiments", json={"dataset_id": dataset_id})
    assert created.status_code == 202
    experiment_id = created.json()["id"]

    r = api_client.post(
        f"/api/v1/experiments/{experiment_id}/cost-comparison", json=_COST_BODY
    )
    assert r.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {},  # missing required fields
        {"fn_cost": -1.0, "fp_cost": 1.0, "minimum_recall": 0.0},
        {"fn_cost": 1.0, "fp_cost": 1.0, "minimum_recall": 1.5},
        {"fn_cost": 1.0, "fp_cost": 1.0, "minimum_recall": 0.0, "shell": "rm -rf /"},
    ],
)
def test_cost_comparison_rejects_invalid_request(api_client, payload):
    # Body validation fails before the experiment id is ever resolved.
    r = api_client.post(
        "/api/v1/experiments/does-not-exist/cost-comparison", json=payload
    )
    assert r.status_code == 422

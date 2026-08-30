"""Integration tests for the P1 monitoring REST API.

Exercise ``POST /api/v1/monitoring/runs`` (synchronous 201), list, and detail
against an in-memory SQLite database with a real uploaded baseline dataset and a
real registered-model lineage, and a real ``MonitoringService`` wired to the
settings bridge. Covers: success + drift, list/get, source exclusivity (422),
model not-found / not-deployable / production-dataset-not-found status codes,
stable failed runs (201 + stable error), arbitrary-field rejection, request-id
propagation, and no filesystem-path leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from maintai.api.main import create_app
from maintai.api.monitoring import build_monitoring_service
from maintai.application.datasets import DatasetService
from maintai.audit.repository import AuditRepository
from maintai.audit.service import AuditService
from maintai.config import Settings
from maintai.db.dataset_repository import DatasetRepository
from maintai.db.experiment_repository import ExperimentRepository
from maintai.db.model_repository import ModelRepository
from maintai.db.models import (
    DEPLOYMENT_STATUS_CANDIDATE,
    DEPLOYMENT_STATUS_DEMO_DEPLOYED,
    EXPERIMENT_STATUS_SUCCEEDED,
    MODEL_RUN_STATUS_SUCCESS,
    Experiment,
    ModelRun,
    RegisteredModel,
    new_id,
)

MONITORING_URL = "/api/v1/monitoring/runs"


def _baseline_csv(n: int = 600, seed: int = 0, sensor_shift: float = 0.0) -> bytes:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
            "asset_id": [f"M{i % 20:02d}" for i in range(n)],
            "serial_no": [f"s{i:05d}" for i in range(n)],
            "sensor": np.round(rng.normal(sensor_shift, 0.5, n), 4),
            "vibration": np.round(rng.normal(10.0, 1.0, n), 4),
            "cat": rng.choice(["a", "b", "c"], n),
            "flag": rng.choice([True, False], n),
            "failure": (rng.random(n) < 0.25).astype(int),
        }
    )
    return frame.to_csv(index=False).encode("utf-8")


def _categorical_csv(n: int = 200, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
            "asset_id": [f"M{i % 20:02d}" for i in range(n)],
            "cat": rng.choice(["a", "b", "c"], n),
            "flag": rng.choice([True, False], n),
            "failure": (rng.random(n) < 0.25).astype(int),
        }
    )
    return frame.to_csv(index=False).encode("utf-8")


def _prepared_dataset(dataset_service: DatasetService, data: bytes) -> str:
    uploaded = dataset_service.upload(data, "baseline.csv")
    dataset_id = uploaded["id"]
    dataset_service.recommend_task(
        dataset_id,
        target_column="failure",
        asset_id_column="asset_id",
        timestamp_column="timestamp",
    )
    return dataset_id


def _register_lineage(
    session_factory,
    dataset_id: str,
    *,
    deployment_status: str = DEPLOYMENT_STATUS_DEMO_DEPLOYED,
) -> RegisteredModel:
    experiment = Experiment(
        id=new_id(),
        dataset_id=dataset_id,
        name=f"exp-binary-{dataset_id[:8]}",
        task_type="binary_classification",
        status=EXPERIMENT_STATUS_SUCCEEDED,
        training_plan_json={
            "plan": {
                "target": "failure",
                "task": "binary_classification",
                "features": [],
                "excluded": ["serial_no", "asset_id", "timestamp"],
                "model_names": ["random_forest"],
            }
        },
    )
    experiment_repository = ExperimentRepository(session_factory)
    experiment_repository.create(experiment)
    model_run = ModelRun(
        id=new_id(),
        experiment_id=experiment.id,
        model_name="random_forest",
        status=MODEL_RUN_STATUS_SUCCESS,
    )
    experiment_repository.create_model_run(model_run)
    registered = RegisteredModel(
        id=new_id(),
        model_run_id=model_run.id,
        experiment_id=experiment.id,
        name="failure-risk-demo",
        version="1",
        mlflow_model_uri="models:/failure-risk-demo/1",
        alias="candidate",
        approval_status="not_required_demo",
        deployment_status=deployment_status,
        deployed=(deployment_status == DEPLOYMENT_STATUS_DEMO_DEPLOYED),
    )
    ModelRepository(session_factory).create(registered)
    return registered


# -- fixtures ----------------------------------------------------------------


@pytest.fixture()
def storage_root(tmp_path):
    return tmp_path / "storage"


@pytest.fixture()
def dataset_service(session_factory, storage_root):
    return DatasetService(
        repository=DatasetRepository(session_factory),
        session_factory=session_factory,
        audit=AuditService(AuditRepository(session_factory)),
        storage_root=storage_root,
    )


@pytest.fixture()
def api_client(session_factory, dataset_service):
    settings = Settings(monitoring={"anomaly": {"contamination": 0.05, "seed": 42}})
    monitoring_service = build_monitoring_service(session_factory, settings, dataset_service)
    app = create_app(
        session_factory=session_factory,
        dataset_service=dataset_service,
        monitoring_service=monitoring_service,
        settings=settings,
        mlflow_healthcheck=lambda: None,
    )
    with TestClient(app) as client:
        yield client


def _deployed_model(session_factory, dataset_service) -> RegisteredModel:
    dataset_id = _prepared_dataset(dataset_service, _baseline_csv())
    return _register_lineage(session_factory, dataset_id)


# -- success / list / get -----------------------------------------------------


def test_post_run_returns_201_succeeded(session_factory, dataset_service, api_client):
    registered = _deployed_model(session_factory, dataset_service)

    r = api_client.post(
        MONITORING_URL, json={"model_id": registered.id, "replay_kind": "normal"}
    )
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "succeeded"
    assert body["registered_model_id"] == registered.id
    assert body["dataset_id"]
    assert body["replay_kind"] == "normal"
    assert body["production_dataset_id"] is None
    assert body["drift"]["overall_severity"] == "LOW"
    assert body["recommendation"]["recommended"] is False
    assert body["input_summary"]["source"] == "replay"
    assert body["error_message"] is None


def test_list_and_get(session_factory, dataset_service, api_client):
    registered = _deployed_model(session_factory, dataset_service)
    created = api_client.post(
        MONITORING_URL, json={"model_id": registered.id, "replay_kind": "severe"}
    ).json()

    listing = api_client.get(MONITORING_URL)
    assert listing.status_code == 200
    assert [item["id"] for item in listing.json()] == [created["id"]]

    got = api_client.get(f"{MONITORING_URL}/{created['id']}")
    assert got.status_code == 200
    assert got.json()["id"] == created["id"]
    assert got.json()["drift"]["overall_severity"] == "HIGH"

    assert api_client.get(f"{MONITORING_URL}/missing").status_code == 404


# -- source exclusivity / validation ------------------------------------------


def test_source_xor_returns_422(session_factory, dataset_service, api_client):
    registered = _deployed_model(session_factory, dataset_service)
    model_id = registered.id

    assert api_client.post(MONITORING_URL, json={"model_id": model_id}).status_code == 422
    assert (
        api_client.post(
            MONITORING_URL,
            json={"model_id": model_id, "replay_kind": "normal", "production_dataset_id": "x"},
        ).status_code
        == 422
    )
    assert (
        api_client.post(
            MONITORING_URL, json={"model_id": model_id, "replay_kind": "bogus"}
        ).status_code
        == 422
    )
    assert (
        api_client.post(
            MONITORING_URL,
            json={
                "model_id": model_id,
                "replay_kind": "normal",
                "baseline_anomaly_rate": 1.1,
            },
        ).status_code
        == 422
    )


def test_rejects_arbitrary_fields(session_factory, dataset_service, api_client):
    registered = _deployed_model(session_factory, dataset_service)
    r = api_client.post(
        MONITORING_URL,
        json={"model_id": registered.id, "replay_kind": "normal", "shell": "rm -rf /"},
    )
    assert r.status_code == 422


# -- status codes -------------------------------------------------------------


def test_model_errors_status_codes(session_factory, dataset_service, api_client):
    dataset_id = _prepared_dataset(dataset_service, _baseline_csv())
    archived = _register_lineage(session_factory, dataset_id, deployment_status="archived")

    assert (
        api_client.post(
            MONITORING_URL, json={"model_id": "missing", "replay_kind": "normal"}
        ).status_code
        == 404
    )
    assert (
        api_client.post(
            MONITORING_URL, json={"model_id": archived.id, "replay_kind": "normal"}
        ).status_code
        == 409
    )

    candidate = _register_lineage(
        session_factory, dataset_id, deployment_status=DEPLOYMENT_STATUS_CANDIDATE
    )
    assert (
        api_client.post(
            MONITORING_URL,
            json={"model_id": candidate.id, "production_dataset_id": "missing"},
        ).status_code
        == 404
    )


def test_failed_run_still_201_with_stable_error(session_factory, dataset_service, api_client):
    dataset_id = _prepared_dataset(dataset_service, _categorical_csv())
    registered = _register_lineage(session_factory, dataset_id)

    r = api_client.post(
        MONITORING_URL, json={"model_id": registered.id, "replay_kind": "normal"}
    )
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "failed"
    assert "numeric" in body["error_message"]
    assert "Traceback" not in body["error_message"]


# -- request id / path hygiene ------------------------------------------------


def test_request_id_preserved(session_factory, dataset_service, api_client):
    registered = _deployed_model(session_factory, dataset_service)
    r = api_client.post(
        MONITORING_URL,
        json={"model_id": registered.id, "replay_kind": "normal"},
        headers={"X-Request-ID": "mon-req-1"},
    )
    assert r.status_code == 201
    assert r.headers["x-request-id"] == "mon-req-1"


def test_no_filesystem_path_leak(session_factory, dataset_service, api_client, storage_root):
    registered = _deployed_model(session_factory, dataset_service)
    r = api_client.post(
        MONITORING_URL, json={"model_id": registered.id, "replay_kind": "normal"}
    )
    assert r.status_code == 201
    assert "file:" not in r.text
    assert str(storage_root) not in r.text

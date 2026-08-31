"""Integration tests for the P0 model registry + prediction REST API.

Exercise the HTTP layer end-to-end against an in-memory SQLite database, a real
MLflow local file backend, and real ``tmp_path`` artifact storage, with the
dataset/experiment/registry/prediction services injected into the app factory
(so the default builders — which point at ``http://mlflow:5000`` — are never
constructed and no network is touched).

The tests cover the full loop over HTTP — upload → profile → task → train →
register candidate → deploy demo → single + batch predict — plus the strict
feature contract (missing/extra/type), unknown categories, not-found/not-eligible/
conflict/not-deployed status codes, the absence of any champion/production route,
audit + prediction-event hygiene, and the guarantee that no filesystem path ever
leaks into a response.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from maintai.api.main import create_app
from maintai.application.datasets import DatasetService
from maintai.application.experiments import ExperimentService
from maintai.application.models import ModelRegistryService
from maintai.application.predictions import PredictionService
from maintai.audit.repository import AuditRepository
from maintai.audit.service import AuditService
from maintai.db.dataset_repository import DatasetRepository
from maintai.db.experiment_repository import ExperimentRepository
from maintai.db.model_repository import ModelRepository
from maintai.db.prediction_repository import PredictionRepository
from maintai.mlops import MLflowRegistry, MLflowTracker, RegistryError

EXPERIMENT_NAME = "p0_model_prediction_api_test"
CLASSIFICATION_MODELS = {"logistic_regression", "random_forest", "xgboost"}

DISCLAIMER = "Model-based explanation, not a verified physical root cause."

UPLOAD_URL = "/api/v1/datasets/upload"


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


def _record(sensor: float = 0.0, cat: str = "a", flag: bool = True) -> dict:
    return {"sensor": sensor, "cat": cat, "flag": flag}


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
def registry(tmp_path):
    return MLflowRegistry((tmp_path / "mlruns").as_uri())


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
def model_registry_service(session_factory, registry, artifact_root):
    return ModelRegistryService(
        session_factory=session_factory,
        model_repository=ModelRepository(session_factory),
        experiment_repository=ExperimentRepository(session_factory),
        audit=AuditService(AuditRepository(session_factory)),
        registry=registry,
        artifact_root=artifact_root,
    )


@pytest.fixture()
def prediction_service(session_factory, artifact_root):
    return PredictionService(
        session_factory=session_factory,
        model_repository=ModelRepository(session_factory),
        prediction_repository=PredictionRepository(session_factory),
        audit=AuditService(AuditRepository(session_factory)),
        artifact_root=artifact_root,
    )


@pytest.fixture()
def api_client(
    session_factory,
    dataset_service,
    experiment_service,
    model_registry_service,
    prediction_service,
):
    app = create_app(
        session_factory=session_factory,
        dataset_service=dataset_service,
        experiment_service=experiment_service,
        model_registry_service=model_registry_service,
        prediction_service=prediction_service,
        mlflow_healthcheck=lambda: None,
    )
    with TestClient(app) as client:
        yield client


# -- HTTP helpers -------------------------------------------------------------


def _upload(client: TestClient) -> str:
    r = client.post(
        UPLOAD_URL,
        files={"file": ("sensor_data.csv", _classification_csv(), "text/csv")},
    )
    assert r.status_code == 201
    return r.json()["id"]


def _prepare_dataset(client: TestClient) -> str:
    dataset_id = _upload(client)
    assert client.post(f"/api/v1/datasets/{dataset_id}/profile").status_code == 200
    assert (
        client.post(
            f"/api/v1/datasets/{dataset_id}/task-recommendation",
            json={
                "target_column": "failure",
                "asset_id_column": "asset_id",
                "timestamp_column": "timestamp",
            },
        ).status_code
        == 200
    )
    return dataset_id


def _train_detail(client: TestClient, dataset_id: str) -> dict:
    created = client.post("/api/v1/experiments", json={"dataset_id": dataset_id})
    assert created.status_code == 202
    experiment_id = created.json()["id"]
    detail = client.get(f"/api/v1/experiments/{experiment_id}").json()
    assert detail["status"] == "succeeded"
    return detail


def _recommended_run(detail: dict) -> dict:
    return next(
        run for run in detail["model_runs"]
        if run["model_name"] == detail["recommended_model"]
    )


def _register(client: TestClient, run_id: str, name: str | None = None) -> dict:
    body = {} if name is None else {"name": name}
    r = client.post(f"/api/v1/models/{run_id}/register", json=body)
    assert r.status_code == 200
    return r.json()


def _deployed_registered_model(client: TestClient) -> dict:
    dataset_id = _prepare_dataset(client)
    detail = _train_detail(client, dataset_id)
    run = _recommended_run(detail)
    registered = _register(client, run["id"])
    assert (
        client.post(f"/api/v1/models/{registered['id']}/deploy-demo").status_code == 200
    )
    return registered


# -- full lifecycle -----------------------------------------------------------


def test_full_flow_via_api(api_client, session_factory):
    dataset_id = _prepare_dataset(api_client)
    detail = _train_detail(api_client, dataset_id)
    run = _recommended_run(detail)

    # no models before registration
    assert api_client.get("/api/v1/models").json() == []

    registered = _register(api_client, run["id"], "failure-risk-demo")
    assert registered["deployment_status"] == "candidate"
    assert registered["name"] == "failure-risk-demo"
    assert registered["mlflow_model_uri"].startswith("models:/")
    assert registered["alias"] == "candidate"
    assert "artifact_uri" not in registered  # package basename is never exposed
    assert "file:" not in str(registered)

    # list + get return the stable shape
    listing = api_client.get("/api/v1/models").json()
    assert [item["id"] for item in listing] == [registered["id"]]
    got = api_client.get(f"/api/v1/models/{registered['id']}").json()
    assert got["id"] == registered["id"]
    assert got["mlflow_model_uri"].startswith("models:/")

    # candidate is not deployed yet → predict is rejected with 409
    r = api_client.post(
        "/api/v1/predict",
        json={"model_id": registered["id"], "records": [_record()]},
    )
    assert r.status_code == 409

    deployed = api_client.post(
        f"/api/v1/models/{registered['id']}/deploy-demo"
    ).json()
    assert deployed["deployment_status"] == "demo_deployed"

    # single predict (exactly one record) with explanation
    single = api_client.post(
        "/api/v1/predict",
        json={"model_id": registered["id"], "records": [_record(0.5, "a", True)]},
    )
    assert single.status_code == 200
    body = single.json()
    assert body["model_id"] == registered["id"]
    assert body["model_version"] == registered["version"]
    assert body["count"] == 1
    record = body["records"][0]
    assert record["prediction"] in (0, 1)
    assert isinstance(record["confidence"], float)
    assert isinstance(record["positive_probability"], float)
    assert record["explanation"]["disclaimer"] == DISCLAIMER

    # batch predict (2 records)
    batch = api_client.post(
        "/api/v1/predict/batch",
        json={
            "model_id": registered["id"],
            "records": [_record(0.0, "a", True), _record(1.5, "b", False)],
        },
    )
    assert batch.status_code == 200
    assert batch.json()["count"] == 2
    for record in batch.json()["records"]:
        assert record["prediction"] in (0, 1)

    # prediction events persisted with hashes, not raw input
    events = PredictionRepository(session_factory).list(
        registered_model_id=registered["id"]
    )
    assert len(events) == 3
    for event in events:
        assert event.model_version == registered["version"]
        assert len(event.input_hash) == 64

    # audit: register / deploy_demo / predict, predict payload never raw input
    audit = AuditRepository(session_factory)
    by_action = {
        event.action: event
        for event in audit.list(entity_type="registered_model", entity_id=registered["id"])
    }
    assert "model.register" in by_action
    assert "model.deploy_demo" in by_action
    predict_events = [
        event
        for event in audit.list(entity_type="registered_model", entity_id=registered["id"])
        if event.action == "prediction.predict"
    ]
    assert len(predict_events) == 2
    for event in predict_events:
        payload = event.payload_json or {}
        assert "records" not in payload
        assert "input" not in payload
        assert payload["count"] in (1, 2)
        assert len(payload["input_hashes"]) == payload["count"]


# -- per-behaviour status codes -----------------------------------------------


def test_candidate_not_deployed_predict_409(api_client):
    dataset_id = _prepare_dataset(api_client)
    detail = _train_detail(api_client, dataset_id)
    run = _recommended_run(detail)
    registered = _register(api_client, run["id"])

    assert registered["deployment_status"] == "candidate"
    r = api_client.post(
        "/api/v1/predict",
        json={"model_id": registered["id"], "records": [_record()]},
    )
    assert r.status_code == 409


def test_unknown_category_allowed(api_client):
    registered = _deployed_registered_model(api_client)

    r = api_client.post(
        "/api/v1/predict",
        json={"model_id": registered["id"], "records": [_record(0.5, "zzz", True)]},
    )
    assert r.status_code == 200
    assert r.json()["records"][0]["prediction"] in (0, 1)


def test_invalid_records_422(api_client):
    registered = _deployed_registered_model(api_client)
    model_id = registered["id"]

    # missing feature
    r = api_client.post(
        "/api/v1/predict",
        json={"model_id": model_id, "records": [{"sensor": 0.0, "cat": "a"}]},
    )
    assert r.status_code == 422

    # extra feature
    r = api_client.post(
        "/api/v1/predict",
        json={
            "model_id": model_id,
            "records": [{"sensor": 0.0, "cat": "a", "flag": True, "extra": 1}],
        },
    )
    assert r.status_code == 422

    # wrong type (string where numeric is required)
    r = api_client.post(
        "/api/v1/predict",
        json={"model_id": model_id, "records": [_record("hot", "a", True)]},
    )
    assert r.status_code == 422

    # single endpoint requires exactly one record
    r = api_client.post(
        "/api/v1/predict",
        json={"model_id": model_id, "records": [_record(), _record()]},
    )
    assert r.status_code == 422

    r = api_client.post("/api/v1/predict", json={"model_id": model_id, "records": []})
    assert r.status_code == 422

    # batch endpoint caps resource-heavy local explanations at 100 records
    r = api_client.post(
        "/api/v1/predict/batch",
        json={"model_id": model_id, "records": [_record()] * 1001},
    )
    assert r.status_code == 422


def test_invalid_body_422(api_client):
    dataset_id = _prepare_dataset(api_client)
    detail = _train_detail(api_client, dataset_id)
    run = _recommended_run(detail)

    # register with a path-separator name → 422
    r = api_client.post(
        f"/api/v1/models/{run['id']}/register", json={"name": "evil/name"}
    )
    assert r.status_code == 422

    # predict with unknown body fields → 422
    r = api_client.post(
        "/api/v1/predict",
        json={"model_id": "x", "records": [_record()], "shell": "rm -rf /"},
    )
    assert r.status_code == 422


def test_not_found_404(api_client):
    assert api_client.get("/api/v1/models/does-not-exist").status_code == 404
    assert (
        api_client.post("/api/v1/models/does-not-exist/deploy-demo").status_code == 404
    )
    assert api_client.post("/api/v1/models/does-not-exist/register", json={}).status_code == 404

    for url in ("/api/v1/predict", "/api/v1/predict/batch"):
        r = api_client.post(
            url, json={"model_id": "does-not-exist", "records": [_record()]}
        )
        assert r.status_code == 404


def test_register_conflict_409(api_client):
    dataset_id = _prepare_dataset(api_client)
    detail = _train_detail(api_client, dataset_id)
    run = _recommended_run(detail)

    _register(api_client, run["id"], "name-a")
    r = api_client.post(
        f"/api/v1/models/{run['id']}/register", json={"name": "name-b"}
    )
    assert r.status_code == 409


def test_register_not_eligible_409(api_client):
    dataset_id = _prepare_dataset(api_client)
    detail = _train_detail(api_client, dataset_id)
    non_recommended = next(
        run for run in detail["model_runs"]
        if run["model_name"] != detail["recommended_model"]
    )

    r = api_client.post(
        f"/api/v1/models/{non_recommended['id']}/register", json={}
    )
    assert r.status_code == 409


def test_unsupported_production_endpoints_absent(api_client):
    for path in (
        "/api/v1/models/x/champion",
        "/api/v1/models/x/deploy-production",
        "/api/v1/models/x/production",
    ):
        assert api_client.post(path, json={}).status_code == 404
        assert api_client.get(path).status_code == 404


# -- stable 500 without path leak ---------------------------------------------


def test_artifact_corruption_500_without_path(api_client, session_factory, artifact_root):
    registered = _deployed_registered_model(api_client)
    stored = ModelRepository(session_factory).get(registered["id"])
    (artifact_root / stored.artifact_uri).write_bytes(b"not a joblib archive")

    r = api_client.post(
        "/api/v1/predict",
        json={"model_id": registered["id"], "records": [_record()]},
    )
    assert r.status_code == 500
    assert r.json()["detail"] == "prediction failed"
    assert str(artifact_root) not in r.text
    assert "file:" not in r.text
    assert "Traceback" not in r.text


def test_register_registry_backend_500(api_client, model_registry_service, monkeypatch):
    dataset_id = _prepare_dataset(api_client)
    detail = _train_detail(api_client, dataset_id)
    run = _recommended_run(detail)

    def boom(run_id, name):
        raise RegistryError("MLflow registry failed: ConnectionError")

    monkeypatch.setattr(model_registry_service._registry, "register_run", boom)

    r = api_client.post(f"/api/v1/models/{run['id']}/register", json={})
    assert r.status_code == 500
    assert r.json()["detail"] == "model registration failed"


# -- path hygiene / request-id ------------------------------------------------


def test_no_filesystem_path_leak(api_client, artifact_root):
    registered = _deployed_registered_model(api_client)

    texts = [
        api_client.get("/api/v1/models").text,
        api_client.get(f"/api/v1/models/{registered['id']}").text,
        api_client.post(
            "/api/v1/predict",
            json={"model_id": registered["id"], "records": [_record()]},
        ).text,
    ]
    for text in texts:
        assert "file:" not in text
        assert str(artifact_root) not in text
        assert ".joblib" not in text


def test_request_id_preserved(api_client):
    registered = _deployed_registered_model(api_client)

    r = api_client.post(
        "/api/v1/predict",
        json={"model_id": registered["id"], "records": [_record()]},
        headers={"X-Request-ID": "pred-abc123"},
    )
    assert r.status_code == 200
    assert r.headers["x-request-id"] == "pred-abc123"

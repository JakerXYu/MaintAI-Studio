"""Integration tests for the P0 registry + prediction application services.

These exercise the full loop — upload → train → register candidate → reject
champion → deploy demo → single/batch predict — against a real in-memory SQLite
database, a real MLflow local file backend, and a real artifact directory (all
under ``tmp_path``), with a small synthetic dataset. They also cover the strict
feature contract, unknown categories, single-active-demo-version demotion,
prediction events + audit hygiene (no raw input), and artifact corruption/path
safety. A small regression artifact is constructed directly to verify the
empirical interval (and the absence of any probability).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from maintai.application.datasets import DatasetService
from maintai.application.experiments import ExperimentService
from maintai.application.models import ModelRegistryService
from maintai.application.predictions import (
    InvalidRecordsError,
    ModelNotDeployedError,
    PredictionError,
    PredictionService,
)
from maintai.audit.repository import AuditRepository
from maintai.audit.service import AuditService
from maintai.db.dataset_repository import DatasetRepository
from maintai.db.experiment_repository import ExperimentRepository
from maintai.db.model_repository import ModelRepository
from maintai.db.models import (
    DEPLOYMENT_STATUS_DEMO_DEPLOYED,
    RegisteredModel,
    new_id,
)
from maintai.db.prediction_repository import PredictionRepository
from maintai.ml import package
from maintai.ml.preprocess import build_preprocessor
from maintai.mlops import MLflowRegistry, MLflowTracker, RegistryError

EXPERIMENT_NAME = "p0_registry_prediction_test"
CLASSIFICATION_MODELS = {"logistic_regression", "random_forest", "xgboost"}

DISCLAIMER = "Model-based explanation, not a verified physical root cause."
EMPIRICAL_DISCLAIMER = "empirical prediction interval; not a statistical guarantee"


def _classification_csv(n: int = 240, n_pos: int = 120, seed: int = 0) -> bytes:
    """Small, balanced, synthetic binary-classification CSV (mirrors experiment tests)."""
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


def _prepared_dataset(dataset_service: DatasetService) -> str:
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


def _recommended_model_run(experiment_service: ExperimentService, experiment_id: str) -> dict:
    experiment = experiment_service.get(experiment_id)
    recommended = experiment["recommended_model"]
    return next(run for run in experiment["model_runs"] if run["model_name"] == recommended)


def _predict_audit(audit_repository: AuditRepository, registered_id: str):
    return [
        event
        for event in audit_repository.list(entity_type="registered_model", entity_id=registered_id)
        if event.action == "prediction.predict"
    ]


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


@pytest.fixture()
def model_repository(session_factory):
    return ModelRepository(session_factory)


@pytest.fixture()
def prediction_repository(session_factory):
    return PredictionRepository(session_factory)


@pytest.fixture()
def registry_service(
    session_factory,
    model_repository,
    audit_repository,
    registry,
    artifact_root,
):
    return ModelRegistryService(
        session_factory=session_factory,
        model_repository=model_repository,
        experiment_repository=ExperimentRepository(session_factory),
        audit=AuditService(audit_repository),
        registry=registry,
        artifact_root=artifact_root,
    )


@pytest.fixture()
def prediction_service(
    session_factory,
    model_repository,
    prediction_repository,
    audit_repository,
    artifact_root,
):
    return PredictionService(
        session_factory=session_factory,
        model_repository=model_repository,
        prediction_repository=prediction_repository,
        audit=AuditService(audit_repository),
        artifact_root=artifact_root,
    )


# -- full lifecycle ----------------------------------------------------------


def test_full_registry_and_prediction_lifecycle(
    dataset_service,
    experiment_service,
    registry,
    registry_service,
    prediction_service,
    audit_repository,
    prediction_repository,
):
    dataset_id = _prepared_dataset(dataset_service)
    experiment_id = experiment_service.create(dataset_id)["id"]
    result = experiment_service.run(experiment_id)
    assert result["status"] == "succeeded"
    run = _recommended_model_run(experiment_service, experiment_id)

    # register → candidate
    registered = registry_service.register(run["id"], "failure-risk-demo")
    assert registered["deployment_status"] == "candidate"
    assert registered["mlflow_model_uri"].startswith("models:/")
    assert registered["alias"] == "candidate"
    assert "artifact_uri" not in registered  # package basename is never exposed
    assert "file:" not in str(registered)

    # duplicate register is idempotent
    again = registry_service.register(run["id"], registered["name"])
    assert again["id"] == registered["id"]

    # champion promotion is rejected (P1, human approval)
    with pytest.raises(RegistryError, match="approval"):
        registry.set_alias(registered["name"], "champion", registered["version"])

    # not deployed yet → predict is rejected
    with pytest.raises(ModelNotDeployedError):
        prediction_service.predict(registered["id"], [_record()])

    # deploy demo
    deployed = registry_service.deploy_demo(registered["id"])
    assert deployed["deployment_status"] == "demo_deployed"
    assert registry_service.get(registered["id"])["deployment_status"] == "demo_deployed"

    # Copilot-style preview is read-only: no PredictionEvent or audit write.
    preview = prediction_service.predict(
        registered["id"],
        [_record(0.0, "a", True)],
        persist=False,
    )
    assert preview["count"] == 1
    assert prediction_repository.list(registered_model_id=registered["id"]) == []
    assert _predict_audit(audit_repository, registered["id"]) == []

    # single predict: decoded label + probabilities + explanation + disclaimer
    single = prediction_service.predict(registered["id"], [_record(0.0, "a", True)])
    assert single["model_version"] == registered["version"]
    assert single["count"] == 1
    rec = single["records"][0]
    assert rec["prediction"] in (0, 1)
    assert isinstance(rec["positive_probability"], float)
    assert isinstance(rec["confidence"], float)
    assert rec["positive_label"] in (0, 1)
    explanation = rec["explanation"]
    assert isinstance(explanation["top_positive"], list)
    assert isinstance(explanation["top_negative"], list)
    assert explanation["top_positive"] or explanation["top_negative"]
    assert explanation["disclaimer"] == DISCLAIMER

    # batch predict (2 records)
    batch = prediction_service.predict(
        registered["id"],
        [_record(0.0, "a", True), _record(1.5, "b", False)],
    )
    assert batch["count"] == 2
    assert len(batch["records"]) == 2
    for record in batch["records"]:
        assert record["prediction"] in (0, 1)
        assert record["explanation"]["disclaimer"] == DISCLAIMER

    # prediction events were persisted with hashes, not raw input
    events = prediction_repository.list(registered_model_id=registered["id"])
    assert len(events) == 3
    for event in events:
        assert event.model_version == registered["version"]
        assert len(event.input_hash) == 64
        assert "prediction" in (event.prediction_json or {})

    # audit records only count/model/input hashes, never the raw payload
    predict_audits = _predict_audit(audit_repository, registered["id"])
    assert len(predict_audits) == 2
    for event in predict_audits:
        payload = event.payload_json or {}
        assert "records" not in payload
        assert "input" not in payload
        assert payload["count"] in (1, 2)
        assert payload["model_name"] == registered["name"]
        assert len(payload["input_hashes"]) == payload["count"]


def test_unknown_category_is_allowed(
    dataset_service,
    experiment_service,
    registry_service,
    prediction_service,
):
    dataset_id = _prepared_dataset(dataset_service)
    experiment_id = experiment_service.create(dataset_id)["id"]
    experiment_service.run(experiment_id)
    run = _recommended_model_run(experiment_service, experiment_id)
    registered = registry_service.register(run["id"], run["model_name"])
    registry_service.deploy_demo(registered["id"])

    result = prediction_service.predict(registered["id"], [_record(0.5, "zzz", True)])
    assert result["count"] == 1
    assert result["records"][0]["prediction"] in (0, 1)


def test_missing_and_extra_features_rejected(
    dataset_service,
    experiment_service,
    registry_service,
    prediction_service,
):
    dataset_id = _prepared_dataset(dataset_service)
    experiment_id = experiment_service.create(dataset_id)["id"]
    experiment_service.run(experiment_id)
    run = _recommended_model_run(experiment_service, experiment_id)
    registered = registry_service.register(run["id"], run["model_name"])
    registry_service.deploy_demo(registered["id"])

    with pytest.raises(InvalidRecordsError, match="missing"):
        prediction_service.predict(registered["id"], [{"sensor": 0.0, "cat": "a"}])
    with pytest.raises(InvalidRecordsError, match="unexpected"):
        prediction_service.predict(
            registered["id"],
            [{"sensor": 0.0, "cat": "a", "flag": True, "extra": 1}],
        )
    with pytest.raises(InvalidRecordsError, match="between"):
        prediction_service.predict(registered["id"], [])
    with pytest.raises(InvalidRecordsError, match="numeric"):
        prediction_service.predict(registered["id"], [_record("hot", "a", True)])


def test_single_active_demo_version(
    dataset_service,
    experiment_service,
    registry_service,
    model_repository,
):
    dataset_id = _prepared_dataset(dataset_service)
    first_id = experiment_service.create(dataset_id)["id"]
    second_id = experiment_service.create(dataset_id)["id"]
    experiment_service.run(first_id)
    experiment_service.run(second_id)

    run1 = _recommended_model_run(experiment_service, first_id)
    run2 = _recommended_model_run(experiment_service, second_id)
    name = run1["model_name"]

    reg1 = registry_service.register(run1["id"], name)
    reg2 = registry_service.register(run2["id"], name)
    assert reg1["version"] != reg2["version"]

    registry_service.deploy_demo(reg1["id"])
    registry_service.deploy_demo(reg2["id"])

    assert registry_service.get(reg1["id"])["deployment_status"] == "candidate"
    assert registry_service.get(reg2["id"])["deployment_status"] == "demo_deployed"

    deployed_versions = [model for model in model_repository.list_by_name(name) if model.deployed]
    assert len(deployed_versions) == 1
    assert deployed_versions[0].id == reg2["id"]


# -- artifact corruption / path safety ----------------------------------------


def test_corrupt_artifact_yields_stable_error(
    dataset_service,
    experiment_service,
    registry_service,
    prediction_service,
    model_repository,
    artifact_root,
):
    dataset_id = _prepared_dataset(dataset_service)
    experiment_id = experiment_service.create(dataset_id)["id"]
    experiment_service.run(experiment_id)
    run = _recommended_model_run(experiment_service, experiment_id)
    registered = registry_service.register(run["id"], run["model_name"])
    registry_service.deploy_demo(registered["id"])

    stored = model_repository.get(registered["id"])
    (artifact_root / stored.artifact_uri).write_bytes(b"not a joblib archive")

    with pytest.raises(PredictionError) as exc_info:
        prediction_service.predict(registered["id"], [_record()])
    message = str(exc_info.value)
    assert str(artifact_root) not in message
    assert "file:" not in message
    assert "Traceback" not in message


def test_path_traversal_artifact_rejected(
    prediction_service,
    model_repository,
):
    model = RegisteredModel(
        id=new_id(),
        name="evil",
        version="1",
        artifact_uri="../../evil.joblib",
        mlflow_model_uri="models:/evil/1",
        deployment_status=DEPLOYMENT_STATUS_DEMO_DEPLOYED,
        deployed=True,
    )
    model_repository.create(model)

    with pytest.raises(PredictionError) as exc_info:
        prediction_service.predict(model.id, [_record()])
    assert "file:" not in str(exc_info.value)


# -- regression path ----------------------------------------------------------


def test_regression_interval_without_probability(
    prediction_service,
    model_repository,
    artifact_root,
):
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(
        {
            "num": np.round(rng.normal(0.0, 1.0, n), 4),
            "cat": rng.choice(["a", "b"], n),
        }
    )
    y = X["num"] * 1.5 + (X["cat"] == "b").astype(int) * 2.0 + rng.normal(0.0, 0.1, n)
    preprocessor = build_preprocessor(X, ["num", "cat"])
    preprocessor.fit(X)
    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", RandomForestRegressor(n_estimators=20, random_state=0)),
        ]
    )
    pipeline.fit(X, y)
    residuals = np.abs(y.to_numpy() - pipeline.predict(X))
    quantile = float(np.quantile(residuals, 0.9))

    path = package.save_artifact(
        pipeline,
        model_name="reg_demo",
        version="1",
        task="regression",
        features=["num", "cat"],
        training_summary={
            "residual_interval": {"coverage": 0.9, "quantile": quantile},
            "feature_baseline": {
                "num": float(X["num"].median()),
                "cat": X["cat"].mode().iloc[0],
            },
        },
        artifact_dir=artifact_root,
    )

    model = RegisteredModel(
        id=new_id(),
        name="reg_demo",
        version="1",
        artifact_uri=Path(path).name,
        mlflow_model_uri="models:/reg_demo/1",
        deployment_status=DEPLOYMENT_STATUS_DEMO_DEPLOYED,
        deployed=True,
    )
    model_repository.create(model)

    result = prediction_service.predict(model.id, [{"num": 0.5, "cat": "a"}])
    record = result["records"][0]
    assert isinstance(record["prediction"], float)
    assert "probability" not in record
    assert "positive_probability" not in record
    assert "confidence" not in record
    assert "interval" in record
    assert record["interval"]["lower"] <= record["prediction"] <= record["interval"]["upper"]
    assert record["interval"]["coverage"] == 0.9
    assert record["interval"]["disclaimer"] == EMPIRICAL_DISCLAIMER
    assert record["explanation"]["disclaimer"] == DISCLAIMER

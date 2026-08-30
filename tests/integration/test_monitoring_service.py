"""Integration tests for the P1 monitoring application service + repository.

These exercise a full monitoring run end-to-end against an in-memory SQLite
database (``StaticPool`` test double) and real dataset storage (``tmp_path``),
with a simplified-but-real P0 lineage (a real uploaded baseline dataset, a real
experiment/model-run/registered-model metadata chain, and a real
``MonitoringService`` wired to the settings bridge). They cover: normal/severe
replay drift, increased-failure-risk anomaly + recommendation, uploaded
production datasets, persistence + audit hygiene (no raw rows), source
exclusivity, not-found / not-deployable, stable failure, DB check constraints,
and the guarantee that monitoring never mutates approvals/experiments/registry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from maintai.application.datasets import DatasetService
from maintai.application.monitoring import (
    MonitoringDatasetNotFoundError,
    MonitoringModelNotFoundError,
    MonitoringNotDeployableError,
    MonitoringService,
    MonitoringSourceError,
)
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
    ApprovalRequest,
    Experiment,
    ModelRun,
    MonitoringRun,
    RegisteredModel,
    new_id,
)
from maintai.db.monitoring_repository import MonitoringRepository
from maintai.monitoring.config import anomaly_config, drift_config, recommend_config


def _baseline_csv(n: int = 600, seed: int = 0, sensor_shift: float = 0.0) -> bytes:
    """Synthetic binary-classification CSV with numeric + categorical features."""
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
    """Categorical-only CSV (no numeric feature columns) to force a stable failure."""
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
    target: str = "failure",
    resolved_features: list[str] | None = None,
) -> RegisteredModel:
    """Persist a real experiment/model-run/registered-model lineage without training."""
    excluded = ["serial_no", "asset_id", "timestamp"]
    experiment = Experiment(
        id=new_id(),
        dataset_id=dataset_id,
        name=f"exp-binary-{dataset_id[:8]}",
        task_type="binary_classification",
        status=EXPERIMENT_STATUS_SUCCEEDED,
        training_plan_json={
            "resolved_features": resolved_features
            or ["sensor", "vibration", "cat", "flag"],
            "plan": {
                "target": target,
                "task": "binary_classification",
                "features": [],
                "excluded": excluded,
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
def audit_repository(session_factory):
    return AuditRepository(session_factory)


@pytest.fixture()
def monitoring_repository(session_factory):
    return MonitoringRepository(session_factory)


@pytest.fixture()
def service(session_factory, dataset_service, audit_repository):
    settings = Settings(monitoring={"anomaly": {"contamination": 0.05, "seed": 42}})
    return MonitoringService(
        session_factory=session_factory,
        dataset_service=dataset_service,
        model_repository=ModelRepository(session_factory),
        experiment_repository=ExperimentRepository(session_factory),
        repository=MonitoringRepository(session_factory),
        audit=AuditService(audit_repository),
        anomaly_config=anomaly_config(settings),
        drift_config=drift_config(settings),
        recommend_config=recommend_config(settings),
    )


# -- replay drift / recommendation -------------------------------------------


def test_normal_replay_is_low_drift(service, session_factory, dataset_service):
    dataset_id = _prepared_dataset(dataset_service, _baseline_csv())
    registered = _register_lineage(session_factory, dataset_id)

    result = service.run(registered.id, replay_kind="normal")

    assert result["status"] == "succeeded", result["error_message"]
    assert result["replay_kind"] == "normal"
    assert result["production_dataset_id"] is None
    assert result["drift"]["overall_severity"] == "LOW"
    assert result["recommendation"]["recommended"] is False
    assert result["anomaly"]["numeric_features"]
    assert "rows" not in result["anomaly"]
    assert len(result["anomaly"]["top_anomalies"]) <= 20
    assert all(
        "value" not in feature
        for row in result["anomaly"]["top_anomalies"]
        for feature in row["top_deviating_features"]
    )
    assert result["input_summary"]["feature_allowlist"] == [
        "sensor",
        "vibration",
        "cat",
        "flag",
    ]


def test_severe_replay_is_high_drift_and_recommends(
    service, session_factory, dataset_service
):
    dataset_id = _prepared_dataset(dataset_service, _baseline_csv())
    registered = _register_lineage(session_factory, dataset_id)

    result = service.run(registered.id, replay_kind="severe")

    assert result["status"] == "succeeded"
    assert result["drift"]["overall_severity"] == "HIGH"
    assert result["recommendation"]["recommended"] is True
    assert "high_drift" in result["recommendation"]["triggers"]


def test_increased_failure_risk_anomaly_and_recommend(
    service, session_factory, dataset_service
):
    dataset_id = _prepared_dataset(dataset_service, _baseline_csv())
    registered = _register_lineage(session_factory, dataset_id)

    result = service.run(registered.id, replay_kind="increased_failure_risk")

    assert result["status"] == "succeeded"
    assert result["anomaly"]["flagged_count"] > 0
    assert result["recommendation"]["recommended"] is True


# -- uploaded production ------------------------------------------------------


def test_uploaded_production_dataset(service, session_factory, dataset_service):
    baseline_id = _prepared_dataset(dataset_service, _baseline_csv())
    registered = _register_lineage(session_factory, baseline_id)
    production_id = dataset_service.upload(
        _baseline_csv(seed=1, sensor_shift=2.0), "production.csv"
    )["id"]

    result = service.run(registered.id, production_dataset_id=production_id)

    assert result["status"] == "succeeded"
    assert result["production_dataset_id"] == production_id
    assert result["replay_kind"] is None
    assert result["drift"] is not None
    assert result["anomaly"] is not None
    assert result["input_summary"]["source"] == "uploaded"


def test_uploaded_production_dataset_not_found(service, session_factory, dataset_service):
    dataset_id = _prepared_dataset(dataset_service, _baseline_csv())
    registered = _register_lineage(session_factory, dataset_id)

    with pytest.raises(MonitoringDatasetNotFoundError):
        service.run(registered.id, production_dataset_id="missing")


# -- persistence / audit ------------------------------------------------------


def test_run_persists_and_audits_without_raw_rows(
    service, session_factory, dataset_service, audit_repository, monitoring_repository
):
    dataset_id = _prepared_dataset(dataset_service, _baseline_csv())
    registered = _register_lineage(session_factory, dataset_id)

    result = service.run(registered.id, replay_kind="normal")

    persisted = monitoring_repository.get(result["id"])
    assert persisted is not None
    assert persisted.status == "succeeded"
    assert persisted.drift_json is not None
    assert persisted.anomaly_json is not None
    assert persisted.recommendation_json is not None
    assert persisted.input_summary_json is not None
    assert persisted.completed_at is not None

    events = audit_repository.list(entity_type="registered_model", entity_id=registered.id)
    run_events = [e for e in events if e.action == "monitoring.run"]
    assert len(run_events) == 1
    payload = run_events[0].payload_json or {}
    assert payload["monitoring_run_id"] == result["id"]
    assert payload["status"] == "succeeded"
    assert payload["source"] == "replay"
    assert payload["replay_kind"] == "normal"
    # audit stores only metadata, never raw rows or result blobs
    assert "records" not in payload
    assert "rows" not in payload
    assert "features" not in payload
    assert "drift" not in payload
    assert "anomaly" not in payload
    assert "recommendation" not in payload


# -- source exclusivity -------------------------------------------------------


def test_source_xor_validation(service, session_factory, dataset_service):
    dataset_id = _prepared_dataset(dataset_service, _baseline_csv())
    registered = _register_lineage(session_factory, dataset_id)

    with pytest.raises(MonitoringSourceError):
        service.run(registered.id)  # neither source
    with pytest.raises(MonitoringSourceError):
        service.run(registered.id, replay_kind="normal", production_dataset_id=dataset_id)
    with pytest.raises(MonitoringSourceError):
        service.run(registered.id, replay_kind="bogus")
    with pytest.raises(MonitoringSourceError, match="differ"):
        service.run(registered.id, production_dataset_id=dataset_id)


# -- model eligibility --------------------------------------------------------


def test_not_deployable_and_not_found(service, session_factory, dataset_service):
    dataset_id = _prepared_dataset(dataset_service, _baseline_csv())
    archived = _register_lineage(session_factory, dataset_id, deployment_status="archived")

    with pytest.raises(MonitoringNotDeployableError):
        service.run(archived.id, replay_kind="normal")
    with pytest.raises(MonitoringModelNotFoundError):
        service.run("missing", replay_kind="normal")

    candidate = _register_lineage(
        session_factory, dataset_id, deployment_status=DEPLOYMENT_STATUS_CANDIDATE
    )
    result = service.run(candidate.id, replay_kind="normal")
    assert result["status"] == "succeeded"


# -- stable failure -----------------------------------------------------------


def test_failure_persists_stable_error(
    service, session_factory, dataset_service, monitoring_repository
):
    dataset_id = _prepared_dataset(dataset_service, _categorical_csv())
    registered = _register_lineage(
        session_factory,
        dataset_id,
        resolved_features=["cat", "flag"],
    )

    result = service.run(registered.id, replay_kind="normal")

    assert result["status"] == "failed"
    assert "numeric" in result["error_message"]
    assert "Traceback" not in result["error_message"]
    assert result["drift"] is None
    assert result["anomaly"] is None

    persisted = monitoring_repository.get(result["id"])
    assert persisted.status == "failed"
    assert persisted.error_message == result["error_message"]


# -- no side effects ----------------------------------------------------------


def test_no_approval_experiment_or_registry_mutation(
    service, session_factory, dataset_service
):
    dataset_id = _prepared_dataset(dataset_service, _baseline_csv())
    registered = _register_lineage(session_factory, dataset_id)

    with session_factory() as session:
        before_approvals = session.scalar(select(func.count()).select_from(ApprovalRequest))
        before_experiments = session.scalar(select(func.count()).select_from(Experiment))
        before_registered = session.scalar(select(func.count()).select_from(RegisteredModel))

    service.run(registered.id, replay_kind="severe")

    with session_factory() as session:
        after_approvals = session.scalar(select(func.count()).select_from(ApprovalRequest))
        after_experiments = session.scalar(select(func.count()).select_from(Experiment))
        after_registered = session.scalar(select(func.count()).select_from(RegisteredModel))

    assert after_approvals == before_approvals == 0
    assert after_experiments == before_experiments
    assert after_registered == before_registered
    # the service owns no training, deployment, or registry capability
    assert not hasattr(service, "registry")
    assert not hasattr(service, "train")


# -- database check constraints ----------------------------------------------


def _insert_raw(session_factory, **overrides):
    kwargs = {
        "registered_model_id": "rm-1",
        "dataset_id": "ds-1",
        "replay_kind": "normal",
        "status": "succeeded",
    }
    kwargs.update(overrides)
    with session_factory.begin() as session:
        session.add(MonitoringRun(id=new_id(), **kwargs))


def test_monitoring_run_check_constraints(session_factory):
    with pytest.raises(IntegrityError):
        _insert_raw(session_factory, production_dataset_id="p", replay_kind="normal")
    with pytest.raises(IntegrityError):
        _insert_raw(session_factory, replay_kind=None, production_dataset_id=None)
    with pytest.raises(IntegrityError):
        _insert_raw(session_factory, status="bogus")
    with pytest.raises(IntegrityError):
        _insert_raw(session_factory, replay_kind="bogus")

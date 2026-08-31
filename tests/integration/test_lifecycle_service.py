"""Integration tests for the P1 champion/challenger lifecycle service.

Exercises the full promotion loop — record challenger + propose approval →
approve/modify → execute promotion (MLflow champion alias + archive old
champion + idempotent receipt + audit) — against an in-memory SQLite database
and a real MLflow local file backend (``tmp_path``). Also covers request
guards (already champion / duplicate pending), execution guards (pending
approval, action/entity/effective-payload validation), modified-decision
payloads, and idempotency. P0 ``deployment_status``/``deployed`` semantics are
left untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from maintai.application.lifecycle import (
    LifecycleApprovalNotReadyError,
    LifecycleNotPromotableError,
    LifecycleValidationError,
    ModelLifecycleService,
)
from maintai.approvals import ApprovalConflictError, ApprovalService
from maintai.audit.repository import AuditRepository
from maintai.audit.service import AuditService
from maintai.db.approval_execution_repository import ApprovalExecutionRepository
from maintai.db.approval_repository import ApprovalRepository
from maintai.db.lifecycle_repository import ModelLifecycleRepository
from maintai.db.model_repository import ModelRepository
from maintai.db.models import (
    APPROVAL_ACTION_MODEL_PROMOTION,
    APPROVAL_STATUS_APPROVED,
    APPROVAL_STATUS_MODIFIED,
    APPROVAL_STATUS_PENDING,
    MODEL_LIFECYCLE_ARCHIVED,
    MODEL_LIFECYCLE_CHALLENGER,
    MODEL_LIFECYCLE_CHAMPION,
    ApprovalRequest,
    RegisteredModel,
    new_id,
)
from maintai.ml.preprocess import build_preprocessor
from maintai.mlops import MLflowRegistry, MLflowTracker

EXPERIMENT_NAME = "p1_lifecycle_test"
MODEL_NAME = "failure_model"


def _fit_pipeline():
    rng = np.random.default_rng(0)
    n = 80
    frame = pd.DataFrame(
        {
            "num": rng.normal(0, 1, n),
            "cat": rng.choice(["a", "b"], n),
            "target": (rng.normal(0, 1, n) > 0).astype(int),
        }
    )
    features = ["num", "cat"]
    X = frame[features]
    preprocessor = build_preprocessor(X, features)
    preprocessor.fit(X)
    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(n_estimators=10, random_state=0)),
        ]
    )
    pipeline.fit(X, frame["target"])
    return pipeline, X


# -- fixtures ----------------------------------------------------------------


@pytest.fixture()
def mlflow_registry(tmp_path):
    return MLflowRegistry((tmp_path / "mlruns").as_uri())


@pytest.fixture()
def tracker(tmp_path):
    return MLflowTracker((tmp_path / "mlruns").as_uri(), EXPERIMENT_NAME)


@pytest.fixture()
def model_repository(session_factory):
    return ModelRepository(session_factory)


def _register_model(mlflow_registry, tracker, model_repository, *, f1: float) -> dict:
    """Log a run, register it in MLflow, and mirror it as a DB ``RegisteredModel``."""
    pipeline, _ = _fit_pipeline()
    tracked = tracker.log_model_run(
        model_name=MODEL_NAME,
        task="binary_classification",
        pipeline=pipeline,
        metrics={"f1": f1},
    )
    rv = mlflow_registry.register_run(tracked.run_id, MODEL_NAME)
    model = RegisteredModel(
        id=new_id(),
        name=MODEL_NAME,
        version=rv.version,
        mlflow_model_uri=rv.model_uri,
        alias=rv.alias,
        deployment_status="candidate",
        deployed=False,
    )
    model_repository.create(model)
    return {"id": model.id, "name": MODEL_NAME, "version": rv.version}


def _make_services(session_factory, model_repository, mlflow_registry):
    """Build a lifecycle service plus the shared approval service it uses."""
    audit = AuditService(AuditRepository(session_factory))
    approval_repository = ApprovalRepository(session_factory)
    approval_service = ApprovalService(
        session_factory=session_factory,
        repository=approval_repository,
        audit=audit,
    )
    lifecycle_service = ModelLifecycleService(
        session_factory=session_factory,
        model_repository=model_repository,
        lifecycle_repository=ModelLifecycleRepository(session_factory),
        approval_repository=approval_repository,
        execution_repository=ApprovalExecutionRepository(session_factory),
        audit=audit,
        registry=mlflow_registry,
        approval_service=approval_service,
    )
    return lifecycle_service, approval_service


def _decide(approval_service, approval_id, decision="approve", **kwargs):
    kwargs.setdefault("human_actor_id", "engineer-1")
    return approval_service.decide(approval_id, decision=decision, **kwargs)


# -- request guards -----------------------------------------------------------


def test_request_promotion_creates_challenger_and_approval(
    session_factory, model_repository, mlflow_registry, tracker
):
    svc, _ = _make_services(session_factory, model_repository, mlflow_registry)
    model = _register_model(mlflow_registry, tracker, model_repository, f1=0.9)

    result = svc.request_promotion(model["id"])

    assert result["lifecycle_status"] == MODEL_LIFECYCLE_CHALLENGER
    assert result["approval_status"] == APPROVAL_STATUS_PENDING
    assert result["proposed_payload"]["registered_model_id"] == model["id"]

    lifecycle = ModelLifecycleRepository(session_factory).get_by_registered_model(model["id"])
    assert lifecycle is not None
    assert lifecycle.status == MODEL_LIFECYCLE_CHALLENGER

    approval = ApprovalRepository(session_factory).get(result["approval_id"])
    assert approval.action_type == APPROVAL_ACTION_MODEL_PROMOTION
    assert approval.entity_type == "registered_model"
    assert approval.entity_id == model["id"]


def test_duplicate_promotion_request_conflicts(
    session_factory, model_repository, mlflow_registry, tracker
):
    svc, _ = _make_services(session_factory, model_repository, mlflow_registry)
    model = _register_model(mlflow_registry, tracker, model_repository, f1=0.9)
    svc.request_promotion(model["id"])

    with pytest.raises(ApprovalConflictError, match="already pending"):
        svc.request_promotion(model["id"])


def test_request_promotion_rejects_champion(
    session_factory, model_repository, mlflow_registry, tracker
):
    svc, approval_service = _make_services(session_factory, model_repository, mlflow_registry)
    model = _register_model(mlflow_registry, tracker, model_repository, f1=0.9)

    request = svc.request_promotion(model["id"])
    _decide(approval_service, request["approval_id"])
    svc.execute_promotion(model["id"], request["approval_id"], human_actor_id="engineer-1")

    with pytest.raises(LifecycleNotPromotableError, match="champion"):
        svc.request_promotion(model["id"])


# -- execution guards ---------------------------------------------------------


def test_execute_promotion_requires_approved(
    session_factory, model_repository, mlflow_registry, tracker
):
    svc, _ = _make_services(session_factory, model_repository, mlflow_registry)
    model = _register_model(mlflow_registry, tracker, model_repository, f1=0.9)
    request = svc.request_promotion(model["id"])

    with pytest.raises(LifecycleApprovalNotReadyError, match="pending"):
        svc.execute_promotion(model["id"], request["approval_id"], human_actor_id="engineer-1")


def test_execute_rejects_wrong_action_entity_and_payload(
    session_factory, model_repository, mlflow_registry, tracker
):
    svc, _ = _make_services(session_factory, model_repository, mlflow_registry)
    model = _register_model(mlflow_registry, tracker, model_repository, f1=0.9)
    approval_repo = ApprovalRepository(session_factory)

    def raw_approval(**overrides):
        kwargs = {
            "action_type": APPROVAL_ACTION_MODEL_PROMOTION,
            "entity_type": "registered_model",
            "entity_id": model["id"],
            "status": APPROVAL_STATUS_APPROVED,
            "requested_by_type": "agent",
            "version": 1,
            "proposed_payload": {"registered_model_id": model["id"]},
        }
        kwargs.update(overrides)
        approval = ApprovalRequest(id=new_id(), **kwargs)
        approval_repo.create(approval)
        return approval

    wrong_action = raw_approval(action_type="retraining_deployment")
    with pytest.raises(LifecycleValidationError, match="action"):
        svc.execute_promotion(model["id"], wrong_action.id, human_actor_id="engineer-1")

    wrong_entity = raw_approval(entity_id="some-other-model")
    with pytest.raises(LifecycleValidationError, match="entity_id"):
        svc.execute_promotion(model["id"], wrong_entity.id, human_actor_id="engineer-1")

    wrong_payload = raw_approval(proposed_payload={"registered_model_id": "other-model"})
    with pytest.raises(LifecycleValidationError, match="registered_model_id"):
        svc.execute_promotion(model["id"], wrong_payload.id, human_actor_id="engineer-1")


# -- full promotion flow ------------------------------------------------------


def test_promote_archives_previous_champion(
    session_factory, model_repository, mlflow_registry, tracker
):
    svc, approval_service = _make_services(session_factory, model_repository, mlflow_registry)
    v1 = _register_model(mlflow_registry, tracker, model_repository, f1=0.5)
    v2 = _register_model(mlflow_registry, tracker, model_repository, f1=0.95)

    request1 = svc.request_promotion(v1["id"])
    _decide(approval_service, request1["approval_id"])
    result1 = svc.execute_promotion(v1["id"], request1["approval_id"], human_actor_id="engineer-1")
    assert result1["executed"] is True
    assert result1["lifecycle_status"] == MODEL_LIFECYCLE_CHAMPION
    assert result1["archived_champion_ids"] == []

    request2 = svc.request_promotion(v2["id"])
    _decide(approval_service, request2["approval_id"])
    result2 = svc.execute_promotion(v2["id"], request2["approval_id"], human_actor_id="engineer-1")
    assert result2["executed"] is True
    assert result2["archived_champion_ids"] == [v1["id"]]

    lifecycle_repo = ModelLifecycleRepository(session_factory)
    assert lifecycle_repo.get_by_registered_model(v1["id"]).status == MODEL_LIFECYCLE_ARCHIVED
    assert lifecycle_repo.get_by_registered_model(v2["id"]).status == MODEL_LIFECYCLE_CHAMPION

    client = MlflowClient(mlflow_registry.tracking_uri)
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    assert str(champion.version) == v2["version"]


def test_execute_promotion_is_idempotent(
    session_factory, model_repository, mlflow_registry, tracker
):
    svc, approval_service = _make_services(session_factory, model_repository, mlflow_registry)
    model = _register_model(mlflow_registry, tracker, model_repository, f1=0.9)
    request = svc.request_promotion(model["id"])
    _decide(approval_service, request["approval_id"])

    first = svc.execute_promotion(model["id"], request["approval_id"], human_actor_id="engineer-1")
    second = svc.execute_promotion(model["id"], request["approval_id"], human_actor_id="engineer-1")

    assert first["executed"] is True
    assert second["executed"] is False
    assert second["receipt_id"] == first["receipt_id"]
    assert len(ApprovalExecutionRepository(session_factory).list()) == 1


def test_modified_uses_complete_decision_payload(
    session_factory, model_repository, mlflow_registry, tracker
):
    svc, approval_service = _make_services(session_factory, model_repository, mlflow_registry)
    model = _register_model(mlflow_registry, tracker, model_repository, f1=0.9)
    request = svc.request_promotion(model["id"])

    _decide(
        approval_service,
        request["approval_id"],
        decision="modify",
        reason="retarget the champion alias",
        payload={
            "registered_model_id": model["id"],
            "name": model["name"],
            "version": model["version"],
            "to_alias": "champion",
        },
    )

    result = svc.execute_promotion(model["id"], request["approval_id"], human_actor_id="engineer-1")
    assert result["executed"] is True
    assert result["approval_status"] == APPROVAL_STATUS_MODIFIED

    # A modified payload that retargets a different model must be rejected.
    other = _register_model(mlflow_registry, tracker, model_repository, f1=0.1)
    request2 = svc.request_promotion(other["id"])
    _decide(
        approval_service,
        request2["approval_id"],
        decision="modify",
        reason="wrong target",
        payload={"registered_model_id": "different-model"},
    )
    with pytest.raises(LifecycleValidationError, match="registered_model_id"):
        svc.execute_promotion(other["id"], request2["approval_id"], human_actor_id="engineer-1")


def test_execute_audits_mutation_without_payload_leak(
    session_factory, model_repository, mlflow_registry, tracker
):
    svc, approval_service = _make_services(session_factory, model_repository, mlflow_registry)
    model = _register_model(mlflow_registry, tracker, model_repository, f1=0.9)
    request = svc.request_promotion(model["id"])
    _decide(approval_service, request["approval_id"])
    svc.execute_promotion(model["id"], request["approval_id"], human_actor_id="engineer-1")

    events = AuditRepository(session_factory).list(
        entity_type="registered_model", entity_id=model["id"]
    )
    promote_events = [e for e in events if e.action == "model.promote_champion"]
    assert len(promote_events) == 1
    assert promote_events[0].actor_type == "user"
    assert promote_events[0].actor_id == "engineer-1"
    payload = promote_events[0].payload_json or {}
    assert payload["registered_model_id"] == model["id"]
    assert payload["alias"] == "champion"
    assert "proposed_payload" not in payload
    assert "decision_payload" not in payload

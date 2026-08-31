"""Integration tests for the P1 champion/challenger lifecycle REST API.

Exercises ``POST /models/{id}/promotion-request`` and
``POST /models/{id}/promote`` against an in-memory SQLite database with an
injected :class:`ModelLifecycleService` (fake MLflow registry, no network). Also
verifies the human gate (401 missing/wrong token, 503 unset token, 422 missing
human header), 404 unknown model/approval, 409 not-yet-approved, 422 invalid
action/entity/payload and arbitrary fields, and idempotent execution via the
receipt. Approval decisions themselves never execute — only the explicit
``promote`` endpoint does.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from maintai.api.lifecycle import build_lifecycle_router
from maintai.application.lifecycle import ModelLifecycleService
from maintai.approvals import ApprovalService
from maintai.audit.repository import AuditRepository
from maintai.audit.service import AuditService
from maintai.db.approval_execution_repository import ApprovalExecutionRepository
from maintai.db.approval_repository import ApprovalRepository
from maintai.db.lifecycle_repository import ModelLifecycleRepository
from maintai.db.model_repository import ModelRepository
from maintai.db.models import (
    MODEL_LIFECYCLE_CHALLENGER,
    MODEL_LIFECYCLE_CHAMPION,
    RegisteredModel,
    new_id,
)
from maintai.mlops.registry import RegistryVersion

TOKEN = "demo-approval-token"


class _FakeRegistry:
    """In-memory stand-in for the MLflow registry (no network in API tests)."""

    def __init__(self) -> None:
        self.promotions: list[tuple[str, str]] = []

    def promote_champion(self, model_name: str, version: str) -> RegistryVersion:
        self.promotions.append((model_name, version))
        return RegistryVersion(
            name=model_name,
            version=version,
            run_id=None,
            model_uri=f"models:/{model_name}/{version}",
            alias="champion",
            status="READY",
        )


def _auth(token: str = TOKEN, actor_id: str = "engineer-1") -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "X-Human-Actor-ID": actor_id,
    }


def _register_model(model_repository: ModelRepository, *, name="failure_model", version="1") -> str:
    model = RegisteredModel(
        id=new_id(),
        name=name,
        version=version,
        mlflow_model_uri=f"models:/{name}/{version}",
        alias="candidate",
        deployment_status="candidate",
        deployed=False,
    )
    model_repository.create(model)
    return model.id


# -- fixtures ----------------------------------------------------------------


@pytest.fixture()
def model_repository(session_factory):
    return ModelRepository(session_factory)


@pytest.fixture()
def approval_service(session_factory):
    return ApprovalService(
        session_factory=session_factory,
        repository=ApprovalRepository(session_factory),
        audit=AuditService(AuditRepository(session_factory)),
    )


@pytest.fixture()
def lifecycle_service(session_factory, model_repository, approval_service):
    return ModelLifecycleService(
        session_factory=session_factory,
        model_repository=model_repository,
        lifecycle_repository=ModelLifecycleRepository(session_factory),
        approval_repository=ApprovalRepository(session_factory),
        execution_repository=ApprovalExecutionRepository(session_factory),
        audit=AuditService(AuditRepository(session_factory)),
        registry=_FakeRegistry(),
        approval_service=approval_service,
    )


@pytest.fixture()
def api_client(lifecycle_service):
    app = FastAPI()
    app.include_router(build_lifecycle_router(lifecycle_service, token=TOKEN))
    with TestClient(app) as client:
        yield client


def _request_url(registered_id: str) -> str:
    return f"/models/{registered_id}/promotion-request"


def _promote_url(registered_id: str) -> str:
    return f"/models/{registered_id}/promote"


def _request_promotion(api_client, registered_id: str):
    return api_client.post(_request_url(registered_id), json={})


# -- promotion request --------------------------------------------------------


def test_promotion_request_returns_approval_and_challenger(
    api_client, model_repository
):
    registered_id = _register_model(model_repository)
    r = _request_promotion(api_client, registered_id)
    assert r.status_code == 201
    body = r.json()
    assert body["registered_model_id"] == registered_id
    assert body["lifecycle_status"] == MODEL_LIFECYCLE_CHALLENGER
    assert body["approval_status"] == "pending"
    assert body["proposed_payload"]["registered_model_id"] == registered_id


def test_promotion_request_unknown_model_returns_404(api_client):
    assert _request_promotion(api_client, "missing").status_code == 404


def test_promotion_request_rejects_arbitrary_fields(api_client, model_repository):
    registered_id = _register_model(model_repository)
    r = api_client.post(
        _request_url(registered_id),
        json={"requested_by_type": "agent", "shell": "rm -rf /"},
    )
    assert r.status_code == 422


# -- promote human gate -------------------------------------------------------


def test_promote_requires_token(api_client, model_repository, approval_service):
    registered_id = _register_model(model_repository)
    request = _request_promotion(api_client, registered_id).json()
    approval_service.decide(
        request["approval_id"], decision="approve", human_actor_id="engineer-1"
    )
    r = api_client.post(_promote_url(registered_id), json={"approval_id": request["approval_id"]})
    assert r.status_code == 401


def test_promote_rejects_wrong_token(api_client, model_repository, approval_service):
    registered_id = _register_model(model_repository)
    request = _request_promotion(api_client, registered_id).json()
    approval_service.decide(
        request["approval_id"], decision="approve", human_actor_id="engineer-1"
    )
    r = api_client.post(
        _promote_url(registered_id),
        json={"approval_id": request["approval_id"]},
        headers=_auth(token="wrong-token"),
    )
    assert r.status_code == 401


def test_promote_unset_token_returns_503(lifecycle_service):
    app = FastAPI()
    app.include_router(build_lifecycle_router(lifecycle_service, token=None))
    with TestClient(app) as client:
        r = client.post(
            _promote_url("any"),
            json={"approval_id": "any"},
            headers=_auth(),
        )
    assert r.status_code == 503


def test_promote_requires_human_header(api_client, model_repository, approval_service):
    registered_id = _register_model(model_repository)
    request = _request_promotion(api_client, registered_id).json()
    approval_service.decide(
        request["approval_id"], decision="approve", human_actor_id="engineer-1"
    )
    r = api_client.post(
        _promote_url(registered_id),
        json={"approval_id": request["approval_id"]},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == 422


# -- promote execution --------------------------------------------------------


def test_promote_unknown_approval_returns_404(api_client, model_repository):
    registered_id = _register_model(model_repository)
    r = api_client.post(
        _promote_url(registered_id),
        json={"approval_id": "missing"},
        headers=_auth(),
    )
    assert r.status_code == 404


def test_promote_requires_approved_returns_409(api_client, model_repository):
    registered_id = _register_model(model_repository)
    request = _request_promotion(api_client, registered_id).json()
    r = api_client.post(
        _promote_url(registered_id),
        json={"approval_id": request["approval_id"]},
        headers=_auth(),
    )
    assert r.status_code == 409


def test_promote_rejects_arbitrary_fields(api_client, model_repository, approval_service):
    registered_id = _register_model(model_repository)
    request = _request_promotion(api_client, registered_id).json()
    approval_service.decide(
        request["approval_id"], decision="approve", human_actor_id="engineer-1"
    )
    r = api_client.post(
        _promote_url(registered_id),
        json={"approval_id": request["approval_id"], "force": True},
        headers=_auth(),
    )
    assert r.status_code == 422


def test_promote_executes_approved(
    api_client, model_repository, approval_service, session_factory
):
    registered_id = _register_model(model_repository)
    request = _request_promotion(api_client, registered_id).json()
    approval_service.decide(
        request["approval_id"], decision="approve", human_actor_id="engineer-1"
    )

    r = api_client.post(
        _promote_url(registered_id),
        json={"approval_id": request["approval_id"]},
        headers=_auth(),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["executed"] is True
    assert body["lifecycle_status"] == MODEL_LIFECYCLE_CHAMPION
    assert body["mlflow_model_uri"] == "models:/failure_model/1"
    assert body["receipt_id"]

    lifecycle = ModelLifecycleRepository(session_factory).get_by_registered_model(registered_id)
    assert lifecycle.status == MODEL_LIFECYCLE_CHAMPION
    receipt = ApprovalExecutionRepository(session_factory).get_by_approval(request["approval_id"])
    assert receipt is not None
    assert receipt.id == body["receipt_id"]


def test_promote_is_idempotent(
    api_client, model_repository, approval_service, session_factory
):
    registered_id = _register_model(model_repository)
    request = _request_promotion(api_client, registered_id).json()
    approval_service.decide(
        request["approval_id"], decision="approve", human_actor_id="engineer-1"
    )

    first = api_client.post(
        _promote_url(registered_id),
        json={"approval_id": request["approval_id"]},
        headers=_auth(),
    ).json()
    second = api_client.post(
        _promote_url(registered_id),
        json={"approval_id": request["approval_id"]},
        headers=_auth(),
    ).json()

    assert first["executed"] is True
    assert second["executed"] is False
    assert second["receipt_id"] == first["receipt_id"]
    assert len(ApprovalExecutionRepository(session_factory).list()) == 1

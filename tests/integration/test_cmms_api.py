"""Integration tests for the P1 mock CMMS REST API (vertical slice).

Exercises ``POST /api/v1/cmms/work-orders/draft`` and the bounded newest-first
list against an in-memory SQLite database: approve/modify effective-payload
resolution, strict action/entity/payload validation, prediction-event
verification, idempotent draft + execution-receipt creation, bearer gating,
mock/disclaimer markers, list bounds, and audit events that never leak the
proposed/decision payloads. No external connector is involved anywhere.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from maintai.api.main import create_app
from maintai.audit.repository import AuditRepository
from maintai.db.approval_execution_repository import ApprovalExecutionRepository
from maintai.db.cmms_repository import MockCMMSWorkOrderRepository
from maintai.db.models import (
    APPROVAL_ACTION_CMMS_WORK_ORDER,
    APPROVAL_ACTION_MODEL_PROMOTION,
    APPROVAL_REQUESTER_AGENT,
    APPROVAL_STATUS_APPROVED,
    APPROVAL_STATUS_MODIFIED,
    APPROVAL_STATUS_PENDING,
    APPROVAL_STATUS_REJECTED,
    CMMS_WORK_ORDER_STATUS_APPROVED,
    ApprovalRequest,
    PredictionEvent,
    new_id,
)

TOKEN = "demo-cmms-token"
DRAFT_URL = "/api/v1/cmms/work-orders/draft"
LIST_URL = "/api/v1/cmms/work-orders"


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def _payload(event_id: str, **overrides) -> dict:
    base = {
        "asset_id": "MOTOR_017",
        "priority": "HIGH",
        "recommended_action": "Inspect drive-end bearing",
        "prediction_event_id": event_id,
    }
    base.update(overrides)
    return base


def _prediction_event(session_factory) -> PredictionEvent:
    event = PredictionEvent(
        id=new_id(),
        model_version="1",
        asset_id="MOTOR_017",
        probability=0.82,
        prediction_json={"prediction": "failure", "confidence": 0.82},
        input_hash="abc123",
    )
    with session_factory.begin() as session:
        session.add(event)
        session.flush()
    return event


def _approval(
    session_factory,
    *,
    status: str,
    proposed_payload: dict | None,
    decision_payload: dict | None = None,
    entity_id: str,
    action_type: str = APPROVAL_ACTION_CMMS_WORK_ORDER,
    entity_type: str = "prediction_event",
) -> str:
    approval = ApprovalRequest(
        id=new_id(),
        action_type=action_type,
        entity_type=entity_type,
        entity_id=entity_id,
        status=status,
        proposed_payload=proposed_payload,
        decision_payload=decision_payload,
        requested_by_type=APPROVAL_REQUESTER_AGENT,
        version=1 if status == APPROVAL_STATUS_PENDING else 2,
        decided_by=None if status == APPROVAL_STATUS_PENDING else "engineer-1",
    )
    with session_factory.begin() as session:
        session.add(approval)
        session.flush()
    return approval.id


@pytest.fixture()
def api_client(session_factory):
    app = create_app(
        session_factory=session_factory,
        approval_token=TOKEN,
        mlflow_healthcheck=lambda: None,
    )
    with TestClient(app) as client:
        yield client


def _draft(client: TestClient, approval_id: str, **headers) -> TestClient:
    return client.post(DRAFT_URL, json={"approval_id": approval_id}, headers=headers)


# -- success / effective payload ---------------------------------------------


def test_draft_approved_returns_201_mock_and_disclaimer(session_factory, api_client):
    event = _prediction_event(session_factory)
    approval_id = _approval(
        session_factory,
        status=APPROVAL_STATUS_APPROVED,
        proposed_payload=_payload(event.id),
        entity_id=event.id,
    )

    r = _draft(api_client, approval_id, **_auth())
    assert r.status_code == 201
    body = r.json()
    assert body["mock"] is True
    assert "no external" in body["disclaimer"]
    assert body["status"] == CMMS_WORK_ORDER_STATUS_APPROVED
    assert body["asset_id"] == "MOTOR_017"
    assert body["priority"] == "HIGH"
    assert body["recommended_action"] == "Inspect drive-end bearing"
    assert body["approval_id"] == approval_id
    assert body["execution_id"]
    assert body["id"]


def test_draft_uses_modified_decision_payload(session_factory, api_client):
    event = _prediction_event(session_factory)
    approval_id = _approval(
        session_factory,
        status=APPROVAL_STATUS_MODIFIED,
        proposed_payload=_payload(event.id, priority="LOW"),
        decision_payload=_payload(event.id, priority="CRITICAL"),
        entity_id=event.id,
    )

    r = _draft(api_client, approval_id, **_auth())
    assert r.status_code == 201
    assert r.json()["priority"] == "CRITICAL"


def test_draft_is_idempotent(session_factory, api_client):
    event = _prediction_event(session_factory)
    approval_id = _approval(
        session_factory,
        status=APPROVAL_STATUS_APPROVED,
        proposed_payload=_payload(event.id),
        entity_id=event.id,
    )

    first = _draft(api_client, approval_id, **_auth())
    assert first.status_code == 201
    second = _draft(api_client, approval_id, **_auth())
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert second.json()["execution_id"] == first.json()["execution_id"]

    assert len(MockCMMSWorkOrderRepository(session_factory).list()) == 1
    assert len(ApprovalExecutionRepository(session_factory).list()) == 1


# -- bearer gate --------------------------------------------------------------


def test_draft_requires_bearer(session_factory, api_client):
    event = _prediction_event(session_factory)
    approval_id = _approval(
        session_factory,
        status=APPROVAL_STATUS_APPROVED,
        proposed_payload=_payload(event.id),
        entity_id=event.id,
    )
    assert _draft(api_client, approval_id).status_code == 401
    assert (
        _draft(api_client, approval_id, **{"Authorization": "Bearer wrong"}).status_code
        == 401
    )


def test_draft_disabled_without_token(session_factory):
    app = create_app(session_factory=session_factory, mlflow_healthcheck=lambda: None)
    with TestClient(app) as client:
        r = client.post(DRAFT_URL, json={"approval_id": "x"})
    assert r.status_code == 503


# -- action / entity / status validation -------------------------------------


def test_draft_rejects_pending(session_factory, api_client):
    event = _prediction_event(session_factory)
    approval_id = _approval(
        session_factory,
        status=APPROVAL_STATUS_PENDING,
        proposed_payload=_payload(event.id),
        entity_id=event.id,
    )
    assert _draft(api_client, approval_id, **_auth()).status_code == 409


def test_draft_rejects_rejected(session_factory, api_client):
    event = _prediction_event(session_factory)
    approval_id = _approval(
        session_factory,
        status=APPROVAL_STATUS_REJECTED,
        proposed_payload=_payload(event.id),
        entity_id=event.id,
    )
    assert _draft(api_client, approval_id, **_auth()).status_code == 409


def test_draft_rejects_wrong_action_type(session_factory, api_client):
    event = _prediction_event(session_factory)
    approval_id = _approval(
        session_factory,
        status=APPROVAL_STATUS_APPROVED,
        proposed_payload=_payload(event.id),
        entity_id=event.id,
        action_type=APPROVAL_ACTION_MODEL_PROMOTION,
    )
    assert _draft(api_client, approval_id, **_auth()).status_code == 422


def test_draft_rejects_wrong_entity_type(session_factory, api_client):
    event = _prediction_event(session_factory)
    approval_id = _approval(
        session_factory,
        status=APPROVAL_STATUS_APPROVED,
        proposed_payload=_payload(event.id),
        entity_id=event.id,
        entity_type="asset",
    )
    assert _draft(api_client, approval_id, **_auth()).status_code == 422


def test_draft_rejects_missing_required_payload_field(session_factory, api_client):
    event = _prediction_event(session_factory)
    approval_id = _approval(
        session_factory,
        status=APPROVAL_STATUS_APPROVED,
        proposed_payload={"asset_id": "MOTOR_017", "priority": "HIGH"},
        entity_id=event.id,
    )
    assert _draft(api_client, approval_id, **_auth()).status_code == 422


def test_draft_rejects_unknown_payload_field(session_factory, api_client):
    event = _prediction_event(session_factory)
    approval_id = _approval(
        session_factory,
        status=APPROVAL_STATUS_APPROVED,
        proposed_payload=_payload(event.id, shell="rm -rf /"),
        entity_id=event.id,
    )
    assert _draft(api_client, approval_id, **_auth()).status_code == 422


def test_draft_rejects_mismatched_prediction_event_id(session_factory, api_client):
    event = _prediction_event(session_factory)
    approval_id = _approval(
        session_factory,
        status=APPROVAL_STATUS_APPROVED,
        proposed_payload=_payload(event.id),
        entity_id=new_id(),
    )
    assert _draft(api_client, approval_id, **_auth()).status_code == 422


def test_draft_rejects_missing_prediction_event(session_factory, api_client):
    event_id = new_id()
    approval_id = _approval(
        session_factory,
        status=APPROVAL_STATUS_APPROVED,
        proposed_payload=_payload(event_id),
        entity_id=event_id,
    )
    assert _draft(api_client, approval_id, **_auth()).status_code == 404


def test_draft_rejects_unknown_approval(session_factory, api_client):
    assert _draft(api_client, "missing", **_auth()).status_code == 404


def test_draft_rejects_arbitrary_body_field(session_factory, api_client):
    r = api_client.post(
        DRAFT_URL,
        json={"approval_id": "x", "shell": "rm -rf /"},
        headers=_auth(),
    )
    assert r.status_code == 422


# -- list ---------------------------------------------------------------------


def test_list_bounded_newest_first(session_factory, api_client):
    first_event = _prediction_event(session_factory)
    first_approval = _approval(
        session_factory,
        status=APPROVAL_STATUS_APPROVED,
        proposed_payload=_payload(first_event.id),
        entity_id=first_event.id,
    )
    second_event = _prediction_event(session_factory)
    second_approval = _approval(
        session_factory,
        status=APPROVAL_STATUS_APPROVED,
        proposed_payload=_payload(second_event.id),
        entity_id=second_event.id,
    )

    first = _draft(api_client, first_approval, **_auth()).json()
    second = _draft(api_client, second_approval, **_auth()).json()

    listing = api_client.get(LIST_URL)
    assert listing.status_code == 200
    items = listing.json()
    assert [item["id"] for item in items] == [second["id"], first["id"]]
    assert api_client.get(LIST_URL, params={"limit": 1}).json()[0]["id"] == second["id"]


# -- audit hygiene ------------------------------------------------------------


def test_draft_audit_never_leaks_payloads(session_factory, api_client):
    event = _prediction_event(session_factory)
    approval_id = _approval(
        session_factory,
        status=APPROVAL_STATUS_APPROVED,
        proposed_payload=_payload(event.id, reason="topsecret-reason"),
        entity_id=event.id,
    )
    _draft(api_client, approval_id, **_auth())

    events = AuditRepository(session_factory).list(
        entity_type="prediction_event", entity_id=event.id
    )
    draft_events = [e for e in events if e.action == "cmms.work_order_draft"]
    assert len(draft_events) == 1
    for event_row in events:
        payload = event_row.payload_json or {}
        assert "proposed_payload" not in payload
        assert "decision_payload" not in payload
        assert "topsecret" not in str(payload)
        assert "MOTOR_017" not in str(payload)
        assert "Inspect drive-end bearing" not in str(payload)
        assert TOKEN not in str(payload)

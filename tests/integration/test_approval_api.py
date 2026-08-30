"""Integration tests for the P1 approval REST API (human-decision gate).

Exercises ``POST /api/v1/approvals`` (propose), list redaction, full detail,
and the token-gated ``approve``/``reject``/``modify`` decisions against an
in-memory SQLite database with an injected :class:`ApprovalService` and a mocked
bearer token (no network). Also verifies: 401 missing/invalid token, 503 unset
token, 422 missing human header, 409 stale/double decisions, 422 invalid action,
that an agent cannot override the actor type, request-id propagation, and that
audit events never leak payloads or the token.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update

from maintai.api.approvals import build_approval_service
from maintai.api.main import create_app
from maintai.audit.repository import AuditRepository
from maintai.config import Settings
from maintai.db.models import (
    APPROVAL_ACTION_MODEL_PROMOTION,
    APPROVAL_STATUS_APPROVED,
    APPROVAL_STATUS_MODIFIED,
    APPROVAL_STATUS_PENDING,
    APPROVAL_STATUS_REJECTED,
    ApprovalRequest,
)

TOKEN = "demo-approval-token"
APPROVALS_URL = "/api/v1/approvals"


def _auth(token: str = TOKEN, actor_id: str = "engineer-1") -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "X-Human-Actor-ID": actor_id,
    }


def _propose(client: TestClient, **overrides) -> TestClient:
    body = {
        "action_type": APPROVAL_ACTION_MODEL_PROMOTION,
        "entity_type": "registered_model",
        "entity_id": "model-1",
        "requested_by_type": "agent",
        "proposed_payload": {"from": "challenger", "to": "champion"},
    }
    body.update(overrides)
    return client.post(APPROVALS_URL, json=body)


@pytest.fixture()
def api_client(session_factory):
    service = build_approval_service(session_factory)
    app = create_app(
        session_factory=session_factory,
        approval_service=service,
        approval_token=TOKEN,
        mlflow_healthcheck=lambda: None,
    )
    with TestClient(app) as client:
        yield client


def _decision_url(approval_id: str, verb: str) -> str:
    return f"{APPROVALS_URL}/{approval_id}/{verb}"


# -- propose / list / get ----------------------------------------------------


def test_propose_returns_pending_detail(api_client):
    r = _propose(api_client)
    assert r.status_code == 201
    body = r.json()
    assert body["action_type"] == APPROVAL_ACTION_MODEL_PROMOTION
    assert body["status"] == APPROVAL_STATUS_PENDING
    assert body["version"] == 1
    assert body["proposed_payload"] == {"from": "challenger", "to": "champion"}
    assert body["decision_payload"] is None


def test_list_returns_redacted_summaries(api_client):
    created = _propose(api_client).json()
    r = api_client.get(APPROVALS_URL)
    assert r.status_code == 200
    items = r.json()
    assert [item["id"] for item in items] == [created["id"]]
    assert "proposed_payload" not in items[0]
    assert "decision_payload" not in items[0]


def test_get_returns_full_detail(api_client):
    created = _propose(api_client).json()
    r = api_client.get(f"{APPROVALS_URL}/{created['id']}", headers=_auth())
    assert r.status_code == 200
    body = r.json()
    assert body["proposed_payload"] == {"from": "challenger", "to": "champion"}
    assert body["decision_payload"] is None


def test_get_unknown_returns_404(api_client):
    assert api_client.get(f"{APPROVALS_URL}/missing", headers=_auth()).status_code == 404


def test_get_detail_requires_token(api_client):
    created = _propose(api_client).json()
    assert api_client.get(f"{APPROVALS_URL}/{created['id']}").status_code == 401


def test_propose_invalid_action_returns_422(api_client):
    assert _propose(api_client, action_type="bogus").status_code == 422


def test_propose_invalid_requester_returns_422(api_client):
    assert _propose(api_client, requested_by_type="bogus").status_code == 422


def test_propose_rejects_arbitrary_fields(api_client):
    r = api_client.post(
        APPROVALS_URL,
        json={
            "action_type": APPROVAL_ACTION_MODEL_PROMOTION,
            "entity_type": "registered_model",
            "entity_id": "model-1",
            "requested_by_type": "agent",
            "shell": "rm -rf /",
        },
    )
    assert r.status_code == 422


# -- decision gate (token + human header) ------------------------------------


def test_decision_requires_token(api_client):
    created = _propose(api_client).json()
    r = api_client.post(
        _decision_url(created["id"], "approve"), json={"expected_version": 1}
    )
    assert r.status_code == 401


def test_decision_rejects_wrong_token(api_client):
    created = _propose(api_client).json()
    r = api_client.post(
        _decision_url(created["id"], "approve"),
        json={"expected_version": 1},
        headers=_auth(token="wrong-token"),
    )
    assert r.status_code == 401


def test_decision_requires_human_actor_header(api_client):
    created = _propose(api_client).json()
    r = api_client.post(
        _decision_url(created["id"], "approve"),
        json={"expected_version": 1},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == 422


def test_decision_requires_expected_version(api_client):
    created = _propose(api_client).json()
    r = api_client.post(
        _decision_url(created["id"], "approve"), json={}, headers=_auth()
    )
    assert r.status_code == 422


def test_decision_rejects_unset_token_with_503(session_factory, monkeypatch):
    monkeypatch.delenv("APPROVAL_API_TOKEN", raising=False)
    service = build_approval_service(session_factory)
    app = create_app(
        session_factory=session_factory,
        approval_service=service,
        settings=Settings(_env_file=None),
        mlflow_healthcheck=lambda: None,
    )
    with TestClient(app) as client:
        created = _propose(client).json()
        r = client.post(
            _decision_url(created["id"], "approve"),
            json={"expected_version": 1},
            headers=_auth(),
        )
    assert r.status_code == 503


# -- approve / reject / modify ------------------------------------------------


def test_approve_with_correct_token(api_client):
    created = _propose(api_client).json()
    r = api_client.post(
        _decision_url(created["id"], "approve"),
        json={"expected_version": 1},
        headers=_auth(),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == APPROVAL_STATUS_APPROVED
    assert body["version"] == 2
    assert body["decided_by"] == "engineer-1"
    assert body["reason"] is None


def test_reject_requires_and_stores_reason(api_client):
    created = _propose(api_client).json()
    missing = api_client.post(
        _decision_url(created["id"], "reject"),
        json={"expected_version": 1},
        headers=_auth(),
    )
    assert missing.status_code == 422

    ok = api_client.post(
        _decision_url(created["id"], "reject"),
        json={"expected_version": 1, "reason": "not enough evidence"},
        headers=_auth(),
    )
    assert ok.status_code == 200
    body = ok.json()
    assert body["status"] == APPROVAL_STATUS_REJECTED
    assert body["reason"] == "not enough evidence"


def test_modify_requires_and_stores_reason_and_payload(api_client):
    created = _propose(api_client).json()

    no_reason = api_client.post(
        _decision_url(created["id"], "modify"),
        json={"expected_version": 1, "payload": {"to": "previous_champion"}},
        headers=_auth(),
    )
    assert no_reason.status_code == 422

    no_payload = api_client.post(
        _decision_url(created["id"], "modify"),
        json={"expected_version": 1, "reason": "adjust"},
        headers=_auth(),
    )
    assert no_payload.status_code == 422

    ok = api_client.post(
        _decision_url(created["id"], "modify"),
        json={
            "expected_version": 1,
            "reason": "change the target alias",
            "payload": {"from": "challenger", "to": "previous_champion"},
        },
        headers=_auth(),
    )
    assert ok.status_code == 200
    body = ok.json()
    assert body["status"] == APPROVAL_STATUS_MODIFIED
    assert body["decision_payload"] == {"from": "challenger", "to": "previous_champion"}


# -- conflicts -----------------------------------------------------------------


def test_double_decide_is_conflict(api_client):
    created = _propose(api_client).json()
    assert (
        api_client.post(
            _decision_url(created["id"], "approve"),
            json={"expected_version": 1},
            headers=_auth(),
        ).status_code
        == 200
    )
    r = api_client.post(
        _decision_url(created["id"], "reject"),
        json={"expected_version": 2, "reason": "changed my mind"},
        headers=_auth(),
    )
    assert r.status_code == 409


def test_stale_version_is_conflict(api_client, session_factory):
    created = _propose(api_client).json()
    with session_factory.begin() as session:
        session.execute(
            update(ApprovalRequest)
            .where(ApprovalRequest.id == created["id"])
            .values(version=2)
        )
    r = api_client.post(
        _decision_url(created["id"], "approve"),
        json={"expected_version": 1},
        headers=_auth(),
    )
    assert r.status_code == 409


# -- actor-type / impersonation ------------------------------------------------


def test_agent_cannot_supply_actor_type(api_client):
    created = _propose(api_client).json()
    r = api_client.post(
        _decision_url(created["id"], "approve"),
        json={"expected_version": 1, "human_actor_type": "agent"},
        headers=_auth(actor_id="agent-1"),
    )
    assert r.status_code == 422


def test_decision_audit_is_user_and_never_leaks(api_client, session_factory):
    created = _propose(api_client).json()
    api_client.post(
        _decision_url(created["id"], "modify"),
        json={
            "expected_version": 1,
            "reason": "adjust",
            "payload": {"secret": "topsecret-decision"},
        },
        headers=_auth(actor_id="agent-impersonator"),
    )

    audit = AuditRepository(session_factory)
    events = audit.list(entity_type="registered_model", entity_id="model-1")
    decide_events = [e for e in events if e.action == "approval.modify"]
    assert len(decide_events) == 1
    # The API always records the deciding principal as a human "user", never the
    # self-asserted value, and never the bearer token or payloads.
    assert decide_events[0].actor_type == "user"
    assert decide_events[0].actor_id == "agent-impersonator"
    for event in events:
        payload = event.payload_json or {}
        assert "topsecret" not in str(payload)
        assert "proposed_payload" not in payload
        assert "decision_payload" not in payload
        assert TOKEN not in str(payload)


# -- request-id ----------------------------------------------------------------


def test_request_id_echoed(api_client):
    r = _propose(api_client)
    assert r.headers.get("x-request-id")

    r2 = api_client.post(
        APPROVALS_URL,
        json={
            "action_type": APPROVAL_ACTION_MODEL_PROMOTION,
            "entity_type": "registered_model",
            "entity_id": "model-2",
            "requested_by_type": "agent",
        },
        headers={"X-Request-ID": "approval-req-1"},
    )
    assert r2.status_code == 201
    assert r2.headers["x-request-id"] == "approval-req-1"

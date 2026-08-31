"""Integration tests for the P1 technician-feedback REST API.

Exercises ``POST`` and ``GET`` under
``/api/v1/predictions/{prediction_id}/feedback`` against an in-memory SQLite
database. Covers: 201 append + echo, the ``X-Human-Actor-ID`` gate (missing or
blank header → 422, no bearer token required), prediction existence (404),
outcome/comment validation (422), arbitrary-field rejection, list ordering,
request-id propagation, and that audit events never contain the comment content.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from maintai.api.main import create_app
from maintai.audit.repository import AuditRepository
from maintai.db.models import (
    FEEDBACK_OUTCOME_CONFIRMED_ISSUE,
    FEEDBACK_OUTCOME_FALSE_ALARM,
    PredictionEvent,
    new_id,
)

FEEDBACK_URL = "/api/v1/predictions/{prediction_id}/feedback"


def _prediction(session_factory) -> PredictionEvent:
    event = PredictionEvent(id=new_id(), model_version="1")
    with session_factory.begin() as session:
        session.add(event)
        session.flush()
        return event


@pytest.fixture()
def api_client(session_factory):
    app = create_app(session_factory=session_factory, mlflow_healthcheck=lambda: None)
    with TestClient(app) as client:
        yield client


def _post(client: TestClient, prediction_id: str, *, actor_id="engineer-1", **body):
    return client.post(
        FEEDBACK_URL.format(prediction_id=prediction_id),
        json=body,
        headers={"X-Human-Actor-ID": actor_id},
    )


# -- append / echo ------------------------------------------------------------


def test_post_returns_201_and_echoes(session_factory, api_client):
    prediction = _prediction(session_factory)
    r = _post(
        api_client,
        prediction.id,
        outcome=FEEDBACK_OUTCOME_CONFIRMED_ISSUE,
        comment="bearing overheating",
    )
    assert r.status_code == 201
    body = r.json()
    assert body["id"]
    assert body["prediction_event_id"] == prediction.id
    assert body["outcome"] == FEEDBACK_OUTCOME_CONFIRMED_ISSUE
    assert body["comment"] == "bearing overheating"
    assert body["technician_id"] == "engineer-1"
    assert body["created_at"]


def test_post_requires_no_bearer_token(session_factory, api_client):
    # No Authorization header at all — feedback identity is the human header only.
    prediction = _prediction(session_factory)
    r = api_client.post(
        FEEDBACK_URL.format(prediction_id=prediction.id),
        json={"outcome": FEEDBACK_OUTCOME_FALSE_ALARM},
        headers={"X-Human-Actor-ID": "engineer-1"},
    )
    assert r.status_code == 201


# -- human actor header -------------------------------------------------------


def test_post_requires_human_actor_header(session_factory, api_client):
    prediction = _prediction(session_factory)
    r = api_client.post(
        FEEDBACK_URL.format(prediction_id=prediction.id),
        json={"outcome": FEEDBACK_OUTCOME_FALSE_ALARM},
    )
    assert r.status_code == 422


def test_post_requires_nonempty_human_actor_header(session_factory, api_client):
    prediction = _prediction(session_factory)
    r = api_client.post(
        FEEDBACK_URL.format(prediction_id=prediction.id),
        json={"outcome": FEEDBACK_OUTCOME_FALSE_ALARM},
        headers={"X-Human-Actor-ID": "   "},
    )
    assert r.status_code == 422


# -- validation ---------------------------------------------------------------


def test_post_unknown_prediction_returns_404(session_factory, api_client):
    r = _post(api_client, "missing", outcome=FEEDBACK_OUTCOME_FALSE_ALARM)
    assert r.status_code == 404


def test_post_invalid_outcome_returns_422(session_factory, api_client):
    prediction = _prediction(session_factory)
    r = _post(api_client, prediction.id, outcome="bogus")
    assert r.status_code == 422


def test_post_comment_too_long_returns_422(session_factory, api_client):
    prediction = _prediction(session_factory)
    r = _post(
        api_client,
        prediction.id,
        outcome=FEEDBACK_OUTCOME_FALSE_ALARM,
        comment="x" * 2049,
    )
    assert r.status_code == 422


def test_post_rejects_arbitrary_fields(session_factory, api_client):
    prediction = _prediction(session_factory)
    r = api_client.post(
        FEEDBACK_URL.format(prediction_id=prediction.id),
        json={"outcome": FEEDBACK_OUTCOME_FALSE_ALARM, "shell": "rm -rf /"},
        headers={"X-Human-Actor-ID": "engineer-1"},
    )
    assert r.status_code == 422


# -- list ---------------------------------------------------------------------


def test_get_lists_feedback_newest_first(session_factory, api_client):
    prediction = _prediction(session_factory)
    first = _post(api_client, prediction.id, outcome=FEEDBACK_OUTCOME_FALSE_ALARM).json()
    second = _post(
        api_client, prediction.id, outcome=FEEDBACK_OUTCOME_CONFIRMED_ISSUE
    ).json()

    r = api_client.get(FEEDBACK_URL.format(prediction_id=prediction.id))
    assert r.status_code == 200
    assert [item["id"] for item in r.json()] == [second["id"], first["id"]]


def test_get_unknown_prediction_returns_empty(session_factory, api_client):
    r = api_client.get(FEEDBACK_URL.format(prediction_id="missing"))
    assert r.status_code == 200
    assert r.json() == []


# -- audit / request id -------------------------------------------------------


def test_audit_never_contains_comment(session_factory, api_client):
    prediction = _prediction(session_factory)
    _post(
        api_client,
        prediction.id,
        outcome=FEEDBACK_OUTCOME_CONFIRMED_ISSUE,
        comment="topsecret-comment",
    )
    events = AuditRepository(session_factory).list(
        entity_type="prediction_event", entity_id=prediction.id
    )
    assert len(events) == 1
    assert events[0].actor_type == "user"
    assert events[0].actor_id == "engineer-1"
    payload = events[0].payload_json or {}
    assert "comment" not in payload
    assert "topsecret-comment" not in str(payload)


def test_request_id_echoed(session_factory, api_client):
    prediction = _prediction(session_factory)
    r = api_client.post(
        FEEDBACK_URL.format(prediction_id=prediction.id),
        json={"outcome": FEEDBACK_OUTCOME_FALSE_ALARM},
        headers={"X-Human-Actor-ID": "engineer-1", "X-Request-ID": "feedback-req-1"},
    )
    assert r.status_code == 201
    assert r.headers["x-request-id"] == "feedback-req-1"

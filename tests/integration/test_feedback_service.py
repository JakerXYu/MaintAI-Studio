"""Integration tests for the P1 technician-feedback service.

Exercises the append-only feedback service against an in-memory SQLite database:
the four canonical outcomes, prediction existence verification, bounded comment
and technician-id validation, ``registered_model_id`` inheritance from the
prediction, atomic audit writes that never contain the comment content, and the
append-only guarantee (no update/delete path on the repository).
"""

from __future__ import annotations

import pytest

from maintai.audit.repository import AuditRepository
from maintai.audit.service import AuditService
from maintai.db.feedback_repository import TechnicianFeedbackRepository
from maintai.db.models import (
    FEEDBACK_OUTCOME_CONFIRMED_ISSUE,
    FEEDBACK_OUTCOME_DIFFERENT_ISSUE,
    FEEDBACK_OUTCOME_FALSE_ALARM,
    FEEDBACK_OUTCOME_NO_ACTION_NEEDED,
    PredictionEvent,
    RegisteredModel,
    new_id,
)
from maintai.feedback import (
    FeedbackPredictionNotFoundError,
    FeedbackService,
    FeedbackValidationError,
)

OUTCOMES = (
    FEEDBACK_OUTCOME_CONFIRMED_ISSUE,
    FEEDBACK_OUTCOME_FALSE_ALARM,
    FEEDBACK_OUTCOME_DIFFERENT_ISSUE,
    FEEDBACK_OUTCOME_NO_ACTION_NEEDED,
)


class _FailingAudit:
    """Audit double whose writes always fail (used to force rollback)."""

    def record(self, **kwargs):
        raise RuntimeError("audit unavailable")


# -- fixtures ----------------------------------------------------------------


def make_service(session_factory, audit=None):
    return FeedbackService(
        session_factory=session_factory,
        repository=TechnicianFeedbackRepository(session_factory),
        audit=audit if audit is not None else AuditService(AuditRepository(session_factory)),
    )


def _prediction(session_factory, *, registered_model_id=None) -> PredictionEvent:
    event = PredictionEvent(
        id=new_id(), model_version="1", registered_model_id=registered_model_id
    )
    with session_factory.begin() as session:
        session.add(event)
        session.flush()
        return event


# -- outcomes -----------------------------------------------------------------


def test_create_accepts_all_four_outcomes(session_factory):
    service = make_service(session_factory)
    repository = TechnicianFeedbackRepository(session_factory)
    prediction = _prediction(session_factory)

    for outcome in OUTCOMES:
        created = service.create(
            prediction.id, outcome=outcome, technician_id="engineer-1"
        )
        assert created["outcome"] == outcome
        assert created["prediction_event_id"] == prediction.id
        persisted = repository.get(created["id"])
        assert persisted.outcome == outcome


def test_create_rejects_invalid_outcome(session_factory):
    service = make_service(session_factory)
    prediction = _prediction(session_factory)
    with pytest.raises(FeedbackValidationError, match="outcome"):
        service.create(prediction.id, outcome="bogus", technician_id="engineer-1")


# -- prediction existence -----------------------------------------------------


def test_create_verifies_prediction_exists(session_factory):
    service = make_service(session_factory)
    repository = TechnicianFeedbackRepository(session_factory)
    with pytest.raises(FeedbackPredictionNotFoundError, match="not found"):
        service.create("missing", outcome=FEEDBACK_OUTCOME_FALSE_ALARM, technician_id="t1")
    assert repository.list(prediction_event_id="missing") == []


def test_create_inherits_registered_model_id(session_factory):
    service = make_service(session_factory)
    registered = RegisteredModel(id=new_id(), name="demo", version="1")
    with session_factory.begin() as session:
        session.add(registered)
        session.flush()
    prediction = _prediction(session_factory, registered_model_id=registered.id)

    created = service.create(
        prediction.id, outcome=FEEDBACK_OUTCOME_CONFIRMED_ISSUE, technician_id="t1"
    )
    assert created["registered_model_id"] == registered.id


# -- technician id / comment --------------------------------------------------


def test_create_requires_technician_id(session_factory):
    service = make_service(session_factory)
    prediction = _prediction(session_factory)
    with pytest.raises(FeedbackValidationError, match="technician_id"):
        service.create(prediction.id, outcome=FEEDBACK_OUTCOME_FALSE_ALARM, technician_id="")
    with pytest.raises(FeedbackValidationError, match="technician_id"):
        service.create(
            prediction.id, outcome=FEEDBACK_OUTCOME_FALSE_ALARM, technician_id="   "
        )
    with pytest.raises(FeedbackValidationError, match="technician_id"):
        service.create(
            prediction.id,
            outcome=FEEDBACK_OUTCOME_FALSE_ALARM,
            technician_id="x" * 65,
        )


def test_create_bounds_comment(session_factory):
    service = make_service(session_factory)
    prediction = _prediction(session_factory)
    with pytest.raises(FeedbackValidationError, match="comment"):
        service.create(
            prediction.id,
            outcome=FEEDBACK_OUTCOME_FALSE_ALARM,
            technician_id="t1",
            comment="x" * 2049,
        )


def test_create_stores_comment_and_technician(session_factory):
    service = make_service(session_factory)
    prediction = _prediction(session_factory)
    created = service.create(
        prediction.id,
        outcome=FEEDBACK_OUTCOME_DIFFERENT_ISSUE,
        technician_id="engineer-9",
        comment="wrong bearing",
    )
    assert created["comment"] == "wrong bearing"
    assert created["technician_id"] == "engineer-9"
    assert created["created_at"] is not None


# -- audit / append-only ------------------------------------------------------


def test_create_audit_never_contains_comment(session_factory):
    service = make_service(session_factory)
    audit_repository = AuditRepository(session_factory)
    prediction = _prediction(session_factory)

    service.create(
        prediction.id,
        outcome=FEEDBACK_OUTCOME_CONFIRMED_ISSUE,
        technician_id="engineer-1",
        comment="topsecret-comment",
    )
    events = audit_repository.list(entity_type="prediction_event", entity_id=prediction.id)
    assert len(events) == 1
    assert events[0].action == "feedback.create"
    assert events[0].actor_type == "user"
    assert events[0].actor_id == "engineer-1"
    payload = events[0].payload_json or {}
    assert payload["outcome"] == FEEDBACK_OUTCOME_CONFIRMED_ISSUE
    assert "comment" not in payload
    assert "topsecret-comment" not in str(payload)


def test_create_rolls_back_when_audit_fails(session_factory):
    repository = TechnicianFeedbackRepository(session_factory)
    prediction = _prediction(session_factory)
    failing = make_service(session_factory, audit=_FailingAudit())
    with pytest.raises(RuntimeError, match="audit unavailable"):
        failing.create(
            prediction.id,
            outcome=FEEDBACK_OUTCOME_FALSE_ALARM,
            technician_id="engineer-1",
        )
    assert repository.list(prediction_event_id=prediction.id) == []


def test_repository_is_append_only(session_factory):
    repository = TechnicianFeedbackRepository(session_factory)
    assert not hasattr(repository, "update")
    assert not hasattr(repository, "delete")


# -- list ---------------------------------------------------------------------


def test_list_returns_newest_first(session_factory):
    service = make_service(session_factory)
    prediction = _prediction(session_factory)
    first = service.create(
        prediction.id, outcome=FEEDBACK_OUTCOME_FALSE_ALARM, technician_id="t1"
    )
    second = service.create(
        prediction.id, outcome=FEEDBACK_OUTCOME_CONFIRMED_ISSUE, technician_id="t2"
    )

    listing = service.list(prediction.id)
    assert [item["id"] for item in listing] == [second["id"], first["id"]]

    other = _prediction(session_factory)
    assert service.list(other.id) == []

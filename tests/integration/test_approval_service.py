"""Integration tests for the P1 approval service + repository.

Exercises the human-in-the-loop approval state machine against an in-memory
SQLite database (``StaticPool`` test double): the four gated action types,
``approve``/``reject``/``modify`` decisions, strict field validation, duplicate
and stale compare-and-swap conflicts, audit-failure rollback, list filters,
database check constraints, and the guarantee that proposing/deciding never
touches the model registry (no side effects).
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from maintai.approvals import (
    ApprovalConflictError,
    ApprovalNotFoundError,
    ApprovalService,
    ApprovalValidationError,
)
from maintai.audit.repository import AuditRepository
from maintai.audit.service import AuditService
from maintai.db.approval_repository import ApprovalRepository
from maintai.db.models import (
    APPROVAL_ACTION_CMMS_WORK_ORDER,
    APPROVAL_ACTION_MAINTENANCE_ACTION,
    APPROVAL_ACTION_MODEL_PROMOTION,
    APPROVAL_ACTION_RETRAINING_DEPLOYMENT,
    APPROVAL_REQUESTER_AGENT,
    APPROVAL_REQUESTER_SYSTEM,
    APPROVAL_REQUESTER_USER,
    APPROVAL_STATUS_APPROVED,
    APPROVAL_STATUS_MODIFIED,
    APPROVAL_STATUS_PENDING,
    APPROVAL_STATUS_REJECTED,
    ApprovalRequest,
    RegisteredModel,
    new_id,
)

ACTIONS = (
    APPROVAL_ACTION_MODEL_PROMOTION,
    APPROVAL_ACTION_RETRAINING_DEPLOYMENT,
    APPROVAL_ACTION_MAINTENANCE_ACTION,
    APPROVAL_ACTION_CMMS_WORK_ORDER,
)

REQUESTERS = (
    APPROVAL_REQUESTER_AGENT,
    APPROVAL_REQUESTER_USER,
    APPROVAL_REQUESTER_SYSTEM,
)


class _FailingAudit:
    """Audit double whose writes always fail (used to force rollback)."""

    def record(self, **kwargs):
        raise RuntimeError("audit unavailable")


# -- fixtures ----------------------------------------------------------------


def make_service(session_factory, audit=None):
    return ApprovalService(
        session_factory=session_factory,
        repository=ApprovalRepository(session_factory),
        audit=audit if audit is not None else AuditService(AuditRepository(session_factory)),
    )


def _propose(service, *, action_type=APPROVAL_ACTION_MODEL_PROMOTION, **kwargs):
    defaults = {
        "action_type": action_type,
        "entity_type": "registered_model",
        "entity_id": "model-1",
        "requested_by_type": APPROVAL_REQUESTER_AGENT,
        "proposed_payload": {"from": "challenger", "to": "champion"},
    }
    defaults.update(kwargs)
    return service.propose(**defaults)


# -- action types / proposers -------------------------------------------------


def test_propose_supports_all_four_action_types(session_factory):
    service = make_service(session_factory)
    repository = ApprovalRepository(session_factory)
    for action_type in ACTIONS:
        created = _propose(service, action_type=action_type)
        assert created["action_type"] == action_type
        assert created["status"] == APPROVAL_STATUS_PENDING
        assert created["version"] == 1
        assert created["decided_at"] is None
        assert created["decided_by"] is None
        persisted = repository.get(created["id"])
        assert persisted.action_type == action_type
        assert persisted.status == APPROVAL_STATUS_PENDING


def test_propose_accepts_all_requester_types(session_factory):
    service = make_service(session_factory)
    for index, requester in enumerate(REQUESTERS):
        created = _propose(
            service,
            requested_by_type=requester,
            entity_id=f"model-{index}",
        )
        assert created["requested_by_type"] == requester
        assert created["requested_by_id"] is None


def test_propose_records_audit_without_payload(session_factory):
    service = make_service(session_factory)
    audit_repository = AuditRepository(session_factory)
    created = _propose(
        service,
        proposed_payload={"secret": "topsecret-proposal"},
    )
    events = audit_repository.list(
        entity_type="registered_model", entity_id="model-1"
    )
    propose_events = [e for e in events if e.action == "approval.propose"]
    assert len(propose_events) == 1
    payload = propose_events[0].payload_json or {}
    assert payload["approval_id"] == created["id"]
    assert payload["status"] == APPROVAL_STATUS_PENDING
    assert "proposed_payload" not in payload
    assert "secret" not in str(payload)


# -- decisions ---------------------------------------------------------------


def test_approve_transitions_and_audits(session_factory):
    service = make_service(session_factory)
    audit_repository = AuditRepository(session_factory)
    created = _propose(service)
    decided = service.decide(
        created["id"], decision="approve", human_actor_id="engineer-1"
    )
    assert decided["status"] == APPROVAL_STATUS_APPROVED
    assert decided["version"] == 2
    assert decided["decided_by"] == "engineer-1"
    assert decided["decided_at"] is not None
    assert decided["reason"] is None
    assert decided["decision_payload"] is None

    events = audit_repository.list(entity_type="registered_model", entity_id="model-1")
    actions = [e.action for e in events]
    assert "approval.propose" in actions
    assert "approval.approve" in actions


def test_reject_requires_and_stores_reason(session_factory):
    service = make_service(session_factory)
    created = _propose(service)
    decided = service.decide(
        created["id"],
        decision="reject",
        human_actor_id="engineer-1",
        reason="not enough evidence",
    )
    assert decided["status"] == APPROVAL_STATUS_REJECTED
    assert decided["reason"] == "not enough evidence"
    assert decided["decision_payload"] is None


def test_modify_requires_and_stores_reason_and_payload(session_factory):
    service = make_service(session_factory)
    created = _propose(service)
    decided = service.decide(
        created["id"],
        decision="modify",
        human_actor_id="engineer-1",
        reason="change the target alias",
        payload={"from": "challenger", "to": "previous_champion"},
    )
    assert decided["status"] == APPROVAL_STATUS_MODIFIED
    assert decided["reason"] == "change the target alias"
    assert decided["decision_payload"] == {
        "from": "challenger",
        "to": "previous_champion",
    }


def test_decision_audit_never_leaks_payloads(session_factory):
    service = make_service(session_factory)
    audit_repository = AuditRepository(session_factory)
    created = _propose(
        service,
        proposed_payload={"secret": "topsecret-proposal"},
    )
    service.decide(
        created["id"],
        decision="modify",
        human_actor_id="engineer-1",
        reason="adjust",
        payload={"secret": "topsecret-decision"},
    )
    events = audit_repository.list(entity_type="registered_model", entity_id="model-1")
    for event in events:
        payload = event.payload_json or {}
        assert "proposed_payload" not in payload
        assert "decision_payload" not in payload
        assert "topsecret" not in str(payload)


# -- validation / not-found --------------------------------------------------


def test_propose_field_validation(session_factory):
    service = make_service(session_factory)
    with pytest.raises(ApprovalValidationError, match="action_type"):
        service.propose(
            action_type="bogus",
            entity_type="registered_model",
            entity_id="model-1",
            requested_by_type=APPROVAL_REQUESTER_AGENT,
        )
    with pytest.raises(ApprovalValidationError, match="requested_by_type"):
        service.propose(
            action_type=APPROVAL_ACTION_MODEL_PROMOTION,
            entity_type="registered_model",
            entity_id="model-1",
            requested_by_type="bogus",
        )
    with pytest.raises(ApprovalValidationError, match="entity_type"):
        service.propose(
            action_type=APPROVAL_ACTION_MODEL_PROMOTION,
            entity_type="",
            entity_id="model-1",
            requested_by_type=APPROVAL_REQUESTER_AGENT,
        )
    with pytest.raises(ApprovalValidationError, match="entity_id"):
        service.propose(
            action_type=APPROVAL_ACTION_MODEL_PROMOTION,
            entity_type="registered_model",
            entity_id="",
            requested_by_type=APPROVAL_REQUESTER_AGENT,
        )


def test_decide_field_validation(session_factory):
    service = make_service(session_factory)
    created = _propose(service)
    with pytest.raises(ApprovalValidationError, match="human_actor_id"):
        service.decide(created["id"], decision="approve", human_actor_id="")
    with pytest.raises(ApprovalValidationError, match="human user"):
        service.decide(
            created["id"],
            decision="approve",
            human_actor_id="agent-1",
            human_actor_type="agent",
        )
    with pytest.raises(ApprovalValidationError, match="decision"):
        service.decide(created["id"], decision="maybe", human_actor_id="engineer-1")
    with pytest.raises(ApprovalValidationError, match="reason"):
        service.decide(created["id"], decision="reject", human_actor_id="engineer-1")
    with pytest.raises(ApprovalValidationError, match="reason"):
        service.decide(
            created["id"],
            decision="modify",
            human_actor_id="engineer-1",
            payload={"to": "previous_champion"},
        )
    with pytest.raises(ApprovalValidationError, match="payload"):
        service.decide(
            created["id"],
            decision="modify",
            human_actor_id="engineer-1",
            reason="adjust",
        )
    with pytest.raises(ApprovalValidationError, match="non-empty payload"):
        service.decide(
            created["id"],
            decision="modify",
            human_actor_id="engineer-1",
            reason="adjust",
            payload={},
        )


def test_duplicate_pending_proposal_is_rejected_but_terminal_allows_new(session_factory):
    service = make_service(session_factory)
    created = _propose(service)
    with pytest.raises(ApprovalConflictError, match="already pending"):
        _propose(service)
    service.decide(created["id"], decision="approve", human_actor_id="engineer-1")
    replacement = _propose(service)
    assert replacement["status"] == APPROVAL_STATUS_PENDING


def test_get_and_decide_unknown_id_raise_not_found(session_factory):
    service = make_service(session_factory)
    with pytest.raises(ApprovalNotFoundError):
        service.get("missing")
    with pytest.raises(ApprovalNotFoundError):
        service.decide("missing", decision="approve", human_actor_id="engineer-1")


# -- duplicate / stale CAS ---------------------------------------------------


def test_double_decide_is_conflict(session_factory):
    service = make_service(session_factory)
    created = _propose(service)
    service.decide(created["id"], decision="approve", human_actor_id="engineer-1")
    with pytest.raises(ApprovalConflictError, match="already approved"):
        service.decide(
            created["id"],
            decision="reject",
            human_actor_id="engineer-2",
            reason="changed my mind",
        )


def test_stale_version_transition_conflicts(session_factory):
    service = make_service(session_factory)
    repository = ApprovalRepository(session_factory)
    created = _propose(service)

    # Simulate a concurrent writer bumping the version while still pending.
    with session_factory.begin() as session:
        session.execute(
            update(ApprovalRequest)
            .where(ApprovalRequest.id == created["id"])
            .values(version=2)
        )

    with pytest.raises(ApprovalConflictError):
        repository.transition(
            created["id"], 1, status=APPROVAL_STATUS_APPROVED
        )


def test_service_rejects_stale_expected_version(session_factory):
    service = make_service(session_factory)
    created = _propose(service)
    with session_factory.begin() as session:
        session.execute(
            update(ApprovalRequest)
            .where(ApprovalRequest.id == created["id"])
            .values(version=2)
        )
    with pytest.raises(ApprovalConflictError):
        service.decide(
            created["id"],
            decision="approve",
            human_actor_id="engineer-1",
            expected_version=1,
        )


def test_transition_on_already_decided_conflicts(session_factory):
    service = make_service(session_factory)
    repository = ApprovalRepository(session_factory)
    created = _propose(service)
    service.decide(created["id"], decision="approve", human_actor_id="engineer-1")

    with pytest.raises(ApprovalConflictError):
        repository.transition(
            created["id"], 2, status=APPROVAL_STATUS_REJECTED, reason="nope"
        )


# -- audit failure rollback --------------------------------------------------


def test_propose_rolls_back_when_audit_fails(session_factory):
    repository = ApprovalRepository(session_factory)
    failing = make_service(session_factory, audit=_FailingAudit())
    with pytest.raises(RuntimeError, match="audit unavailable"):
        failing.propose(
            action_type=APPROVAL_ACTION_MODEL_PROMOTION,
            entity_type="registered_model",
            entity_id="model-1",
            requested_by_type=APPROVAL_REQUESTER_AGENT,
        )
    assert repository.list() == []


def test_decide_rolls_back_when_audit_fails(session_factory):
    repository = ApprovalRepository(session_factory)
    service = make_service(session_factory)
    created = _propose(service)

    failing = make_service(session_factory, audit=_FailingAudit())
    with pytest.raises(RuntimeError, match="audit unavailable"):
        failing.decide(created["id"], decision="approve", human_actor_id="engineer-1")

    persisted = repository.get(created["id"])
    assert persisted.status == APPROVAL_STATUS_PENDING
    assert persisted.version == 1
    assert persisted.decided_by is None
    assert persisted.decided_at is None


# -- filters -----------------------------------------------------------------


def test_list_filters(session_factory):
    service = make_service(session_factory)
    _propose(service, action_type=APPROVAL_ACTION_MODEL_PROMOTION, entity_id="model-1")
    _propose(service, action_type=APPROVAL_ACTION_CMMS_WORK_ORDER, entity_id="asset-9")
    _propose(
        service,
        action_type=APPROVAL_ACTION_RETRAINING_DEPLOYMENT,
        entity_id="model-1",
    )

    assert len(service.list()) == 3
    assert len(service.list(action_type=APPROVAL_ACTION_MODEL_PROMOTION)) == 1
    assert len(service.list(entity_id="model-1")) == 2
    assert len(service.list(status=APPROVAL_STATUS_PENDING)) == 3
    assert len(service.list(status=APPROVAL_STATUS_APPROVED)) == 0
    assert "proposed_payload" not in service.list()[0]
    assert "decision_payload" not in service.list()[0]

    decided = service.decide(
        service.list(action_type=APPROVAL_ACTION_CMMS_WORK_ORDER)[0]["id"],
        decision="approve",
        human_actor_id="engineer-1",
    )
    assert decided["status"] == APPROVAL_STATUS_APPROVED
    assert len(service.list(status=APPROVAL_STATUS_APPROVED)) == 1
    assert len(service.list(status=APPROVAL_STATUS_PENDING)) == 2


# -- database constraints ----------------------------------------------------


def _insert_raw(session_factory, **overrides):
    kwargs = {
        "action_type": APPROVAL_ACTION_MODEL_PROMOTION,
        "entity_type": "registered_model",
        "entity_id": "model-1",
        "status": APPROVAL_STATUS_PENDING,
        "requested_by_type": APPROVAL_REQUESTER_AGENT,
        "version": 1,
    }
    kwargs.update(overrides)
    with session_factory.begin() as session:
        session.add(ApprovalRequest(id=new_id(), **kwargs))


def test_check_constraints_reject_invalid_values(session_factory):
    with pytest.raises(IntegrityError):
        _insert_raw(session_factory, action_type="bogus")
    with pytest.raises(IntegrityError):
        _insert_raw(session_factory, status="bogus")
    with pytest.raises(IntegrityError):
        _insert_raw(session_factory, requested_by_type="bogus")
    with pytest.raises(IntegrityError):
        _insert_raw(session_factory, version=0)


# -- no registry side effects ------------------------------------------------


def test_no_registry_side_effects(session_factory):
    service = make_service(session_factory)
    assert not hasattr(service, "registry")
    assert not hasattr(service, "model_repository")

    created = _propose(service)
    service.decide(created["id"], decision="approve", human_actor_id="engineer-1")

    with session_factory() as session:
        approvals = session.scalar(
            select(func.count()).select_from(ApprovalRequest)
        )
        registered = session.scalar(select(func.count()).select_from(RegisteredModel))
    assert approvals == 1
    assert registered == 0

"""Integration tests for the P1 persistence foundation (models + repositories).

Exercises the four additive P1 tables against the in-memory SQLite test double
(``StaticPool``): ``ModelLifecycleState`` (one row per registered model,
challenger/champion/archived), ``ApprovalExecution`` (unique approval id,
succeeded receipt), ``TechnicianFeedback`` (append-only, four outcomes), and
``MockCMMSWorkOrder`` (unique approval id, draft -> approved). Covers repository
round-trips, list filters, the one-per-model / one-per-approval unique
invariants, append-only semantics, status transitions, and database check
constraints. P0 columns and semantics are untouched.
"""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from maintai.db.approval_execution_repository import ApprovalExecutionRepository
from maintai.db.cmms_repository import MockCMMSWorkOrderRepository
from maintai.db.feedback_repository import TechnicianFeedbackRepository
from maintai.db.lifecycle_repository import ModelLifecycleRepository
from maintai.db.models import (
    APPROVAL_ACTION_MODEL_PROMOTION,
    APPROVAL_REQUESTER_AGENT,
    APPROVAL_STATUS_PENDING,
    CMMS_WORK_ORDER_STATUS_APPROVED,
    CMMS_WORK_ORDER_STATUS_DRAFT,
    FEEDBACK_OUTCOME_CONFIRMED_ISSUE,
    FEEDBACK_OUTCOME_FALSE_ALARM,
    FEEDBACK_OUTCOME_NO_ACTION_NEEDED,
    MODEL_LIFECYCLE_ARCHIVED,
    MODEL_LIFECYCLE_CHALLENGER,
    MODEL_LIFECYCLE_CHAMPION,
    ApprovalExecution,
    ApprovalRequest,
    MockCMMSWorkOrder,
    ModelLifecycleState,
    TechnicianFeedback,
    new_id,
)


def _approval_id(session_factory, **overrides) -> str:
    """Persist a minimal pending approval and return its primary key."""
    kwargs = {
        "action_type": APPROVAL_ACTION_MODEL_PROMOTION,
        "entity_type": "registered_model",
        # A fresh entity_id avoids the pending (action, entity) unique index.
        "entity_id": new_id(),
        "status": APPROVAL_STATUS_PENDING,
        "requested_by_type": APPROVAL_REQUESTER_AGENT,
        "version": 1,
    }
    kwargs.update(overrides)
    approval = ApprovalRequest(id=new_id(), **kwargs)
    with session_factory.begin() as session:
        session.add(approval)
        session.flush()
    return approval.id


# -- ModelLifecycleState ------------------------------------------------------


def test_lifecycle_create_and_get_by_registered_model(session_factory):
    repo = ModelLifecycleRepository(session_factory)
    lifecycle = repo.create(
        ModelLifecycleState(
            id=new_id(), registered_model_id="rm-1", status=MODEL_LIFECYCLE_CHALLENGER
        )
    )
    fetched = repo.get(lifecycle.id)
    assert fetched is not None
    assert fetched.status == MODEL_LIFECYCLE_CHALLENGER
    by_model = repo.get_by_registered_model("rm-1")
    assert by_model is not None
    assert by_model.id == lifecycle.id
    assert repo.get_by_registered_model("missing") is None


def test_lifecycle_one_row_per_registered_model(session_factory):
    repo = ModelLifecycleRepository(session_factory)
    repo.create(
        ModelLifecycleState(
            id=new_id(), registered_model_id="rm-1", status=MODEL_LIFECYCLE_CHALLENGER
        )
    )
    with pytest.raises(IntegrityError):
        repo.create(
            ModelLifecycleState(
                id=new_id(), registered_model_id="rm-1", status=MODEL_LIFECYCLE_CHAMPION
            )
        )


def test_lifecycle_status_transition_and_list_filter(session_factory):
    repo = ModelLifecycleRepository(session_factory)
    lifecycle = repo.create(
        ModelLifecycleState(
            id=new_id(), registered_model_id="rm-1", status=MODEL_LIFECYCLE_CHALLENGER
        )
    )
    lifecycle.status = MODEL_LIFECYCLE_CHAMPION
    updated = repo.update(lifecycle)
    assert updated.status == MODEL_LIFECYCLE_CHAMPION
    assert repo.get(lifecycle.id).status == MODEL_LIFECYCLE_CHAMPION

    repo.create(
        ModelLifecycleState(
            id=new_id(), registered_model_id="rm-2", status=MODEL_LIFECYCLE_ARCHIVED
        )
    )
    assert len(repo.list()) == 2
    assert len(repo.list(status=MODEL_LIFECYCLE_CHAMPION)) == 1
    assert len(repo.list(status=MODEL_LIFECYCLE_ARCHIVED)) == 1


def test_lifecycle_check_constraint(session_factory):
    with pytest.raises(IntegrityError):
        with session_factory.begin() as session:
            session.add(
                ModelLifecycleState(
                    id=new_id(), registered_model_id="rm-1", status="bogus"
                )
            )


# -- ApprovalExecution --------------------------------------------------------


def test_execution_create_and_get_by_approval(session_factory):
    approval_id = _approval_id(session_factory)
    repo = ApprovalExecutionRepository(session_factory)
    execution = repo.create(
        ApprovalExecution(
            id=new_id(), approval_id=approval_id, receipt_json={"status": "ok"}
        )
    )
    assert execution.succeeded_at is not None
    assert execution.created_at is not None
    fetched = repo.get(execution.id)
    assert fetched is not None
    assert fetched.approval_id == approval_id
    by_approval = repo.get_by_approval(approval_id)
    assert by_approval is not None
    assert by_approval.id == execution.id
    assert repo.get_by_approval("missing") is None


def test_execution_one_receipt_per_approval(session_factory):
    approval_id = _approval_id(session_factory)
    repo = ApprovalExecutionRepository(session_factory)
    repo.create(ApprovalExecution(id=new_id(), approval_id=approval_id))
    with pytest.raises(IntegrityError):
        repo.create(ApprovalExecution(id=new_id(), approval_id=approval_id))


def test_execution_list_newest_first(session_factory):
    repo = ApprovalExecutionRepository(session_factory)
    first = repo.create(
        ApprovalExecution(id=new_id(), approval_id=_approval_id(session_factory))
    )
    second = repo.create(
        ApprovalExecution(id=new_id(), approval_id=_approval_id(session_factory))
    )
    listed = repo.list()
    assert [e.id for e in listed] == [second.id, first.id]


# -- TechnicianFeedback -------------------------------------------------------


def test_feedback_create_and_filters(session_factory):
    repo = TechnicianFeedbackRepository(session_factory)
    first = repo.create(
        TechnicianFeedback(
            id=new_id(),
            registered_model_id="rm-1",
            outcome=FEEDBACK_OUTCOME_CONFIRMED_ISSUE,
            comment="bearing temperature high",
        )
    )
    repo.create(
        TechnicianFeedback(
            id=new_id(), registered_model_id="rm-1", outcome=FEEDBACK_OUTCOME_FALSE_ALARM
        )
    )
    repo.create(
        TechnicianFeedback(
            id=new_id(),
            registered_model_id="rm-2",
            outcome=FEEDBACK_OUTCOME_NO_ACTION_NEEDED,
        )
    )
    fetched = repo.get(first.id)
    assert fetched is not None
    assert fetched.outcome == FEEDBACK_OUTCOME_CONFIRMED_ISSUE
    assert fetched.comment == "bearing temperature high"
    assert len(repo.list()) == 3
    assert len(repo.list(registered_model_id="rm-1")) == 2
    assert len(repo.list(outcome=FEEDBACK_OUTCOME_FALSE_ALARM)) == 1
    assert len(repo.list(prediction_event_id="pe-1")) == 0


def test_feedback_is_append_only(session_factory):
    repo = TechnicianFeedbackRepository(session_factory)
    # The repository intentionally exposes no mutation path.
    assert not hasattr(repo, "update")
    assert not hasattr(repo, "delete")
    assert not hasattr(repo, "remove")


def test_feedback_check_constraint(session_factory):
    with pytest.raises(IntegrityError):
        with session_factory.begin() as session:
            session.add(TechnicianFeedback(id=new_id(), outcome="bogus"))


# -- MockCMMSWorkOrder --------------------------------------------------------


def test_cmms_create_draft_and_transition_to_approved(session_factory):
    approval_id = _approval_id(session_factory)
    repo = MockCMMSWorkOrderRepository(session_factory)
    work_order = repo.create(
        MockCMMSWorkOrder(
            id=new_id(),
            approval_id=approval_id,
            asset_id="MOTOR_017",
            priority="HIGH",
            recommended_action="Inspect drive-end bearing",
            status=CMMS_WORK_ORDER_STATUS_DRAFT,
        )
    )
    assert work_order.status == CMMS_WORK_ORDER_STATUS_DRAFT
    by_approval = repo.get_by_approval(approval_id)
    assert by_approval is not None
    assert by_approval.id == work_order.id

    work_order.status = CMMS_WORK_ORDER_STATUS_APPROVED
    updated = repo.update(work_order)
    assert updated.status == CMMS_WORK_ORDER_STATUS_APPROVED
    assert repo.get(work_order.id).status == CMMS_WORK_ORDER_STATUS_APPROVED


def test_cmms_default_status_is_draft(session_factory):
    repo = MockCMMSWorkOrderRepository(session_factory)
    work_order = repo.create(
        MockCMMSWorkOrder(
            id=new_id(),
            approval_id=_approval_id(session_factory),
            asset_id="MOTOR_001",
            priority="LOW",
            recommended_action="Inspect",
        )
    )
    assert work_order.status == CMMS_WORK_ORDER_STATUS_DRAFT


def test_cmms_one_draft_per_approval(session_factory):
    approval_id = _approval_id(session_factory)
    repo = MockCMMSWorkOrderRepository(session_factory)
    repo.create(
        MockCMMSWorkOrder(
            id=new_id(),
            approval_id=approval_id,
            asset_id="MOTOR_017",
            priority="HIGH",
            recommended_action="Inspect",
        )
    )
    with pytest.raises(IntegrityError):
        repo.create(
            MockCMMSWorkOrder(
                id=new_id(),
                approval_id=approval_id,
                asset_id="MOTOR_018",
                priority="HIGH",
                recommended_action="Inspect",
            )
        )


def test_cmms_list_filters(session_factory):
    repo = MockCMMSWorkOrderRepository(session_factory)
    repo.create(
        MockCMMSWorkOrder(
            id=new_id(),
            approval_id=_approval_id(session_factory),
            asset_id="MOTOR_017",
            priority="HIGH",
            recommended_action="Inspect bearing",
        )
    )
    repo.create(
        MockCMMSWorkOrder(
            id=new_id(),
            approval_id=_approval_id(session_factory),
            asset_id="MOTOR_018",
            priority="LOW",
            recommended_action="Lubricate",
        )
    )
    assert len(repo.list()) == 2
    assert len(repo.list(status=CMMS_WORK_ORDER_STATUS_DRAFT)) == 2
    assert len(repo.list(asset_id="MOTOR_017")) == 1
    assert len(repo.list(asset_id="MOTOR_018")) == 1


def test_cmms_check_constraint(session_factory):
    with pytest.raises(IntegrityError):
        with session_factory.begin() as session:
            session.add(
                MockCMMSWorkOrder(
                    id=new_id(),
                    approval_id="approval-1",
                    asset_id="MOTOR_017",
                    priority="HIGH",
                    recommended_action="Inspect",
                    status="bogus",
                )
            )

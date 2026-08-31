"""Mock CMMS application service (P1 vertical slice, no live connector).

Turns an already-approved ``cmms_work_order`` approval into exactly one mock
work order (created directly in ``approved`` status because the human approval
already happened) plus one ``ApprovalExecution`` success receipt. The action is
idempotent: a second call for the same approval returns the existing work order
and receipt instead of writing duplicates (enforced by the unique
``approval_id`` on both tables).

The service only touches the business DB and the audit log. It never contacts a
live CMMS, never executes arbitrary SQL, and never copies the proposed/decision
payloads into the audit trail (audit events store only ids/action/entity/status).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from maintai.audit.service import AuditService
from maintai.db.approval_execution_repository import ApprovalExecutionRepository
from maintai.db.approval_repository import ApprovalRepository
from maintai.db.cmms_repository import MockCMMSWorkOrderRepository
from maintai.db.models import (
    APPROVAL_ACTION_CMMS_WORK_ORDER,
    APPROVAL_STATUS_APPROVED,
    APPROVAL_STATUS_MODIFIED,
    CMMS_WORK_ORDER_STATUS_APPROVED,
    ApprovalExecution,
    ApprovalRequest,
    MockCMMSWorkOrder,
    PredictionEvent,
    new_id,
)

MOCK_CMMS_DISCLAIMER = (
    "Mock CMMS integration for local demonstration only: no external "
    "maintenance system was contacted and no work order was dispatched."
)

# The CMMS work-order approval must reference exactly one prediction event.
_ENTITY_TYPE_PREDICTION_EVENT = "prediction_event"

# Strict payload allowlist and required keys (everything else is rejected).
_PAYLOAD_KEYS = frozenset(
    {
        "asset_id",
        "priority",
        "recommended_action",
        "prediction_event_id",
        "reason",
        "risk_score",
        "source_model_version",
        "evidence",
    }
)
_REQUIRED_PAYLOAD_KEYS = frozenset(
    {"asset_id", "priority", "recommended_action", "prediction_event_id"}
)

# Only a terminal approved state may be executed (approved or modified).
_EXECUTABLE_STATUSES = frozenset({APPROVAL_STATUS_APPROVED, APPROVAL_STATUS_MODIFIED})

# Length ceilings mirror the ``MockCMMSWorkOrder`` columns (maintai.db.models).
_ASSET_ID_MAX_LEN = 255
_PRIORITY_MAX_LEN = 32
_ACTION_MAX_LEN = 1024
_REASON_MAX_LEN = 2048
_EVENT_ID_MAX_LEN = 64
_MODEL_VERSION_MAX_LEN = 32


class MockCMMSServiceError(Exception):
    """Base class for mock-CMMS service errors."""


class CMMSApprovalNotFoundError(MockCMMSServiceError):
    """Raised when an approval id does not exist."""


class CMMSApprovalNotExecutableError(MockCMMSServiceError):
    """Raised when the approval is not approved/modified (pending/rejected)."""


class CMMSPredictionEventNotFoundError(MockCMMSServiceError):
    """Raised when the referenced prediction event does not exist."""


class CMMSValidationError(MockCMMSServiceError):
    """Raised when the action/entity/payload violates the CMMS contract."""


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


class MockCMMSService:
    """Application service for the mock CMMS vertical slice (execute approved action)."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        approval_repository: ApprovalRepository,
        work_order_repository: MockCMMSWorkOrderRepository,
        execution_repository: ApprovalExecutionRepository,
        audit: AuditService,
    ) -> None:
        self._session_factory = session_factory
        self._approval_repository = approval_repository
        self._work_order_repository = work_order_repository
        self._execution_repository = execution_repository
        self._audit = audit

    # -- rendering ---------------------------------------------------------

    @staticmethod
    def _iso(value) -> str | None:
        return value.isoformat() if value is not None else None

    @staticmethod
    def _to_dict(
        work_order: MockCMMSWorkOrder, execution: ApprovalExecution | None
    ) -> dict[str, Any]:
        """Render JSON-safe detail (evidence only; never approval payloads)."""
        return {
            "id": work_order.id,
            "approval_id": work_order.approval_id,
            "asset_id": work_order.asset_id,
            "priority": work_order.priority,
            "recommended_action": work_order.recommended_action,
            "reason": work_order.reason,
            "risk_score": work_order.risk_score,
            "evidence": work_order.evidence_json,
            "source_model_version": work_order.source_model_version,
            "status": work_order.status,
            "created_at": MockCMMSService._iso(work_order.created_at),
            "updated_at": MockCMMSService._iso(work_order.updated_at),
            "execution_id": execution.id if execution is not None else None,
            "mock": True,
            "disclaimer": MOCK_CMMS_DISCLAIMER,
        }

    # -- validation --------------------------------------------------------

    def _require_executable_payload(self, approval: ApprovalRequest) -> dict[str, Any]:
        """Validate the approval contract and return the effective payload.

        A ``modified`` approval executes its ``decision_payload``; an ``approved``
        approval executes its ``proposed_payload``. Strict checks: the action must
        be ``cmms_work_order``, the entity must be ``prediction_event``, the status
        must be approved/modified, and the payload must match the allowlist with
        the required fields present and typed.
        """
        if approval.action_type != APPROVAL_ACTION_CMMS_WORK_ORDER:
            raise CMMSValidationError(
                f"action_type must be {APPROVAL_ACTION_CMMS_WORK_ORDER!r}"
            )
        if approval.entity_type != _ENTITY_TYPE_PREDICTION_EVENT:
            raise CMMSValidationError(
                f"entity_type must be {_ENTITY_TYPE_PREDICTION_EVENT!r}"
            )
        if approval.status not in _EXECUTABLE_STATUSES:
            raise CMMSApprovalNotExecutableError(
                f"approval {approval.id!r} is {approval.status!r}, not approved/modified"
            )
        payload = (
            approval.decision_payload
            if approval.status == APPROVAL_STATUS_MODIFIED
            else approval.proposed_payload
        )
        return self._validate_payload(payload, approval.entity_id)

    def _validate_payload(
        self, payload: dict[str, Any] | None, entity_id: str
    ) -> dict[str, Any]:
        if not isinstance(payload, dict) or not payload:
            raise CMMSValidationError("approved payload is missing")
        unknown = sorted(set(payload) - _PAYLOAD_KEYS)
        if unknown:
            raise CMMSValidationError(f"unknown payload field(s): {unknown}")
        missing = sorted(_REQUIRED_PAYLOAD_KEYS - set(payload))
        if missing:
            raise CMMSValidationError(f"missing required payload field(s): {missing}")

        asset_id = payload["asset_id"]
        if not _nonempty(asset_id) or len(asset_id) > _ASSET_ID_MAX_LEN:
            raise CMMSValidationError(
                f"asset_id must be a non-empty string <= {_ASSET_ID_MAX_LEN} chars"
            )
        priority = payload["priority"]
        if not _nonempty(priority) or len(priority) > _PRIORITY_MAX_LEN:
            raise CMMSValidationError(
                f"priority must be a non-empty string <= {_PRIORITY_MAX_LEN} chars"
            )
        action = payload["recommended_action"]
        if not _nonempty(action) or len(action) > _ACTION_MAX_LEN:
            raise CMMSValidationError(
                "recommended_action must be a non-empty string <= "
                f"{_ACTION_MAX_LEN} chars"
            )
        event_id = payload["prediction_event_id"]
        if not _nonempty(event_id) or len(event_id) > _EVENT_ID_MAX_LEN:
            raise CMMSValidationError(
                f"prediction_event_id must be a non-empty string <= {_EVENT_ID_MAX_LEN} chars"
            )
        if event_id != entity_id:
            raise CMMSValidationError(
                "prediction_event_id must match the approval entity_id"
            )

        reason = payload.get("reason")
        if reason is not None and (not isinstance(reason, str) or len(reason) > _REASON_MAX_LEN):
            raise CMMSValidationError(f"reason must be a string <= {_REASON_MAX_LEN} chars")

        risk_score = payload.get("risk_score")
        if risk_score is not None and (
            isinstance(risk_score, bool) or not isinstance(risk_score, (int, float))
        ):
            raise CMMSValidationError("risk_score must be a number")

        source_model_version = payload.get("source_model_version")
        if source_model_version is not None and (
            not isinstance(source_model_version, str)
            or len(source_model_version) > _MODEL_VERSION_MAX_LEN
        ):
            raise CMMSValidationError(
                f"source_model_version must be a string <= {_MODEL_VERSION_MAX_LEN} chars"
            )

        evidence = payload.get("evidence")
        if evidence is not None and not isinstance(evidence, dict):
            raise CMMSValidationError("evidence must be an object")
        return payload

    def _verify_prediction_event(self, event_id: str) -> None:
        """Ensure the referenced prediction event actually exists."""
        with self._session_factory() as session:
            event = session.get(PredictionEvent, event_id)
        if event is None:
            raise CMMSPredictionEventNotFoundError(
                f"prediction event {event_id!r} not found"
            )

    # -- public API --------------------------------------------------------

    def create_draft(self, approval_id: str) -> tuple[dict[str, Any], bool]:
        """Create (or return the existing) mock work order + execution receipt.

        Returns ``(detail, created)`` where ``created`` is ``False`` when a draft
        and receipt already existed for this approval (idempotent replay).
        """
        approval = self._approval_repository.get(approval_id)
        if approval is None:
            raise CMMSApprovalNotFoundError(f"approval {approval_id!r} not found")
        payload = self._require_executable_payload(approval)
        self._verify_prediction_event(payload["prediction_event_id"])

        existing = self._work_order_repository.get_by_approval(approval_id)
        if existing is not None:
            receipt = self._execution_repository.get_by_approval(approval_id)
            return self._to_dict(existing, receipt), False

        work_order = MockCMMSWorkOrder(
            id=new_id(),
            approval_id=approval_id,
            asset_id=payload["asset_id"],
            priority=payload["priority"],
            recommended_action=payload["recommended_action"],
            reason=payload.get("reason"),
            risk_score=payload.get("risk_score"),
            evidence_json=payload.get("evidence"),
            source_model_version=payload.get("source_model_version"),
            status=CMMS_WORK_ORDER_STATUS_APPROVED,
        )
        execution = ApprovalExecution(
            id=new_id(),
            approval_id=approval_id,
            receipt_json={
                "work_order_id": work_order.id,
                "status": work_order.status,
                "mock": True,
            },
        )
        try:
            with self._session_factory.begin() as session:
                self._work_order_repository.create(work_order, session=session)
                self._execution_repository.create(execution, session=session)
                self._audit.record(
                    actor_type="system",
                    action="cmms.work_order_draft",
                    entity_type=approval.entity_type,
                    entity_id=approval.entity_id,
                    payload={
                        "work_order_id": work_order.id,
                        "approval_id": approval_id,
                        "action_type": approval.action_type,
                        "entity_type": approval.entity_type,
                        "entity_id": approval.entity_id,
                        "status": work_order.status,
                        "mock": True,
                    },
                    session=session,
                )
        except IntegrityError as exc:
            # A concurrent replay won the unique ``approval_id`` constraint; the
            # draft and receipt already exist, so return them instead of raising.
            existing = self._work_order_repository.get_by_approval(approval_id)
            if existing is None:
                raise MockCMMSServiceError("mock CMMS work-order draft failed") from exc
            receipt = self._execution_repository.get_by_approval(approval_id)
            return self._to_dict(existing, receipt), False
        return self._to_dict(work_order, execution), True

    def list(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """Return work orders newest-first (bounded) with their receipts."""
        work_orders = self._work_order_repository.list(limit=limit, offset=offset)
        if not work_orders:
            return []
        approval_ids = [work_order.approval_id for work_order in work_orders]
        receipts: dict[str, ApprovalExecution] = {}
        with self._session_factory() as session:
            for receipt in session.scalars(
                select(ApprovalExecution).where(
                    ApprovalExecution.approval_id.in_(approval_ids)
                )
            ):
                receipts[receipt.approval_id] = receipt
        return [
            self._to_dict(work_order, receipts.get(work_order.approval_id))
            for work_order in work_orders
        ]

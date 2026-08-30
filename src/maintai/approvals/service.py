"""Approval application service (P1 human-in-the-loop gate).

Coordinates the P1 approval state machine: an agent/system/user *proposes* an
operational action (model promotion, retraining deployment, maintenance action,
or CMMS work order) which is recorded as ``pending``; a human then *decides*
``approve``/``reject``/``modify``. Every proposal and decision is written to the
business DB and to the audit log in a single transaction.

Deciding an approval records the decision only — it never executes the
underlying action. The owning P1 module (champion/challenger, retraining, CMMS)
later reads the terminal row and applies it. Audit events store only metadata
(ids, action type, status, actor) and never the proposed/decision payloads, so
sensitive proposed actions and review content are not duplicated into the audit
trail.

For ``modified``, a future executor must use ``decision_payload`` rather than
the original ``proposed_payload``. This service never executes either payload.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from maintai.audit.service import AuditService
from maintai.db.approval_repository import ApprovalConflictError, ApprovalRepository
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
    new_id,
)

# Canonical decision verbs and their terminal statuses.
_APPROVAL_DECISIONS = {
    "approve": APPROVAL_STATUS_APPROVED,
    "reject": APPROVAL_STATUS_REJECTED,
    "modify": APPROVAL_STATUS_MODIFIED,
}

_APPROVAL_ACTIONS = frozenset(
    {
        APPROVAL_ACTION_MODEL_PROMOTION,
        APPROVAL_ACTION_RETRAINING_DEPLOYMENT,
        APPROVAL_ACTION_MAINTENANCE_ACTION,
        APPROVAL_ACTION_CMMS_WORK_ORDER,
    }
)

_APPROVAL_REQUESTERS = frozenset(
    {APPROVAL_REQUESTER_AGENT, APPROVAL_REQUESTER_USER, APPROVAL_REQUESTER_SYSTEM}
)


class ApprovalServiceError(Exception):
    """Base class for approval application-service errors."""


class ApprovalNotFoundError(ApprovalServiceError):
    """Raised when an approval id does not exist."""


class ApprovalValidationError(ApprovalServiceError):
    """Raised when a proposal or decision violates the approval contract."""


class ApprovalService:
    """Application service for the approval state machine (no action execution)."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        repository: ApprovalRepository,
        audit: AuditService,
    ) -> None:
        self._session_factory = session_factory
        self._repository = repository
        self._audit = audit

    @staticmethod
    def _to_dict(
        approval: ApprovalRequest,
        *,
        include_payloads: bool = True,
    ) -> dict[str, Any]:
        """Render JSON-safe detail or a redacted list summary."""
        result = {
            "id": approval.id,
            "action_type": approval.action_type,
            "entity_type": approval.entity_type,
            "entity_id": approval.entity_id,
            "status": approval.status,
            "requested_by_type": approval.requested_by_type,
            "requested_by_id": approval.requested_by_id,
            "decided_by": approval.decided_by,
            "reason": approval.reason,
            "version": approval.version,
            "created_at": approval.created_at.isoformat() if approval.created_at else None,
            "decided_at": approval.decided_at.isoformat() if approval.decided_at else None,
        }
        if include_payloads:
            result["proposed_payload"] = approval.proposed_payload
            result["decision_payload"] = approval.decision_payload
        return result

    @staticmethod
    def _nonempty(value: str | None) -> bool:
        return isinstance(value, str) and bool(value.strip())

    # -- public API --------------------------------------------------------

    def propose(
        self,
        *,
        action_type: str,
        entity_type: str,
        entity_id: str,
        requested_by_type: str,
        requested_by_id: str | None = None,
        proposed_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a pending approval request (plus its audit event) atomically.

        The request is always created in ``pending`` with ``version=1``; nothing
        is executed here.
        """
        if action_type not in _APPROVAL_ACTIONS:
            raise ApprovalValidationError(
                f"action_type must be one of {sorted(_APPROVAL_ACTIONS)}"
            )
        if requested_by_type not in _APPROVAL_REQUESTERS:
            raise ApprovalValidationError(
                f"requested_by_type must be one of {sorted(_APPROVAL_REQUESTERS)}"
            )
        if not self._nonempty(entity_type):
            raise ApprovalValidationError("entity_type is required")
        if not self._nonempty(entity_id):
            raise ApprovalValidationError("entity_id is required")
        if len(entity_type) > 64 or len(entity_id) > 64:
            raise ApprovalValidationError("entity_type and entity_id must be at most 64 chars")
        if requested_by_id is not None and len(requested_by_id) > 64:
            raise ApprovalValidationError("requested_by_id must be at most 64 chars")
        if self._repository.find_pending(
            action_type=action_type,
            entity_type=entity_type,
            entity_id=entity_id,
        ):
            raise ApprovalConflictError("an equivalent approval request is already pending")

        approval = ApprovalRequest(
            id=new_id(),
            action_type=action_type,
            entity_type=entity_type,
            entity_id=entity_id,
            status=APPROVAL_STATUS_PENDING,
            proposed_payload=proposed_payload,
            requested_by_type=requested_by_type,
            requested_by_id=requested_by_id,
            version=1,
        )
        try:
            with self._session_factory.begin() as session:
                approval = self._repository.create(approval, session=session)
                self._audit.record(
                    actor_type=requested_by_type,
                    actor_id=requested_by_id,
                    action="approval.propose",
                    entity_type=entity_type,
                    entity_id=entity_id,
                    payload={
                        "approval_id": approval.id,
                        "action_type": action_type,
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "status": APPROVAL_STATUS_PENDING,
                        "requested_by_type": requested_by_type,
                    },
                    session=session,
                )
        except IntegrityError as exc:
            raise ApprovalConflictError(
                "an equivalent approval request is already pending"
            ) from exc
        return self._to_dict(approval)

    def decide(
        self,
        approval_id: str,
        *,
        decision: str,
        human_actor_id: str,
        human_actor_type: str = "user",
        expected_version: int | None = None,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Decide a pending approval as ``approve``/``reject``/``modify``.

        ``human_actor_id`` is mandatory and ``human_actor_type`` must be
        ``user``. Route-level authentication must supply that principal.
        ``reject`` and ``modify`` require a
        non-empty ``reason``; ``modify`` additionally requires a non-empty
        ``payload`` (the modified proposal). A terminal approval cannot be
        decided again. The decision and its audit event commit atomically, and
        the underlying action is never executed here.
        """
        if not self._nonempty(human_actor_id):
            raise ApprovalValidationError("human_actor_id is required")
        if human_actor_type != APPROVAL_REQUESTER_USER:
            raise ApprovalValidationError("only a human user may decide an approval")
        if len(human_actor_id) > 64:
            raise ApprovalValidationError("human_actor_id must be at most 64 chars")
        if expected_version is not None and expected_version < 1:
            raise ApprovalValidationError("expected_version must be >= 1")
        status = _APPROVAL_DECISIONS.get(decision)
        if status is None:
            raise ApprovalValidationError(
                f"decision must be one of {sorted(_APPROVAL_DECISIONS)}"
            )
        if decision in ("reject", "modify") and not self._nonempty(reason):
            raise ApprovalValidationError(f"reason is required for decision {decision!r}")
        if decision == "modify" and not (isinstance(payload, dict) and payload):
            raise ApprovalValidationError("non-empty payload is required for decision 'modify'")
        if reason is not None and len(reason) > 2048:
            raise ApprovalValidationError("reason must be at most 2048 chars")

        approval = self._repository.get(approval_id)
        if approval is None:
            raise ApprovalNotFoundError(f"approval {approval_id!r} not found")
        if approval.status != APPROVAL_STATUS_PENDING:
            raise ApprovalConflictError(
                f"approval {approval_id!r} is already {approval.status}"
            )

        with self._session_factory.begin() as session:
            self._repository.transition(
                approval_id,
                expected_version if expected_version is not None else approval.version,
                status=status,
                decision_payload=payload,
                decided_by=human_actor_id,
                reason=reason,
                session=session,
            )
            self._audit.record(
                actor_type="user",
                actor_id=human_actor_id,
                action=f"approval.{decision}",
                entity_type=approval.entity_type,
                entity_id=approval.entity_id,
                payload={
                    "approval_id": approval_id,
                    "action_type": approval.action_type,
                    "entity_type": approval.entity_type,
                    "entity_id": approval.entity_id,
                    "status": status,
                    "decided_by": human_actor_id,
                },
                session=session,
            )
        return self.get(approval_id)

    def get(self, approval_id: str) -> dict[str, Any]:
        """Return an approval request by id."""
        approval = self._repository.get(approval_id)
        if approval is None:
            raise ApprovalNotFoundError(f"approval {approval_id!r} not found")
        return self._to_dict(approval)

    def list(
        self,
        *,
        action_type: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        status: str | None = None,
        requested_by_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return approval summaries matching the given filters."""
        approvals = self._repository.list(
            action_type=action_type,
            entity_type=entity_type,
            entity_id=entity_id,
            status=status,
            requested_by_type=requested_by_type,
            limit=limit,
            offset=offset,
        )
        return [self._to_dict(approval, include_payloads=False) for approval in approvals]

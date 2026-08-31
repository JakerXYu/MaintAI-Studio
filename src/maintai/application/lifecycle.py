"""Model champion/challenger lifecycle application service (P1).

Coordinates the P1 champion/challenger loop on top of the frozen P0 registry:
* ``request_promotion`` records a ``challenger`` lifecycle overlay for a
  registered model and proposes a ``model_promotion`` approval (via the shared
  :class:`~maintai.approvals.service.ApprovalService`) — nothing is executed.
* ``execute_promotion`` applies an *already approved* (or modified) promotion:
  it moves the MLflow ``champion`` alias to the target version, archives the
  previous champion's lifecycle row, writes an idempotent execution receipt, and
  audits the mutation.

P0's ``RegisteredModel.deployment_status``/``deployed`` columns are never
touched, so demo-deploy and demo prediction semantics are unchanged. A promotion
is gated by the approval row, not by the ``deploy_demo`` endpoint; the MLflow
``champion`` alias and the lifecycle overlay are separate from the P0
``candidate`` alias / demo serving path.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from maintai.approvals import ApprovalService
from maintai.audit.service import AuditService
from maintai.db.approval_execution_repository import ApprovalExecutionRepository
from maintai.db.approval_repository import ApprovalRepository
from maintai.db.lifecycle_repository import ModelLifecycleRepository
from maintai.db.model_repository import ModelRepository
from maintai.db.models import (
    APPROVAL_ACTION_MODEL_PROMOTION,
    APPROVAL_STATUS_APPROVED,
    APPROVAL_STATUS_MODIFIED,
    MODEL_LIFECYCLE_ARCHIVED,
    MODEL_LIFECYCLE_CHALLENGER,
    MODEL_LIFECYCLE_CHAMPION,
    ApprovalExecution,
    ModelLifecycleState,
    new_id,
)
from maintai.mlops.registry import CHAMPION_ALIAS, MLflowRegistry, RegistryError

# The approval entity type used by the lifecycle service. It mirrors the value
# the approvals API already uses in its tests/docs and must stay in sync.
_ENTITY_TYPE = "registered_model"


class LifecycleServiceError(Exception):
    """Base class for champion/challenger lifecycle application-service errors."""


class LifecycleModelNotFoundError(LifecycleServiceError):
    """Raised when a registered-model id does not exist."""


class LifecycleNotPromotableError(LifecycleServiceError):
    """Raised when a registered model cannot be promoted (champion/archived)."""


class LifecycleApprovalNotFoundError(LifecycleServiceError):
    """Raised when an approval id does not exist."""


class LifecycleApprovalNotReadyError(LifecycleServiceError):
    """Raised when an approval is not yet approved/modified."""


class LifecycleValidationError(LifecycleServiceError):
    """Raised when the approval's action/entity/effective payload is invalid."""


class LifecyclePromotionError(LifecycleServiceError):
    """Raised when the MLflow champion promotion cannot complete."""


class ModelLifecycleService:
    """Application service for the champion/challenger lifecycle."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        model_repository: ModelRepository,
        lifecycle_repository: ModelLifecycleRepository,
        approval_repository: ApprovalRepository,
        execution_repository: ApprovalExecutionRepository,
        audit: AuditService,
        registry: MLflowRegistry,
        approval_service: ApprovalService,
    ) -> None:
        self._session_factory = session_factory
        self._model_repository = model_repository
        self._lifecycle_repository = lifecycle_repository
        self._approval_repository = approval_repository
        self._execution_repository = execution_repository
        self._audit = audit
        self._registry = registry
        self._approval_service = approval_service

    # -- helpers -----------------------------------------------------------

    def _get_model_or_raise(self, registered_model_id: str):
        model = self._model_repository.get(registered_model_id)
        if model is None:
            raise LifecycleModelNotFoundError(
                f"registered model {registered_model_id!r} not found"
            )
        return model

    @staticmethod
    def _effective_payload(approval) -> dict[str, Any]:
        """Return the payload to execute (proposed for approved, decision for modified)."""
        payload = (
            approval.decision_payload
            if approval.status == APPROVAL_STATUS_MODIFIED
            else approval.proposed_payload
        )
        if not isinstance(payload, dict) or not payload:
            raise LifecycleValidationError("approval has no effective payload to execute")
        return payload

    @staticmethod
    def _validate_payload(payload: dict[str, Any], model) -> None:
        """Validate the effective payload still targets the same registered model."""
        if payload.get("registered_model_id") != model.id:
            raise LifecycleValidationError(
                "effective payload registered_model_id does not match the model"
            )
        if "name" in payload and payload["name"] != model.name:
            raise LifecycleValidationError("effective payload name does not match the model")
        if "version" in payload and payload["version"] != model.version:
            raise LifecycleValidationError("effective payload version does not match the model")

    def _validate_approval(self, approval, registered_model_id: str) -> None:
        """Validate action type, entity, and terminal status before execution."""
        if approval.action_type != APPROVAL_ACTION_MODEL_PROMOTION:
            raise LifecycleValidationError(
                f"approval action must be {APPROVAL_ACTION_MODEL_PROMOTION!r}"
            )
        if approval.entity_type != _ENTITY_TYPE:
            raise LifecycleValidationError(
                f"approval entity_type must be {_ENTITY_TYPE!r}"
            )
        if approval.entity_id != registered_model_id:
            raise LifecycleValidationError(
                "approval entity_id does not match the registered model"
            )
        if approval.status not in (APPROVAL_STATUS_APPROVED, APPROVAL_STATUS_MODIFIED):
            raise LifecycleApprovalNotReadyError(
                f"approval {approval.id!r} is {approval.status!r}, not approved or modified"
            )

    def _idempotent_result(
        self, registered_model_id: str, approval_id: str, execution
    ) -> dict[str, Any]:
        """Render the already-executed receipt (no re-execution)."""
        model = self._model_repository.get(registered_model_id)
        approval = self._approval_repository.get(approval_id)
        lifecycle = self._lifecycle_repository.get_by_registered_model(registered_model_id)
        receipt = execution.receipt_json if isinstance(execution.receipt_json, dict) else {}
        return {
            "registered_model_id": registered_model_id,
            "name": model.name if model else None,
            "version": model.version if model else None,
            "lifecycle_status": lifecycle.status if lifecycle else None,
            "approval_id": approval_id,
            "approval_status": approval.status if approval else None,
            "executed": False,
            "receipt_id": execution.id,
            "archived_champion_ids": receipt.get("archived_champion_ids", []),
            "mlflow_model_uri": (
                f"models:/{model.name}/{model.version}" if model else None
            ),
        }

    # -- public API --------------------------------------------------------

    def request_promotion(
        self,
        registered_model_id: str,
        *,
        requested_by_type: str = "agent",
        requested_by_id: str | None = None,
    ) -> dict[str, Any]:
        """Record a challenger and propose a ``model_promotion`` approval.

        The model must exist and not already be the champion (or archived). The
        challenger overlay is created first; the approval is then proposed via
        the shared approval service, which owns validation and audit. Nothing is
        executed here.
        """
        model = self._get_model_or_raise(registered_model_id)

        proposed_payload: dict[str, Any] = {
            "registered_model_id": registered_model_id,
            "name": model.name,
            "version": model.version,
            "to_alias": CHAMPION_ALIAS,
        }
        with self._session_factory.begin() as session:
            existing = self._lifecycle_repository.get_by_registered_model(
                registered_model_id, session=session
            )
            if existing is not None and existing.status == MODEL_LIFECYCLE_CHAMPION:
                raise LifecycleNotPromotableError(
                    f"registered model {registered_model_id!r} is already the champion"
                )
            if existing is not None and existing.status == MODEL_LIFECYCLE_ARCHIVED:
                raise LifecycleNotPromotableError(
                    f"registered model {registered_model_id!r} is archived"
                )
            if existing is None:
                self._lifecycle_repository.create(
                    ModelLifecycleState(
                        id=new_id(),
                        registered_model_id=registered_model_id,
                        status=MODEL_LIFECYCLE_CHALLENGER,
                    ),
                    session=session,
                )
                self._audit.record(
                    actor_type="system",
                    action="model.lifecycle.ensure_challenger",
                    entity_type=_ENTITY_TYPE,
                    entity_id=registered_model_id,
                    payload={
                        "registered_model_id": registered_model_id,
                        "status": MODEL_LIFECYCLE_CHALLENGER,
                    },
                    session=session,
                )
            approval = self._approval_service.propose(
                action_type=APPROVAL_ACTION_MODEL_PROMOTION,
                entity_type=_ENTITY_TYPE,
                entity_id=registered_model_id,
                requested_by_type=requested_by_type,
                requested_by_id=requested_by_id,
                proposed_payload=proposed_payload,
                session=session,
            )
        return {
            "approval_id": approval["id"],
            "approval_status": approval["status"],
            "registered_model_id": registered_model_id,
            "lifecycle_status": MODEL_LIFECYCLE_CHALLENGER,
            "proposed_payload": proposed_payload,
        }

    def execute_promotion(
        self,
        registered_model_id: str,
        approval_id: str,
        *,
        human_actor_id: str,
    ) -> dict[str, Any]:
        """Execute an approved/modified champion promotion.

        Idempotent: if the approval already has an execution receipt the receipt
        is returned unchanged (no re-execution). Otherwise the approval's
        action/entity/effective payload are validated, the MLflow ``champion``
        alias is moved to this version, the previous champion is archived, the
        receipt is written, and the mutation is audited — all in one DB
        transaction.
        """
        if not (isinstance(human_actor_id, str) and human_actor_id.strip()):
            raise LifecycleValidationError("human_actor_id is required")

        existing_execution = self._execution_repository.get_by_approval(approval_id)
        if existing_execution is not None:
            return self._idempotent_result(
                registered_model_id, approval_id, existing_execution
            )

        model = self._get_model_or_raise(registered_model_id)
        approval = self._approval_repository.get(approval_id)
        if approval is None:
            raise LifecycleApprovalNotFoundError(f"approval {approval_id!r} not found")

        self._validate_approval(approval, registered_model_id)
        self._validate_payload(self._effective_payload(approval), model)

        # External MLflow mutation happens before the DB transaction. If the
        # transaction later fails the alias move has already happened; retrying
        # is safe because set_registered_model_alias is idempotent.
        try:
            promoted = self._registry.promote_champion(model.name, model.version)
        except RegistryError as exc:
            raise LifecyclePromotionError(str(exc)) from exc

        archived_ids: list[str] = []
        try:
            with self._session_factory.begin() as session:
                target = self._model_repository.get(registered_model_id, session=session)
                for sibling in self._model_repository.list_by_name(target.name, session=session):
                    if sibling.id == target.id:
                        continue
                    lifecycle = self._lifecycle_repository.get_by_registered_model(
                        sibling.id, session=session
                    )
                    if lifecycle is not None and lifecycle.status == MODEL_LIFECYCLE_CHAMPION:
                        lifecycle.status = MODEL_LIFECYCLE_ARCHIVED
                        self._lifecycle_repository.update(lifecycle, session=session)
                        archived_ids.append(sibling.id)

                lifecycle = self._lifecycle_repository.get_by_registered_model(
                    target.id, session=session
                )
                if lifecycle is None:
                    lifecycle = ModelLifecycleState(
                        id=new_id(),
                        registered_model_id=target.id,
                        status=MODEL_LIFECYCLE_CHAMPION,
                    )
                    self._lifecycle_repository.create(lifecycle, session=session)
                else:
                    lifecycle.status = MODEL_LIFECYCLE_CHAMPION
                    self._lifecycle_repository.update(lifecycle, session=session)

                receipt = ApprovalExecution(
                    id=new_id(),
                    approval_id=approval_id,
                    receipt_json={
                        "registered_model_id": target.id,
                        "name": target.name,
                        "version": target.version,
                        "alias": CHAMPION_ALIAS,
                        "approval_status": approval.status,
                        "archived_champion_ids": archived_ids,
                    },
                )
                self._execution_repository.create(receipt, session=session)
                self._audit.record(
                    actor_type="user",
                    actor_id=human_actor_id.strip(),
                    action="model.promote_champion",
                    entity_type=_ENTITY_TYPE,
                    entity_id=target.id,
                    payload={
                        "registered_model_id": target.id,
                        "name": target.name,
                        "version": target.version,
                        "alias": CHAMPION_ALIAS,
                        "approval_id": approval_id,
                        "approval_status": approval.status,
                        "archived_champion_ids": archived_ids,
                    },
                    session=session,
                )
        except IntegrityError:
            concurrent = self._execution_repository.get_by_approval(approval_id)
            if concurrent is None:
                raise
            return self._idempotent_result(registered_model_id, approval_id, concurrent)

        return {
            "registered_model_id": registered_model_id,
            "name": model.name,
            "version": model.version,
            "lifecycle_status": MODEL_LIFECYCLE_CHAMPION,
            "approval_id": approval_id,
            "approval_status": approval.status,
            "executed": True,
            "receipt_id": receipt.id,
            "archived_champion_ids": archived_ids,
            "mlflow_model_uri": promoted.model_uri,
        }

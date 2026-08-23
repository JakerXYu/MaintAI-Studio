"""Convenience service for recording audit events."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from maintai.audit.repository import AuditRepository
from maintai.db.models import AuditEvent


class AuditService:
    def __init__(self, repository: AuditRepository) -> None:
        self._repository = repository

    def record(
        self,
        *,
        actor_type: str,
        action: str,
        actor_id: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> AuditEvent:
        event = AuditEvent(
            actor_type=actor_type,
            actor_id=actor_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            payload_json=payload,
        )
        return self._repository.add(event)


def get_audit_service(session_factory: sessionmaker[Session]) -> AuditService:
    return AuditService(AuditRepository(session_factory))

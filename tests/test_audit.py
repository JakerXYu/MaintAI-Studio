"""Minimal audit repository/service tests (SQLite test double)."""

from maintai.audit.repository import AuditRepository
from maintai.audit.service import AuditService
from maintai.db.models import AuditEvent


def test_audit_service_records_event(session_factory):
    service = AuditService(AuditRepository(session_factory))
    ev = service.record(actor_type="system", action="health.check", payload={"ok": True})
    assert ev.id
    assert ev.actor_type == "system"
    assert ev.action == "health.check"


def test_audit_repository_lists_by_entity(session_factory):
    repo = AuditRepository(session_factory)
    repo.add(
        AuditEvent(
            actor_type="system",
            action="dataset.upload",
            entity_type="dataset",
            entity_id="d1",
        )
    )
    rows = repo.list(entity_type="dataset", entity_id="d1")
    assert len(rows) == 1
    assert rows[0].action == "dataset.upload"


def test_audit_repository_clamps_limit(session_factory):
    repo = AuditRepository(session_factory)
    repo.add(AuditEvent(actor_type="system", action="one"))
    repo.add(AuditEvent(actor_type="system", action="two"))
    assert len(repo.list(limit=0)) == 1

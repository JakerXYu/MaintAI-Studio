"""ORM contract tests against the SQLite test double."""

from sqlalchemy import select

from maintai.db.models import AuditEvent, Dataset, Experiment, ModelRun, RegisteredModel


def test_dataset_roundtrip(session_factory):
    with session_factory() as s:
        d = Dataset(name="ai4i", source_type="csv", row_count=10000, column_count=14)
        s.add(d)
        s.commit()
        s.refresh(d)
        assert d.id

    with session_factory() as s:
        got = s.get(Dataset, d.id)
        assert got is not None
        assert got.name == "ai4i"
        assert got.row_count == 10000
        assert got.created_at is not None


def test_model_lifecycle_chain(session_factory):
    with session_factory() as s:
        ds = Dataset(name="demo", source_type="parquet")
        s.add(ds)
        s.commit()
        s.refresh(ds)

        exp = Experiment(dataset_id=ds.id, name="failure-risk", task_type="binary_classification")
        s.add(exp)
        s.commit()
        s.refresh(exp)

        run = ModelRun(experiment_id=exp.id, model_name="RandomForest", status="created")
        s.add(run)
        s.commit()
        s.refresh(run)

        reg = RegisteredModel(
            model_run_id=run.id,
            experiment_id=exp.id,
            name="failure_model",
            version="1",
        )
        s.add(reg)
        s.commit()
        s.refresh(reg)

    with session_factory() as s:
        got = s.scalars(select(ModelRun).where(ModelRun.model_name == "RandomForest")).one()
        assert got.experiment_id == exp.id
        regs = s.scalars(select(RegisteredModel)).all()
        assert len(regs) == 1
        assert regs[0].approval_status == "pending"
        assert regs[0].deployed is False


def test_audit_event_persists_json_payload(session_factory):
    with session_factory() as s:
        e = AuditEvent(
            actor_type="system",
            action="dataset.upload",
            entity_type="dataset",
            entity_id="abc",
            payload_json={"size_mb": 10, "hash": "sha256:deadbeef"},
        )
        s.add(e)
        s.commit()
        s.refresh(e)
        assert e.id

    with session_factory() as s:
        got = s.get(AuditEvent, e.id)
        assert got is not None
        assert got.actor_type == "system"
        assert got.payload_json == {"size_mb": 10, "hash": "sha256:deadbeef"}

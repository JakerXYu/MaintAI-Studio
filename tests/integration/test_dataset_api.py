"""Integration tests for the Dataset REST API.

Exercise the HTTP layer end-to-end against an in-memory SQLite database
(StaticPool) and a ``tmp_path`` storage root, mirroring the service-level
tests in ``tests/integration/test_dataset_service.py``.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from maintai.api.main import create_app
from maintai.application.datasets import DatasetService
from maintai.audit.repository import AuditRepository
from maintai.audit.service import AuditService
from maintai.db.dataset_repository import DatasetRepository

CSV_BYTES = (
    b"timestamp,asset_id,serial_no,sensor_temp,failure\n"
    b"2024-01-01 00:00:00,A,1001,20.1,0\n"
    b"2024-01-01 01:00:00,A,1002,20.3,0\n"
    b"2024-01-01 02:00:00,B,1003,21.0,1\n"
    b"2024-01-01 03:00:00,B,1004,21.4,1\n"
    b"2024-01-01 04:00:00,A,1005,20.2,0\n"
    b"2024-01-01 05:00:00,C,1006,22.1,1\n"
    b"2024-01-01 06:00:00,C,1007,22.3,1\n"
    b"2024-01-01 07:00:00,A,1008,20.0,0\n"
)

UPLOAD_URL = "/api/v1/datasets/upload"


def _parquet_bytes() -> bytes:
    table = pa.table(
        {
            "asset_id": ["A", "B", "A", "B"],
            "sensor": [1.0, 2.0, 3.0, 4.0],
            "failure": [0, 1, 0, 1],
        }
    )
    buffer = pa.BufferOutputStream()
    pq.write_table(table, buffer)
    return buffer.getvalue().to_pybytes()


@pytest.fixture()
def service(session_factory, tmp_path):
    audit = AuditService(AuditRepository(session_factory))
    return DatasetService(
        repository=DatasetRepository(session_factory),
        session_factory=session_factory,
        audit=audit,
        storage_root=tmp_path,
    )


@pytest.fixture()
def api_client(session_factory, service):
    app = create_app(
        session_factory=session_factory,
        dataset_service=service,
        mlflow_healthcheck=lambda: None,
    )
    with TestClient(app) as test_client:
        yield test_client


def _upload(client, filename, content, content_type="text/csv"):
    return client.post(UPLOAD_URL, files={"file": (filename, content, content_type)})


def test_upload_csv_201_stable_shape(api_client):
    r = _upload(api_client, "plant_sensor_log.csv", CSV_BYTES)

    assert r.status_code == 201
    body = r.json()
    assert body["id"]
    assert body["original_filename"] == "plant_sensor_log.csv"
    assert body["source_type"] == "csv"
    assert "file_path" not in body
    assert body["size_bytes"] == len(CSV_BYTES)
    assert body["status"] == "uploaded"
    assert body["row_count"] == 8
    assert body["column_count"] == 5
    assert body["schema"]["columns"]


def test_upload_parquet_list_get(api_client):
    r = _upload(
        api_client,
        "sensor_data.parquet",
        _parquet_bytes(),
        "application/octet-stream",
    )
    assert r.status_code == 201
    body = r.json()
    assert body["source_type"] == "parquet"
    assert "file_path" not in body

    listing = api_client.get("/api/v1/datasets")
    assert listing.status_code == 200
    assert [item["id"] for item in listing.json()] == [body["id"]]

    got = api_client.get(f"/api/v1/datasets/{body['id']}")
    assert got.status_code == 200
    assert got.json()["id"] == body["id"]


def test_duplicate_409_includes_existing_id_without_path(api_client):
    first = _upload(api_client, "one.csv", CSV_BYTES).json()

    r = _upload(api_client, "two.csv", CSV_BYTES)
    assert r.status_code == 409
    detail = r.json()
    assert detail["existing_dataset_id"] == first["id"]
    assert "file_path" not in detail
    assert "path" not in detail


def test_invalid_and_path_filenames_422(api_client):
    assert _upload(api_client, "../evil.csv", b"a,b\n1,2\n").status_code == 422
    assert _upload(api_client, "file.txt", b"x").status_code == 422
    assert _upload(api_client, "", b"a,b\n1,2\n").status_code == 422
    assert _upload(api_client, "empty.csv", b"").status_code == 422


def test_oversize_returns_413(session_factory, tmp_path):
    tiny = DatasetService(
        repository=DatasetRepository(session_factory),
        session_factory=session_factory,
        audit=AuditService(AuditRepository(session_factory)),
        storage_root=tmp_path,
        max_upload_bytes=4,
    )
    app = create_app(
        session_factory=session_factory,
        dataset_service=tiny,
        mlflow_healthcheck=lambda: None,
    )
    with TestClient(app) as client:
        r = _upload(client, "a.csv", b"12345")
    assert r.status_code == 413


def test_profile_quality_task_flow(api_client):
    up = _upload(api_client, "plant_sensor_log.csv", CSV_BYTES).json()
    did = up["id"]

    prof = api_client.post(f"/api/v1/datasets/{did}/profile")
    assert prof.status_code == 200
    prof_body = prof.json()
    assert prof_body["status"] == "profiled"
    assert prof_body["quality_score"] is not None

    qual = api_client.get(f"/api/v1/datasets/{did}/quality")
    assert qual.status_code == 200
    qual_body = qual.json()
    assert qual_body["score"] == prof_body["quality_score"]
    assert qual_body["report"] is not None

    task = api_client.post(
        f"/api/v1/datasets/{did}/task-recommendation",
        json={
            "target_column": "failure",
            "asset_id_column": "asset_id",
            "timestamp_column": "timestamp",
        },
    )
    assert task.status_code == 200
    task_body = task.json()
    assert task_body["dataset_id"] == did
    assert task_body["task"]["recommended_task"] == "binary_classification"
    assert task_body["task"]["trainable"] is True
    assert task_body["leakage"]["verdict"] == "block"
    assert task_body["leakage"]["excluded_features"] == ["serial_no"]
    assert isinstance(task_body["quality_score"], float)

    got = api_client.get(f"/api/v1/datasets/{did}")
    assert got.status_code == 200
    assert got.json()["target_column"] == "failure"
    assert got.json()["task_type"] == "binary_classification"


def test_not_found_404(api_client):
    assert api_client.get("/api/v1/datasets/does-not-exist").status_code == 404
    assert api_client.post("/api/v1/datasets/does-not-exist/profile").status_code == 404
    assert api_client.get("/api/v1/datasets/does-not-exist/quality").status_code == 404
    r = api_client.post(
        "/api/v1/datasets/does-not-exist/task-recommendation",
        json={"target_column": "failure"},
    )
    assert r.status_code == 404


def test_storage_failure_500_without_path_leak(api_client, tmp_path):
    up = _upload(api_client, "data.csv", CSV_BYTES).json()
    stored_files = [path for path in tmp_path.iterdir() if path.is_file()]
    assert len(stored_files) == 1
    stored_files[0].unlink()

    r = api_client.post(f"/api/v1/datasets/{up['id']}/profile")
    assert r.status_code == 500
    assert str(tmp_path) not in r.text
    assert "dataset storage is unavailable" in r.text


def test_mutation_audit_written_once_via_service(api_client, session_factory):
    up = _upload(api_client, "data.csv", CSV_BYTES).json()

    events = AuditRepository(session_factory).list(
        entity_type="dataset", entity_id=up["id"]
    )
    upload_events = [e for e in events if e.action == "dataset.upload"]
    assert len(upload_events) == 1
    assert "file_path" not in upload_events[0].payload_json

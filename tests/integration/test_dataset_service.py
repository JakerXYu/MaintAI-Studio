"""Integration tests for the Dataset application service.

These exercise the full upload → profile → task flow against an in-memory
SQLite database (StaticPool) with files stored under ``tmp_path``.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from maintai.application.datasets import (
    DatasetNotFoundError,
    DatasetService,
    DuplicateDatasetError,
    StorageError,
)
from maintai.audit.repository import AuditRepository
from maintai.audit.service import AuditService
from maintai.data.ingest import (
    DataParseError,
    EmptyFilenameError,
    FileTooLargeError,
    UnsafeFilenameError,
    UnsupportedExtensionError,
)
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
def audit_repository(session_factory):
    return AuditRepository(session_factory)


@pytest.fixture()
def service(session_factory, tmp_path):
    audit = AuditService(AuditRepository(session_factory))
    return DatasetService(
        repository=DatasetRepository(session_factory),
        session_factory=session_factory,
        audit=audit,
        storage_root=tmp_path,
    )


def test_upload_csv_stores_controlled_key(service, tmp_path):
    result = service.upload(CSV_BYTES, "plant_sensor_log.csv")

    assert result["id"]
    assert result["original_filename"] == "plant_sensor_log.csv"
    assert result["source_type"] == "csv"
    assert result["file_path"] == f"{result['id']}.csv"
    assert "plant_sensor_log" not in result["file_path"]
    assert result["size_bytes"] == len(CSV_BYTES)
    assert result["status"] == "uploaded"
    assert result["row_count"] == 8
    assert result["column_count"] == 5
    assert result["schema"]["columns"]

    stored = tmp_path / result["file_path"]
    assert stored.is_file()
    assert stored.resolve().is_relative_to(tmp_path.resolve())
    assert stored.read_bytes() == CSV_BYTES


def test_upload_parquet_and_profile(service):
    data = _parquet_bytes()
    uploaded = service.upload(data, "sensor_data.parquet")

    assert uploaded["source_type"] == "parquet"
    assert uploaded["file_path"] == f"{uploaded['id']}.parquet"

    profiled = service.profile(uploaded["id"])
    assert profiled["status"] == "profiled"
    assert profiled["quality_score"] is not None
    assert set(profiled["profile"].keys()) == {"profile", "quality"}


def test_duplicate_hash_rejected(service):
    service.upload(CSV_BYTES, "one.csv")
    with pytest.raises(DuplicateDatasetError):
        service.upload(CSV_BYTES, "two.csv")


def test_upload_profile_task_flow(service):
    uploaded = service.upload(CSV_BYTES, "plant_sensor_log.csv")
    dataset_id = uploaded["id"]

    profiled = service.profile(dataset_id)
    assert profiled["status"] == "profiled"

    result = service.recommend_task(
        dataset_id,
        target_column="failure",
        asset_id_column="asset_id",
        timestamp_column="timestamp",
    )
    assert result["dataset_id"] == dataset_id
    assert result["task"]["recommended_task"] == "binary_classification"
    assert result["task"]["trainable"] is True
    assert result["leakage"]["verdict"] == "block"
    assert result["leakage"]["excluded_features"] == ["serial_no"]
    assert isinstance(result["quality_score"], float)

    quality = service.get_quality(dataset_id)
    assert quality["score"] == result["quality_score"]
    assert quality["report"] is not None

    fetched = service.get(dataset_id)
    assert fetched["target_column"] == "failure"
    assert fetched["asset_id_column"] == "asset_id"
    assert fetched["timestamp_column"] == "timestamp"
    assert fetched["task_type"] == "binary_classification"
    assert fetched["profile"]["leakage_report"]["excluded_features"] == ["serial_no"]

    listed = service.list()
    assert [item["id"] for item in listed] == [dataset_id]


def test_leakage_excludes_identifier_like_column(service):
    uploaded = service.upload(CSV_BYTES, "data.csv")
    result = service.recommend_task(uploaded["id"], "failure", "asset_id", "timestamp")
    assert "serial_no" in result["leakage"]["excluded_features"]
    # asset_id is a declared identity column, so it is not a leakage feature.
    assert "asset_id" not in result["leakage"]["excluded_features"]


def test_audit_records_allowlisted_metadata_only(service, audit_repository, tmp_path):
    uploaded = service.upload(CSV_BYTES, "plant_sensor_log.csv")
    events = audit_repository.list(entity_type="dataset", entity_id=uploaded["id"])

    upload_events = [e for e in events if e.action == "dataset.upload"]
    assert len(upload_events) == 1
    payload = upload_events[0].payload_json
    assert payload == {
        "dataset_id": uploaded["id"],
        "filename": "plant_sensor_log.csv",
        "source_type": "csv",
        "size_bytes": len(CSV_BYTES),
        "row_count": 8,
        "column_count": 5,
    }
    assert str(tmp_path) not in str(payload)
    assert "file_path" not in payload


def test_get_and_profile_not_found(service):
    with pytest.raises(DatasetNotFoundError):
        service.get("does-not-exist")
    with pytest.raises(DatasetNotFoundError):
        service.profile("does-not-exist")
    with pytest.raises(DatasetNotFoundError):
        service.recommend_task("does-not-exist", "failure")


def test_missing_storage_raises_storage_error(service, tmp_path):
    uploaded = service.upload(CSV_BYTES, "data.csv")
    (tmp_path / uploaded["file_path"]).unlink()
    with pytest.raises(StorageError):
        service.profile(uploaded["id"])


def test_corrupt_storage_raises_data_error(service, tmp_path):
    uploaded = service.upload(_parquet_bytes(), "data.parquet")
    (tmp_path / uploaded["file_path"]).write_bytes(b"not parquet")
    with pytest.raises(DataParseError):
        service.profile(uploaded["id"])


def test_invalid_inputs_propagate_stable_errors(service, session_factory, tmp_path):
    with pytest.raises(EmptyFilenameError):
        service.upload(b"", "")

    with pytest.raises(UnsupportedExtensionError):
        service.upload(b"x", "file.txt")

    with pytest.raises(DataParseError):
        service.upload(b"not parquet", "x.parquet")

    tiny = DatasetService(
        repository=DatasetRepository(session_factory),
        session_factory=session_factory,
        audit=AuditService(AuditRepository(session_factory)),
        storage_root=tmp_path,
        max_upload_bytes=4,
    )
    with pytest.raises(FileTooLargeError):
        tiny.upload(b"12345", "a.csv")


def test_upload_cleans_file_on_db_failure(service, tmp_path, monkeypatch):
    def boom(dataset, *, session=None):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(service._repository, "create", boom)
    with pytest.raises(RuntimeError):
        service.upload(CSV_BYTES, "data.csv")

    assert list(tmp_path.iterdir()) == []


def test_upload_rolls_back_database_and_file_when_audit_fails(
    service,
    tmp_path,
    monkeypatch,
):
    def boom(**kwargs):
        raise RuntimeError("audit unavailable")

    monkeypatch.setattr(service._audit, "record", boom)
    with pytest.raises(RuntimeError, match="audit unavailable"):
        service.upload(CSV_BYTES, "data.csv")

    assert service.list() == []
    assert list(tmp_path.iterdir()) == []


def test_long_filename_is_rejected_before_database_write(service):
    filename = f"{'a' * 252}.csv"
    with pytest.raises(UnsafeFilenameError, match="255"):
        service.upload(CSV_BYTES, filename)

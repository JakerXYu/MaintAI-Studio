"""Ingestion tests (synthetic data only)."""

import hashlib
import io

import pandas as pd
import pytest

from maintai.data.ingest import (
    DataParseError,
    EmptyFilenameError,
    FileTooLargeError,
    UnsafeFilenameError,
    UnsupportedExtensionError,
    ingest_bytes,
    sanitize_filename,
    sha256_bytes,
)


def _frame():
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    frame.to_csv(buf, index=False)
    return buf.getvalue()


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)
    return buf.getvalue()


def test_csv_roundtrip_and_hash():
    frame = _frame()
    data = _csv_bytes(frame)
    result = ingest_bytes(data, "sample.csv")
    assert result.source_type == "csv"
    assert result.sha256 == hashlib.sha256(data).hexdigest()
    pd.testing.assert_frame_equal(result.frame, frame)


def test_parquet_roundtrip():
    frame = _frame()
    data = _parquet_bytes(frame)
    result = ingest_bytes(data, "sample.parquet")
    assert result.source_type == "parquet"
    assert result.sha256 == hashlib.sha256(data).hexdigest()
    pd.testing.assert_frame_equal(result.frame, frame)


def test_unsupported_extension_rejected():
    with pytest.raises(UnsupportedExtensionError):
        ingest_bytes(b"1,2\n", "data.xlsx")


def test_missing_extension_rejected():
    with pytest.raises(UnsupportedExtensionError):
        ingest_bytes(b"1,2\n", "data")


def test_oversize_rejected():
    with pytest.raises(FileTooLargeError):
        ingest_bytes(b"12345678901", "data.csv", max_bytes=10)


def test_parsed_row_and_column_limits_are_enforced():
    data = _csv_bytes(_frame())
    with pytest.raises(FileTooLargeError):
        ingest_bytes(data, "data.csv", max_rows=2)
    with pytest.raises(FileTooLargeError):
        ingest_bytes(data, "data.csv", max_columns=1)


def test_parquet_metadata_row_limit_is_enforced():
    data = _parquet_bytes(_frame())
    with pytest.raises(FileTooLargeError):
        ingest_bytes(data, "data.parquet", max_rows=2)


def test_path_traversal_rejected():
    for bad in ("../evil.csv", "a/b.csv", "a\\b.csv", "C:\\evil.csv", ".."):
        with pytest.raises(UnsafeFilenameError):
            sanitize_filename(bad)


def test_empty_filename_rejected():
    for bad in ("", "   "):
        with pytest.raises(EmptyFilenameError):
            sanitize_filename(bad)


def test_none_filename_rejected():
    with pytest.raises(EmptyFilenameError):
        sanitize_filename(None)


def test_reserved_windows_name_rejected():
    with pytest.raises(UnsafeFilenameError):
        sanitize_filename("CON.csv")


def test_sanitize_returns_plain_name():
    assert sanitize_filename("  data.csv  ") == "data.csv"


def test_invalid_parquet_raises_stable_parse_error():
    with pytest.raises(DataParseError) as error:
        ingest_bytes(b"this is not a parquet file", "bad.parquet")
    assert "Arrow" not in str(error.value)


def test_sha256_is_reproducible():
    data = b"payload"
    assert sha256_bytes(data) == sha256_bytes(data)

"""Deterministic CSV / Parquet ingestion.

Uploads are validated in memory (extension allowlist, safe filename, size cap)
and hashed before being parsed to a DataFrame. All failures raise stable
``DataIngestError`` subclasses so callers (and the API layer) can map them to
4xx responses without leaking arbitrary paths or internal details.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath

import pandas as pd

ALLOWED_EXTENSIONS: tuple[str, ...] = (".csv", ".parquet")
DEFAULT_MAX_UPLOAD_MB: int = 200
DEFAULT_MAX_ROWS: int = 5_000_000
DEFAULT_MAX_COLUMNS: int = 1_000
DEFAULT_MAX_MEMORY_MB: int = 2_048
BYTES_PER_MB: int = 1024 * 1024
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


class DataIngestError(Exception):
    """Base class for deterministic ingestion errors."""


class EmptyFilenameError(DataIngestError):
    """Raised when the upload filename is empty or blank."""


class UnsafeFilenameError(DataIngestError):
    """Raised when the filename contains a path traversal or reserved name."""


class UnsupportedExtensionError(DataIngestError):
    """Raised when the filename extension is not on the allowlist."""


class FileTooLargeError(DataIngestError):
    """Raised when the upload exceeds the configured byte limit."""


class DataParseError(DataIngestError):
    """Raised when file bytes cannot be parsed into a DataFrame."""


def sanitize_filename(name: str | None) -> str:
    """Return a safe, plain filename or raise a stable ``DataIngestError``.

    Rejects empty names, path separators (``/`` and ``\\``), ``..`` traversal,
    drive/stream separators, control characters, and Windows reserved device
    names. No arbitrary path is ever accepted.
    """
    if name is None:
        raise EmptyFilenameError("filename is required")
    stripped = name.strip()
    if not stripped:
        raise EmptyFilenameError("filename must not be empty")
    if any(char in stripped for char in ("\x00", "\n", "\r", "/", "\\")):
        raise UnsafeFilenameError(
            "filename must be a plain name without directories or control characters"
        )
    if stripped in {".", ".."} or stripped.startswith(".."):
        raise UnsafeFilenameError("filename must not be a path traversal")
    if ":" in stripped:
        raise UnsafeFilenameError("filename must not contain a drive or stream separator")
    posix_name = PurePosixPath(stripped).name
    windows_name = PureWindowsPath(stripped).name
    if stripped not in (posix_name, windows_name):
        raise UnsafeFilenameError("filename must be a plain name without directories")
    stem = stripped.rsplit(".", 1)[0].upper()
    if stem in _WINDOWS_RESERVED_NAMES:
        raise UnsafeFilenameError("filename uses a reserved device name")
    return stripped


def _extension_of(filename: str) -> str:
    return PurePosixPath(filename).suffix.lower()


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of the raw upload bytes."""
    return hashlib.sha256(data).hexdigest()


def validate_size(size_bytes: int, max_bytes: int) -> None:
    """Raise ``FileTooLargeError`` when ``size_bytes`` exceeds the cap."""
    if size_bytes > max_bytes:
        raise FileTooLargeError(
            f"upload is {size_bytes} bytes, exceeding the {max_bytes} byte limit"
        )


def _validate_frame_limits(
    frame: pd.DataFrame,
    *,
    max_rows: int,
    max_columns: int,
    max_memory_bytes: int,
) -> None:
    if len(frame) > max_rows:
        raise FileTooLargeError(f"dataset exceeds the {max_rows} row limit")
    if len(frame.columns) > max_columns:
        raise FileTooLargeError(f"dataset exceeds the {max_columns} column limit")
    memory_bytes = int(frame.memory_usage(index=True, deep=True).sum())
    if memory_bytes > max_memory_bytes:
        raise FileTooLargeError(
            f"parsed dataset exceeds the {max_memory_bytes} byte memory limit"
        )


def read_bytes_to_dataframe(
    data: bytes,
    filename: str,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_columns: int = DEFAULT_MAX_COLUMNS,
    max_memory_bytes: int = DEFAULT_MAX_MEMORY_MB * BYTES_PER_MB,
) -> pd.DataFrame:
    """Parse raw bytes into a DataFrame based on the allowlisted extension."""
    ext = _extension_of(filename)
    buffer = io.BytesIO(data)
    try:
        if ext == ".csv":
            frame = pd.read_csv(buffer)
        elif ext == ".parquet":
            import pyarrow.parquet as pq

            metadata = pq.ParquetFile(buffer).metadata
            if metadata.num_rows > max_rows:
                raise FileTooLargeError(f"dataset exceeds the {max_rows} row limit")
            if metadata.num_columns > max_columns:
                raise FileTooLargeError(f"dataset exceeds the {max_columns} column limit")
            uncompressed_bytes = sum(
                metadata.row_group(i).total_byte_size
                for i in range(metadata.num_row_groups)
            )
            if uncompressed_bytes > max_memory_bytes:
                raise FileTooLargeError(
                    f"Parquet metadata exceeds the {max_memory_bytes} byte memory limit"
                )
            buffer.seek(0)
            frame = pd.read_parquet(buffer)
        else:
            raise UnsupportedExtensionError(
                f"unsupported extension {ext or '<none>'!r}; allowed: {ALLOWED_EXTENSIONS}"
            )
    except DataIngestError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize parser failures to a stable type
        raise DataParseError(
            f"failed to parse {filename!r}: invalid or unsupported {ext.lstrip('.')} content"
        ) from exc
    if not isinstance(frame, pd.DataFrame):
        raise DataParseError(f"parser returned no tabular data for {filename!r}")
    _validate_frame_limits(
        frame,
        max_rows=max_rows,
        max_columns=max_columns,
        max_memory_bytes=max_memory_bytes,
    )
    return frame


@dataclass(frozen=True)
class IngestedDataset:
    """Validated, hashed upload plus its parsed DataFrame."""

    name: str
    source_type: str
    sha256: str
    size_bytes: int
    frame: pd.DataFrame


def ingest_bytes(
    data: bytes,
    filename: str,
    max_bytes: int = DEFAULT_MAX_UPLOAD_MB * BYTES_PER_MB,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_columns: int = DEFAULT_MAX_COLUMNS,
    max_memory_bytes: int = DEFAULT_MAX_MEMORY_MB * BYTES_PER_MB,
) -> IngestedDataset:
    """Validate, hash, and parse uploaded bytes into an ``IngestedDataset``."""
    name = sanitize_filename(filename)
    ext = _extension_of(name)
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedExtensionError(
            f"unsupported extension {ext or '<none>'!r}; allowed: {ALLOWED_EXTENSIONS}"
        )
    validate_size(len(data), max_bytes)
    frame = read_bytes_to_dataframe(
        data,
        name,
        max_rows=max_rows,
        max_columns=max_columns,
        max_memory_bytes=max_memory_bytes,
    )
    return IngestedDataset(
        name=name,
        source_type=ext.lstrip("."),
        sha256=sha256_bytes(data),
        size_bytes=len(data),
        frame=frame,
    )

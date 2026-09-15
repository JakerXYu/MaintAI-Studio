"""Bounded Scania APS (Air Pressure System) failure dataset adapter.

This module is deliberately narrow: it records the official UCI metadata,
downloads/extracts the archive with stdlib only, parses the raw CSV into a
deterministic numeric frame, and exposes the official train/test holdout. It
does **not** guess feature meaning, train models, or evaluate anything.

Raw file shape (UCI dataset 421, "APS Failure at Scania Trucks"):

* ``aps_failure_training_set.csv`` — 60,000 rows
* ``aps_failure_test_set.csv``     — 16,000 rows
* archive header ``class,<170 anonymised features>``; missing values are ``na``
* class labels: ``neg`` (no APS failure) / ``pos`` (APS failure)

The UCI web listing reports 171 features, while the downloadable archive has
171 columns total (``class`` plus 170 features). The archive is the executable
source of truth; this discrepancy is preserved in metadata rather than filled
with a fabricated column. The target is normalised to an integer column
``aps_failure`` (neg=0, pos=1), and all 170 feature names stay untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from maintai.data.schemas import SplitResult

# --- provenance / metadata ---------------------------------------------------

SCANIA_APS_URL = (
    "https://archive.ics.uci.edu/static/public/421/"
    "aps+failure+at+scania+trucks.zip"
)
SCANIA_APS_PAGE = (
    "https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks"
)
SCANIA_APS_DOI = "10.24432/C51S51"
SCANIA_APS_CREATOR = "Scania CV AB"
SCANIA_APS_LICENSE = (
    "CC BY 4.0 (per UCI repository listing); the archive's description file "
    "also carries a GPL-3.0-or-later notice (© 2016 Scania CV AB)"
)
SCANIA_APS_CITATION = (
    "APS Failure at Scania Trucks [Dataset]. (2016). UCI Machine Learning "
    "Repository. https://doi.org/10.24432/C51S51."
)

SOURCE_METADATA: dict[str, str | int] = {
    "name": "scania_aps",
    "url": SCANIA_APS_URL,
    "page": SCANIA_APS_PAGE,
    "doi": SCANIA_APS_DOI,
    "creator": SCANIA_APS_CREATOR,
    "license": SCANIA_APS_LICENSE,
    "citation": SCANIA_APS_CITATION,
    "uci_listing_feature_count": 171,
    "archive_feature_count": 170,
    "archive_sha256": "5504d0402f54faaf97ac0ca085a621645763f5cfea2eb29c592b057d43d4db89",
}

# --- official schema facts ----------------------------------------------------

TARGET_COLUMN = "aps_failure"
CLASS_COLUMN = "class"
HEADER_PREFIX = "class,"
MISSING_TOKEN = "na"
POSITIVE_LABEL = "pos"
NEGATIVE_LABEL = "neg"
LABEL_ENCODING = {NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}

N_FEATURES = 170
TRAIN_ROWS = 60_000
TEST_ROWS = 16_000
OFFICIAL_ROW_COUNTS = (TRAIN_ROWS, TEST_ROWS)

# Official IDA 2016 challenge cost metric: FP (unnecessary check) = 10,
# FN (missed faulty truck) = 500. These are the challenge's economics, not ours.
FP_COST = 10.0
FN_COST = 500.0

HEADER_SEARCH_LINES = 100

TRAIN_MEMBER = "aps_failure_training_set.csv"
TEST_MEMBER = "aps_failure_test_set.csv"
DESCRIPTION_MEMBER = "aps_failure_description.txt"
ZIP_MEMBERS: tuple[str, ...] = (TRAIN_MEMBER, TEST_MEMBER)
ZIP_FILENAME = "aps_failure_at_scania_trucks.zip"
ARCHIVE_SHA256 = str(SOURCE_METADATA["archive_sha256"])
MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024

PREPARED_TRAIN_FILENAME = "aps_failure_train.csv"
PREPARED_TEST_FILENAME = "aps_failure_test.csv"
SUMMARY_FILENAME = "dataset_summary.json"


class ScaniaApsError(Exception):
    """Base class for Scania APS adapter errors."""


class ScaniaApsParseError(ScaniaApsError):
    """Raised when a raw Scania APS CSV cannot be parsed or validated."""


class ScaniaApsDownloadError(ScaniaApsError):
    """Raised when the archive cannot be downloaded (see manual fallback)."""


class UnsafeZipMemberError(ScaniaApsError):
    """Raised when a ZIP member name is unsafe (traversal / absolute path)."""


def project_root() -> Path:
    """Locate the repo root by walking up to ``configs`` or ``pyproject.toml``."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs").is_dir() or (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[2]


RAW_DIR = project_root() / "data" / "raw" / "scania_aps"
PREPARED_DIR = RAW_DIR / "prepared"


# --- parsing -------------------------------------------------------------------

def _locate_header_index(path: Path) -> int:
    """Return the 0-based line index of the ``class,`` header (first 100 lines)."""
    with path.open("r", encoding="utf-8-sig", errors="strict") as handle:
        for index in range(HEADER_SEARCH_LINES):
            line = handle.readline()
            if line == "":
                break
            if line.startswith(HEADER_PREFIX):
                return index
    raise ScaniaApsParseError(
        f"header line starting with {HEADER_PREFIX!r} not found within the first "
        f"{HEADER_SEARCH_LINES} lines of {path.name!r}"
    )


def _encode_target(values: pd.Series) -> pd.Series:
    """Map ``neg``/``pos`` to 0/1, rejecting any other label or missing value."""
    if values.isna().any():
        raise ScaniaApsParseError(
            "class column contains missing values; only 'neg'/'pos' are valid"
        )
    as_str = values.astype(str)
    unexpected = sorted(set(as_str) - {NEGATIVE_LABEL, POSITIVE_LABEL})
    if unexpected:
        raise ScaniaApsParseError(
            f"unexpected class label(s) {unexpected}; only "
            f"{NEGATIVE_LABEL!r}/{POSITIVE_LABEL!r} are valid"
        )
    return as_str.map(LABEL_ENCODING).astype("int64")


def read_scania_aps_csv(
    path: Path | str,
    *,
    expected_rows: int | None = None,
) -> pd.DataFrame:
    """Parse a raw Scania APS CSV into a normalised numeric frame.

    Returns a frame with an integer ``aps_failure`` target (neg=0, pos=1)
    followed by the 170 anonymised feature columns (names untouched; ``na`` is
    read as NaN).

    ``expected_rows`` overrides the default official-count validation; when it
    is ``None`` the row count must be exactly 60,000 (train) or 16,000 (test).
    """
    path = Path(path)
    try:
        header_index = _locate_header_index(path)
        frame = pd.read_csv(
            path,
            skiprows=header_index,
            na_values=[MISSING_TOKEN],
            encoding="utf-8-sig",
        )
    except ScaniaApsError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize parser failures
        raise ScaniaApsParseError(f"failed to parse {path.name!r}: {exc}") from exc

    if len(frame.columns) != N_FEATURES + 1:
        raise ScaniaApsParseError(
            f"expected {N_FEATURES + 1} columns (class + {N_FEATURES} features), "
            f"got {len(frame.columns)}"
        )
    if frame.columns[0] != CLASS_COLUMN:
        raise ScaniaApsParseError(
            f"expected first column {CLASS_COLUMN!r}, got {frame.columns[0]!r}"
        )

    target = _encode_target(frame[CLASS_COLUMN])
    out = frame[list(frame.columns[1:])].copy()
    out.insert(0, TARGET_COLUMN, target)

    count = len(out)
    if expected_rows is not None:
        if count != expected_rows:
            raise ScaniaApsParseError(f"expected {expected_rows} rows, got {count}")
    elif count not in OFFICIAL_ROW_COUNTS:
        raise ScaniaApsParseError(
            f"expected an official Scania APS row count "
            f"({TRAIN_ROWS} train / {TEST_ROWS} test), got {count}"
        )
    return out


# --- official holdout split ----------------------------------------------------

def official_split(frame: pd.DataFrame, *, train_rows: int = TRAIN_ROWS) -> SplitResult:
    """Return the official UCI holdout split for a concatenated ``[train; test]`` frame.

    The first ``train_rows`` rows are train and the remaining rows are test.
    There is no resampling, no stratification, and no validation fold, so this
    works unchanged for unit fixtures (it never re-validates official counts).
    """
    count = len(frame)
    if not 0 < train_rows < count:
        raise ScaniaApsError(
            f"official split needs 0 < train_rows < {count} rows, got train_rows={train_rows}"
        )
    return SplitResult(
        strategy="official_holdout",
        train_indices=list(range(train_rows)),
        test_indices=list(range(train_rows, count)),
        validation_indices=[],
        evidence=[
            "official UCI holdout: first rows = training set, remaining rows = test set; "
            "no resampling and no validation fold"
        ],
        params={"train_rows": train_rows, "test_rows": count - train_rows},
    )


# --- download / extraction -------------------------------------------------------

def _manual_download_hint(dest: Path) -> str:
    return (
        f"Download the archive manually from {SCANIA_APS_URL} "
        f"and place it at {dest}"
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_zip(
    url: str = SCANIA_APS_URL,
    dest_dir: Path | str = RAW_DIR,
    *,
    filename: str = ZIP_FILENAME,
    timeout: float = 120.0,
    expected_sha256: str | None = ARCHIVE_SHA256,
) -> Path:
    """Download the archive to ``dest_dir`` with an atomic temp-file write.

    The bytes stream to a ``.part`` temp file in the destination directory and
    are moved into place with ``os.replace``, so a partial download is never
    observed under the final name. On failure the temp file is removed and a
    ``ScaniaApsDownloadError`` with a manual-download fallback is raised.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / filename
    if final.exists():
        if expected_sha256 and _sha256_path(final) != expected_sha256.lower():
            raise ScaniaApsDownloadError(
                f"existing archive SHA-256 mismatch; {_manual_download_hint(final)}"
            )
        return final

    fd, tmp_name = tempfile.mkstemp(
        dir=dest_dir, prefix=f".{filename}.", suffix=".part"
    )
    tmp = Path(tmp_name)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, os.fdopen(
            fd, "wb"
        ) as out:
            fd = -1  # ownership moved to the file object
            shutil.copyfileobj(response, out, length=1024 * 1024)
        os.replace(tmp, final)
        if expected_sha256 and _sha256_path(final) != expected_sha256.lower():
            final.unlink(missing_ok=True)
            raise ScaniaApsDownloadError(
                f"downloaded archive SHA-256 mismatch; {_manual_download_hint(final)}"
            )
        return final
    except Exception as exc:  # noqa: BLE001 - normalize network/write failures
        if fd != -1:
            os.close(fd)
        if tmp.exists():
            tmp.unlink()
        raise ScaniaApsDownloadError(f"{_manual_download_hint(final)}: {exc}") from exc


def _is_safe_member_name(name: str) -> bool:
    """Return True when a ZIP member name is safe to extract (no traversal)."""
    if not name or "\x00" in name:
        return False
    if name.startswith(("/", "\\")) or (len(name) >= 2 and name[1] == ":"):
        return False
    return all(part not in {"", ".", ".."} for part in re.split(r"[/\\]", name))


def extract_zip_members(
    zip_path: Path | str,
    dest_dir: Path | str,
    *,
    members: tuple[str, ...] = ZIP_MEMBERS,
    max_uncompressed_bytes: int = MAX_UNCOMPRESSED_BYTES,
) -> dict[str, Path]:
    """Extract exactly the requested members, rejecting any unsafe names.

    Every member name in the archive is checked first (absolute paths, drive
    paths and ``..`` traversal are rejected). Only the names in ``members`` are
    written out; other files (e.g. the description text) are left in the
    archive. Returns a mapping of member name to extracted path.
    """
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        requested_size = sum(
            info.file_size for info in archive.infolist() if info.filename in members
        )
        if requested_size > max_uncompressed_bytes:
            raise UnsafeZipMemberError(
                f"requested ZIP members expand to {requested_size} bytes; "
                f"limit is {max_uncompressed_bytes}"
            )
        for info in archive.infolist():
            if not _is_safe_member_name(info.filename):
                raise UnsafeZipMemberError(
                    f"unsafe ZIP member {info.filename!r}; refusing to extract"
                )
        names = archive.namelist()
        extracted: dict[str, Path] = {}
        for name in members:
            if name not in names:
                raise ScaniaApsError(
                    f"expected ZIP member {name!r} is missing from {zip_path.name!r}"
                )
            target = dest_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
            extracted[name] = target
    return extracted


# --- preparation -----------------------------------------------------------------

def _to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def build_split_summary(frame: pd.DataFrame, csv_bytes: bytes, path: str) -> dict:
    """Return the per-split summary (SHA-256, shape, classes, missingness)."""
    neg = int((frame[TARGET_COLUMN] == 0).sum())
    pos = int((frame[TARGET_COLUMN] == 1).sum())
    cells = int(frame.size)
    missing = int(frame.isna().sum().sum())
    return {
        "path": path,
        "sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "shape": [int(len(frame)), int(len(frame.columns))],
        "class_distribution": {NEGATIVE_LABEL: neg, POSITIVE_LABEL: pos},
        "missingness": {
            "cells": cells,
            "missing": missing,
            "rate": (missing / cells) if cells else 0.0,
        },
    }


def write_prepared_dataset(
    train: pd.DataFrame,
    test: pd.DataFrame,
    prepared_dir: Path | str = PREPARED_DIR,
    *,
    source: dict | None = None,
    prepared_at: str | None = None,
) -> dict:
    """Write normalised train/test CSVs and ``dataset_summary.json``.

    Returns the summary dict that was serialised. Missing values are written as
    empty CSV cells (the standard pandas NaN representation).
    """
    prepared_dir = Path(prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    train_bytes = _to_csv_bytes(train)
    test_bytes = _to_csv_bytes(test)
    train_path = prepared_dir / PREPARED_TRAIN_FILENAME
    test_path = prepared_dir / PREPARED_TEST_FILENAME
    train_path.write_bytes(train_bytes)
    test_path.write_bytes(test_bytes)

    summary = {
        "dataset": "scania_aps",
        "source": source if source is not None else SOURCE_METADATA,
        "target": TARGET_COLUMN,
        "label_encoding": LABEL_ENCODING,
        "n_features": N_FEATURES,
        "prepared_at": (
            prepared_at if prepared_at is not None else datetime.now(UTC).isoformat()
        ),
        "splits": {
            "train": build_split_summary(train, train_bytes, str(train_path.resolve())),
            "test": build_split_summary(test, test_bytes, str(test_path.resolve())),
        },
    }
    summary_path = prepared_dir / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary

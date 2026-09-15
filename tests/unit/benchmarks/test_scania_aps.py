"""Scania APS adapter tests (synthetic fixtures only)."""

import hashlib
import json
import zipfile

import pandas as pd
import pytest

from maintai.benchmarks import scania_aps
from maintai.benchmarks.scania_aps import (
    N_FEATURES,
    ScaniaApsDownloadError,
    ScaniaApsError,
    ScaniaApsParseError,
    UnsafeZipMemberError,
    _is_safe_member_name,
    download_zip,
    extract_zip_members,
    official_split,
    read_scania_aps_csv,
    write_prepared_dataset,
)


def _feature_names(n: int = N_FEATURES) -> list[str]:
    return [f"f{i:03d}" for i in range(n)]


def _aps_csv(
    labels: list[str],
    n_features: int = N_FEATURES,
    *,
    missing: tuple[int, int] | None = None,
) -> str:
    """Build a minimal APS-style CSV. ``missing`` is a ``(row, col)`` to set to ``na``."""
    header = "class," + ",".join(_feature_names(n_features))
    lines = [header]
    for i, label in enumerate(labels):
        values = [str(i + 1)] * n_features
        if missing is not None and missing[0] == i:
            values[missing[1]] = "na"
        lines.append(",".join([label, *values]))
    return "\n".join(lines) + "\n"


def _write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _write_zip(tmp_path, members: dict[str, str], name: str = "a.zip"):
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as archive:
        for member, content in members.items():
            archive.writestr(member, content)
    return path


# --- metadata -------------------------------------------------------------------


def test_metadata_is_recorded():
    assert scania_aps.SCANIA_APS_URL == (
        "https://archive.ics.uci.edu/static/public/421/"
        "aps+failure+at+scania+trucks.zip"
    )
    assert scania_aps.SCANIA_APS_DOI == "10.24432/C51S51"
    assert "CC BY 4.0" in scania_aps.SCANIA_APS_LICENSE
    assert "10.24432/C51S51" in scania_aps.SCANIA_APS_CITATION
    assert scania_aps.SOURCE_METADATA["creator"] == "Scania CV AB"
    assert scania_aps.SOURCE_METADATA["archive_sha256"] == scania_aps.ARCHIVE_SHA256


# --- parsing --------------------------------------------------------------------


def test_na_becomes_missing(tmp_path):
    path = _write(tmp_path, "t.csv", _aps_csv(["neg", "pos", "neg"], missing=(1, 3)))
    frame = read_scania_aps_csv(path, expected_rows=3)
    # feature f003 lives at output column index 4 (aps_failure is column 0).
    assert pd.isna(frame.iloc[1, 4])
    assert not frame.iloc[0, 4:].isna().any()
    assert frame.iloc[:, 1:].dtypes.apply(lambda d: d.kind).isin(["i", "f"]).all()


def test_label_conversion_neg_pos(tmp_path):
    path = _write(tmp_path, "t.csv", _aps_csv(["neg", "pos", "pos", "neg"]))
    frame = read_scania_aps_csv(path, expected_rows=4)
    assert list(frame["aps_failure"]) == [0, 1, 1, 0]
    assert frame["aps_failure"].dtype == "int64"
    assert "class" not in frame.columns


def test_unexpected_label_rejected(tmp_path):
    path = _write(tmp_path, "t.csv", _aps_csv(["neg", "maybe", "pos"]))
    with pytest.raises(ScaniaApsParseError, match="unexpected class label"):
        read_scania_aps_csv(path, expected_rows=3)


def test_missing_target_label_rejected(tmp_path):
    path = _write(tmp_path, "t.csv", _aps_csv(["neg", "na", "pos"]))
    with pytest.raises(ScaniaApsParseError, match="missing values"):
        read_scania_aps_csv(path, expected_rows=3)


def test_feature_count_and_names_preserved(tmp_path):
    path = _write(tmp_path, "t.csv", _aps_csv(["neg", "pos"]))
    frame = read_scania_aps_csv(path, expected_rows=2)
    assert len(frame.columns) == N_FEATURES + 1
    assert frame.columns[0] == "aps_failure"
    assert list(frame.columns[1:]) == _feature_names()


def test_wrong_feature_count_rejected(tmp_path):
    path = _write(tmp_path, "t.csv", _aps_csv(["neg", "pos"], n_features=169))
    with pytest.raises(ScaniaApsParseError, match="columns"):
        read_scania_aps_csv(path, expected_rows=2)


def test_non_official_row_count_rejected_by_default(tmp_path):
    path = _write(tmp_path, "t.csv", _aps_csv(["neg", "pos", "neg"]))
    with pytest.raises(ScaniaApsParseError, match="official Scania APS row count"):
        read_scania_aps_csv(path)


def test_expected_rows_override_allows_fixture(tmp_path):
    path = _write(tmp_path, "t.csv", _aps_csv(["neg", "pos", "neg"]))
    frame = read_scania_aps_csv(path, expected_rows=3)
    assert len(frame) == 3


def test_header_located_after_preamble(tmp_path):
    body = _aps_csv(["neg", "pos"])
    text = "metadata line 1\n# comment\n" + body
    path = _write(tmp_path, "t.csv", text)
    frame = read_scania_aps_csv(path, expected_rows=2)
    assert list(frame["aps_failure"]) == [0, 1]


def test_missing_header_rejected(tmp_path):
    path = _write(tmp_path, "t.csv", "\n".join(["garbage"] * 100) + "\n")
    with pytest.raises(ScaniaApsParseError, match="header line"):
        read_scania_aps_csv(path, expected_rows=2)


# --- official split -------------------------------------------------------------


def test_official_split_preserves_order_and_no_overlap():
    frame = pd.DataFrame({"aps_failure": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]})
    result = official_split(frame, train_rows=6)
    assert result.strategy == "official_holdout"
    assert result.train_indices == list(range(6))
    assert result.test_indices == list(range(6, 10))
    assert result.validation_indices == []
    assert set(result.train_indices).isdisjoint(result.test_indices)


def test_official_split_rejects_bad_train_rows():
    frame = pd.DataFrame({"aps_failure": [0, 1, 0]})
    with pytest.raises(ScaniaApsError):
        official_split(frame, train_rows=0)
    with pytest.raises(ScaniaApsError):
        official_split(frame, train_rows=3)


# --- preparation / summary -------------------------------------------------------


def test_summary_serialization(tmp_path):
    train = pd.DataFrame(
        {"aps_failure": [0, 1, 0], **{name: [1.0, 2.0, 3.0] for name in _feature_names()}}
    )
    train.iloc[2, 5] = float("nan")  # exactly one missing cell

    test = pd.DataFrame(
        {"aps_failure": [1, 0], **{name: [3.0, 4.0] for name in _feature_names()}}
    )

    prepared = tmp_path / "prepared"
    summary = write_prepared_dataset(
        train, test, prepared, prepared_at="2026-01-01T00:00:00+00:00"
    )

    assert summary["dataset"] == "scania_aps"
    assert summary["prepared_at"] == "2026-01-01T00:00:00+00:00"
    assert summary["n_features"] == N_FEATURES
    assert summary["label_encoding"] == {"neg": 0, "pos": 1}

    assert summary["splits"]["train"]["shape"] == [3, N_FEATURES + 1]
    assert summary["splits"]["test"]["shape"] == [2, N_FEATURES + 1]
    assert summary["splits"]["train"]["class_distribution"] == {"neg": 2, "pos": 1}
    assert summary["splits"]["test"]["class_distribution"] == {"neg": 1, "pos": 1}

    train_bytes = (prepared / "aps_failure_train.csv").read_bytes()
    assert summary["splits"]["train"]["sha256"] == hashlib.sha256(train_bytes).hexdigest()

    train_missing = summary["splits"]["train"]["missingness"]
    assert train_missing["missing"] == 1
    assert train_missing["cells"] == 3 * (N_FEATURES + 1)
    assert train_missing["rate"] == pytest.approx(1 / (3 * (N_FEATURES + 1)))
    assert summary["splits"]["test"]["missingness"]["missing"] == 0

    payload = json.loads((prepared / "dataset_summary.json").read_text(encoding="utf-8"))
    assert payload == summary


# --- ZIP extraction ---------------------------------------------------------------


def test_existing_archive_sha256_is_verified(tmp_path):
    archive = tmp_path / "fixture.zip"
    archive.write_bytes(b"known archive bytes")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()

    assert download_zip(
        dest_dir=tmp_path, filename=archive.name, expected_sha256=digest
    ) == archive
    with pytest.raises(ScaniaApsDownloadError, match="SHA-256 mismatch"):
        download_zip(dest_dir=tmp_path, filename=archive.name, expected_sha256="0" * 64)


def test_extract_rejects_excessive_uncompressed_size(tmp_path):
    zip_path = _write_zip(
        tmp_path,
        {scania_aps.TRAIN_MEMBER: "x" * 60, scania_aps.TEST_MEMBER: "y" * 60},
    )
    with pytest.raises(UnsafeZipMemberError, match="expand"):
        extract_zip_members(zip_path, tmp_path / "out", max_uncompressed_bytes=100)


def test_unsafe_zip_member_rejected(tmp_path):
    zip_path = _write_zip(
        tmp_path, {scania_aps.TRAIN_MEMBER: "x", "../evil.csv": "y"}
    )
    with pytest.raises(UnsafeZipMemberError, match="unsafe ZIP member"):
        extract_zip_members(zip_path, tmp_path / "out")


def test_extract_exact_members_only(tmp_path):
    zip_path = _write_zip(
        tmp_path,
        {
            scania_aps.TRAIN_MEMBER: "a,b\n1,2\n",
            scania_aps.TEST_MEMBER: "a,b\n3,4\n",
            scania_aps.DESCRIPTION_MEMBER: "description",
        },
    )
    out = tmp_path / "out"
    extracted = extract_zip_members(zip_path, out)
    assert set(extracted) == {scania_aps.TRAIN_MEMBER, scania_aps.TEST_MEMBER}
    assert (out / scania_aps.TRAIN_MEMBER).is_file()
    assert (out / scania_aps.TEST_MEMBER).is_file()
    assert not (out / scania_aps.DESCRIPTION_MEMBER).exists()


def test_missing_member_rejected(tmp_path):
    zip_path = _write_zip(tmp_path, {scania_aps.TRAIN_MEMBER: "x"})
    with pytest.raises(ScaniaApsError, match="missing"):
        extract_zip_members(zip_path, tmp_path / "out")


@pytest.mark.parametrize(
    "bad",
    ["../evil.csv", "/abs.csv", "\\abs.csv", "a/../../x", "C:\\evil.csv", ".."],
)
def test_unsafe_member_names_rejected(bad):
    assert _is_safe_member_name(bad) is False


@pytest.mark.parametrize(
    "good",
    ["aps_failure_training_set.csv", "dir/file.csv"],
)
def test_safe_member_names_accepted(good):
    assert _is_safe_member_name(good) is True

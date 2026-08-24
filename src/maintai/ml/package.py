"""Deterministic sklearn pipeline artifact packaging (joblib + manifest).

Saves the complete pipeline plus a manifest (model name / version / task /
features / input schema / training summary / created time / SHA-256) in a ZIP
container. The JSON manifest and pipeline bytes are read without pickle, then
the pipeline checksum is verified before joblib deserialization.

Artifacts are written atomically (temp file → ``os.replace``) into a
caller-provided directory and loaded only from a controlled ``.joblib`` filename
inside that directory. Loading rejects path traversal, corrupt files, untrusted
extensions, unsupported manifest versions, and SHA-256 mismatches.

The artifact directory must remain app-private and trusted. SHA-256 detects
corruption; it is not source authentication and cannot make pickle safe against
an attacker who can replace artifacts and their manifests.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
from pydantic import BaseModel, Field

MANIFEST_VERSION = 1
_ARTIFACT_SUFFIX = ".joblib"
_CONTROLLED_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MANIFEST_MEMBER = "manifest.json"
_PIPELINE_MEMBER = "pipeline.joblib"


class ArtifactError(Exception):
    """Raised for any packaging / loading failure."""


class ArtifactManifest(BaseModel):
    """JSON-safe manifest stored alongside the serialised pipeline."""

    manifest_version: int = MANIFEST_VERSION
    model_name: str
    version: str
    task: str
    features: list[str] = Field(default_factory=list)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    training_summary: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    sha256: str


def _pipeline_bytes(pipeline) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    return buffer.getvalue()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sanitize(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return cleaned or "model"


def _resolve_dir(artifact_dir, *, create: bool) -> Path:
    path = Path(artifact_dir).expanduser().resolve()
    if create and not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ArtifactError("artifact directory is not a directory")
    return path


def _make_filename(model_name: str, version: str, digest: str) -> str:
    return (
        f"{_sanitize(model_name)}_{_sanitize(version)}_{digest[:16]}"
        f"{_ARTIFACT_SUFFIX}"
    )


def _atomic_write(pipeline_bytes: bytes, manifest: ArtifactManifest, target: Path) -> None:
    """Write a non-pickle wrapper to a temp file, then atomically replace."""
    target = Path(target)
    fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            with zipfile.ZipFile(handle, mode="w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr(
                    _MANIFEST_MEMBER,
                    json.dumps(manifest.model_dump(), sort_keys=True).encode("utf-8"),
                )
                archive.writestr(_PIPELINE_MEMBER, pipeline_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save_artifact(
    pipeline,
    *,
    model_name: str,
    version: str,
    task: str,
    features: list[str],
    input_schema: dict[str, Any] | None = None,
    training_summary: dict[str, Any] | None = None,
    artifact_dir,
    created_at: str | None = None,
) -> str:
    """Save the pipeline + manifest into ``artifact_dir`` and return the path."""
    root = _resolve_dir(artifact_dir, create=True)
    created = created_at or datetime.now(UTC).isoformat()
    pipeline_bytes = _pipeline_bytes(pipeline)
    digest = _sha256(pipeline_bytes)
    manifest = ArtifactManifest(
        manifest_version=MANIFEST_VERSION,
        model_name=str(model_name),
        version=str(version),
        task=str(task),
        features=[str(feature) for feature in features],
        input_schema=dict(input_schema or {}),
        training_summary=dict(training_summary or {}),
        created_at=created,
        sha256=digest,
    )
    filename = _make_filename(model_name, version, digest)
    target = root / filename
    _atomic_write(pipeline_bytes, manifest, target)
    return str(target)


def _validate_manifest(raw: Any) -> ArtifactManifest:
    try:
        manifest = ArtifactManifest.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        raise ArtifactError("artifact manifest is invalid") from exc
    if manifest.manifest_version != MANIFEST_VERSION:
        raise ArtifactError(
            f"unsupported manifest version {manifest.manifest_version} "
            f"(expected {MANIFEST_VERSION})"
        )
    return manifest


def load_artifact(artifact_dir, name) -> tuple[Any, ArtifactManifest]:
    """Load and verify an artifact; returns ``(pipeline, manifest)``."""
    root = _resolve_dir(artifact_dir, create=False)
    if not isinstance(name, (str, os.PathLike)):
        raise ArtifactError("artifact name must be a string or path-like value")
    name = os.fspath(name)
    if not _CONTROLLED_NAME.match(name) or not name.endswith(_ARTIFACT_SUFFIX):
        raise ArtifactError(f"untrusted artifact name {name!r}")
    path = (root / name).resolve()
    if path.parent != root:
        raise ArtifactError(f"path traversal rejected for {name!r}")
    if not path.is_file():
        raise ArtifactError(f"artifact not found: {name!r}")

    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            if set(archive.namelist()) != {_MANIFEST_MEMBER, _PIPELINE_MEMBER}:
                raise ArtifactError("artifact payload is malformed")
            manifest_raw = json.loads(archive.read(_MANIFEST_MEMBER).decode("utf-8"))
            pipeline_bytes = archive.read(_PIPELINE_MEMBER)
    except ArtifactError:
        raise
    except (OSError, ValueError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        raise ArtifactError("artifact is corrupt or unreadable") from exc

    manifest = _validate_manifest(manifest_raw)
    if _sha256(pipeline_bytes) != manifest.sha256:
        raise ArtifactError("artifact SHA-256 mismatch")

    try:
        pipeline = joblib.load(io.BytesIO(pipeline_bytes))
    except Exception as exc:  # noqa: BLE001
        raise ArtifactError("artifact pipeline is corrupt or unreadable") from exc

    return pipeline, manifest

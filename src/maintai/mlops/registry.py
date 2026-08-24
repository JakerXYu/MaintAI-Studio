"""MLflow model registry gateway for MaintAI Studio (P0).

Deterministic wrapper over the MLflow model registry, mirroring the tracking
gateway: every operation goes through :class:`mlflow.tracking.MlflowClient`
with an explicit ``tracking_uri`` (never the fluent ``mlflow.*`` global state),
so there is no global active-run or registry residue that can leak between
callers or tests.

P0 registers a trained tracking run as a new ``candidate`` model version. It
never promotes to ``champion`` or ``Production``: those transitions require
human approval and belong to P1. Every returned value is JSON-safe and never
exposes filesystem paths (``file://`` absolute paths included); failures are
converted into a stable :class:`RegistryError` that carries only an exception
class name, never a backend path or traceback.
"""

from __future__ import annotations

import re
from typing import Any

from mlflow.exceptions import RESOURCE_DOES_NOT_EXIST, MlflowException
from mlflow.tracking import MlflowClient
from mlflow.tracking.artifact_utils import get_artifact_uri
from pydantic import BaseModel

from maintai.mlops.tracker import MODEL_ARTIFACT_PATH

CANDIDATE_ALIAS = "candidate"
CHAMPION_ALIAS = "champion"

# Registered model names are readable and filesystem-safe: letters, digits and
# the ``.`` ``_`` ``-`` separators only. Path separators and control characters
# are rejected outright (never silently munged) so a malicious or typo'd name
# cannot escape the registry namespace or inject a filter string.
_REGISTRY_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_PATH_SEPARATOR_RE = re.compile(r"[/\\]")
_NAME_MAX_LEN = 255

_READY_STATUS = "READY"
_DEFAULT_CREATION_TIMEOUT_SECONDS = 60


class RegistryError(Exception):
    """Raised when an MLflow registry operation cannot complete."""


class RegistryVersion(BaseModel):
    """Minimal JSON-safe reference to a registered model version."""

    name: str
    version: str
    run_id: str | None
    model_uri: str
    alias: str | None
    status: str


def _validate_registry_name(name: Any) -> str:
    """Return the validated registry name, or raise a stable :class:`RegistryError`."""
    if not isinstance(name, str) or not name:
        raise RegistryError("registry name must be a non-empty string")
    if len(name) > _NAME_MAX_LEN:
        raise RegistryError(f"registry name is too long (max {_NAME_MAX_LEN} characters)")
    if _CONTROL_RE.search(name):
        raise RegistryError("registry name must not contain control characters")
    if _PATH_SEPARATOR_RE.search(name):
        raise RegistryError("registry name must not contain path separators")
    if not _REGISTRY_NAME_RE.match(name):
        raise RegistryError(
            "registry name may only contain letters, digits, '.', '_' and '-'"
        )
    return name


def _validate_version(version: Any) -> str:
    """Return the version as a normalized string, or raise :class:`RegistryError`."""
    if isinstance(version, bool):
        raise RegistryError("model version must be a positive integer string")
    if isinstance(version, int):
        text = str(version)
    elif isinstance(version, str):
        text = version
    else:
        raise RegistryError("model version must be a positive integer string")
    if not text.isdigit() or int(text) < 1:
        raise RegistryError("model version must be a positive integer string")
    return str(int(text))


def _validate_alias(alias: Any) -> str:
    """Return the alias when it is a P0-supported alias, else raise.

    P0 supports exactly one alias (``candidate``). ``champion`` is rejected with
    an explicit approval/P1 explanation rather than a generic error.
    """
    if alias == CANDIDATE_ALIAS:
        return CANDIDATE_ALIAS
    if alias == CHAMPION_ALIAS:
        raise RegistryError(
            "champion promotion requires human approval and is a P1 feature; "
            "P0 only supports the 'candidate' alias"
        )
    raise RegistryError("P0 only supports the 'candidate' alias")


class MLflowRegistry:
    """Minimal MLflow model registry gateway for P0 candidate registration."""

    def __init__(self, tracking_uri: str) -> None:
        if not isinstance(tracking_uri, str) or not tracking_uri.strip():
            raise RegistryError("tracking_uri must be a non-empty string")
        self.tracking_uri = tracking_uri

    def _client(self) -> MlflowClient:
        return MlflowClient(tracking_uri=self.tracking_uri)

    @staticmethod
    def _candidate_alias(mv: Any) -> str | None:
        aliases = getattr(mv, "aliases", None) or []
        return CANDIDATE_ALIAS if CANDIDATE_ALIAS in aliases else None

    @staticmethod
    def _to_version(mv: Any, alias: str | None) -> RegistryVersion:
        version = str(mv.version)
        return RegistryVersion(
            name=mv.name,
            version=version,
            run_id=mv.run_id,
            model_uri=f"models:/{mv.name}/{version}",
            alias=alias,
            status=mv.status,
        )

    def _get_or_create_registered_model(self, client: MlflowClient, name: str) -> None:
        try:
            client.get_registered_model(name)
        except MlflowException as exc:
            missing_codes = {
                RESOURCE_DOES_NOT_EXIST,
                str(RESOURCE_DOES_NOT_EXIST),
                "RESOURCE_DOES_NOT_EXIST",
            }
            if exc.error_code not in missing_codes:
                raise
            client.create_registered_model(name)

    def register_run(self, run_id: str, model_name: str) -> RegistryVersion:
        """Register ``run_id``'s logged model as a new ``candidate`` version.

        Validates the registry name strictly, resolves the run's ``model``
        artifact to its underlying artifact URI (used as the version source),
        creates the registered model if absent, creates a new model version and
        waits (bounded) for it to become ``READY``, then tags it with the
        ``candidate`` alias. Never promotes to ``champion`` or ``Production``.
        """
        name = _validate_registry_name(model_name)
        if not isinstance(run_id, str) or not run_id.strip():
            raise RegistryError("run_id must be a non-empty string")

        client = self._client()
        try:
            client.get_run(run_id)
            source = get_artifact_uri(
                run_id, MODEL_ARTIFACT_PATH, tracking_uri=self.tracking_uri
            )
            self._get_or_create_registered_model(client, name)
            existing = client.search_model_versions(
                filter_string=f"name = '{name}'",
                max_results=10000,
            )
            matching = [version for version in existing if version.run_id == run_id]
            if matching:
                version = max(matching, key=lambda item: int(str(item.version)))
                client.set_registered_model_alias(
                    name,
                    CANDIDATE_ALIAS,
                    str(version.version),
                )
                return self._to_version(version, alias=CANDIDATE_ALIAS)
            version = client.create_model_version(
                name=name,
                source=source,
                run_id=run_id,
                await_creation_for=_DEFAULT_CREATION_TIMEOUT_SECONDS,
            )
            client.set_registered_model_alias(name, CANDIDATE_ALIAS, str(version.version))
            return self._to_version(version, alias=CANDIDATE_ALIAS)
        except RegistryError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface any backend failure stably
            raise RegistryError(f"MLflow registry failed: {type(exc).__name__}") from exc

    def get_version(self, model_name: str, version: str) -> RegistryVersion:
        """Return the registered model version (P0 alias is ``candidate``)."""
        name = _validate_registry_name(model_name)
        version = _validate_version(version)
        client = self._client()
        try:
            mv = client.get_model_version(name, version)
            return self._to_version(mv, self._candidate_alias(mv))
        except RegistryError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RegistryError(f"MLflow registry failed: {type(exc).__name__}") from exc

    def list_versions(self, model_name: str) -> list[RegistryVersion]:
        """Return all versions of a registered model in ascending version order."""
        name = _validate_registry_name(model_name)
        client = self._client()
        try:
            versions = client.search_model_versions(
                filter_string=f"name = '{name}'", max_results=10000
            )
        except RegistryError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RegistryError(f"MLflow registry failed: {type(exc).__name__}") from exc

        ordered = sorted(versions, key=lambda mv: int(str(mv.version)))
        return [self._to_version(mv, self._candidate_alias(mv)) for mv in ordered]

    def set_alias(self, model_name: str, alias: str, version: str) -> RegistryVersion:
        """Set a P0-supported alias (``candidate``) on a registered version.

        ``champion`` (and any other alias) is rejected: champion promotion is
        gated behind human approval and is a P1 feature.
        """
        name = _validate_registry_name(model_name)
        version = _validate_version(version)
        _validate_alias(alias)
        client = self._client()
        try:
            client.set_registered_model_alias(name, alias, version)
            return self.get_version(name, version)
        except RegistryError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RegistryError(f"MLflow registry failed: {type(exc).__name__}") from exc

    def healthcheck(self) -> bool:
        """Return ``True`` when the registry backend is reachable."""
        try:
            self._client().search_registered_models(max_results=1)
        except Exception as exc:  # noqa: BLE001
            raise RegistryError(f"MLflow registry unreachable: {type(exc).__name__}") from exc
        return True

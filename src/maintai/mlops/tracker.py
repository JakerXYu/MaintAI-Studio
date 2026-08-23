"""MLflow tracking gateway for MaintAI Studio (P0).

Thin, deterministic wrapper around the MLflow backend used by the single
in-process P0 training worker. It talks to MLflow exclusively through
:class:`mlflow.tracking.MlflowClient` (never the fluent ``mlflow.*`` module-level
state), so there is no global active-run residue that can leak between runs or
tests. A run is created explicitly, logged, and terminated ``FINISHED``; any
failure is converted into a stable :class:`TrackingError` and the run is marked
``FAILED``.
"""

from __future__ import annotations

import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

PROJECT_TAG = "maintai-studio"
PHASE_TAG = "P0"
MODEL_ARTIFACT_PATH = "model"
_JSON_ARTIFACTS_PATH = "artifacts"

_RUN_STATUS_FINISHED = "FINISHED"
_RUN_STATUS_FAILED = "FAILED"

# MLflow limits: params may be up to 500 chars, tag values up to 5000.
_PARAM_MAX_LEN = 500
_TAG_MAX_LEN = 5000

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class TrackingError(Exception):
    """Raised when an MLflow tracking operation cannot complete."""


class TrackedRun(BaseModel):
    """Minimal JSON-safe reference to a successfully logged MLflow run."""

    run_id: str
    experiment_id: str
    artifact_uri: str
    model_uri: str
    status: str


def _json_default(value: Any) -> Any:
    """Convert numpy values to JSON-safe builtins for stable serialisation."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        default=_json_default,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _param_value(value: Any) -> str:
    """Render a param as a stable string, truncating long values."""
    value = _scalar(value)
    if value is None:
        text = "null"
    elif isinstance(value, (str, int, float, bool)):
        text = str(value)
    else:
        text = _json_text(value)
    if len(text) > _PARAM_MAX_LEN:
        text = text[: _PARAM_MAX_LEN] + f"...[truncated:{len(text)}]"
    return text


def _metric_value(value: Any) -> float | None:
    """Return a finite float for a metric, or ``None`` when the value is ``None``."""
    if value is None:
        return None
    value = _scalar(value)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TrackingError(f"metric value {value!r} is not numeric") from exc
    if not math.isfinite(result):
        raise TrackingError(f"metric value {value!r} is not finite")
    return result


def _tag_value(value: Any) -> str:
    value = _scalar(value)
    if value is None:
        text = ""
    elif isinstance(value, (str, int, float, bool)):
        text = str(value)
    else:
        text = _json_text(value)
    if len(text) > _TAG_MAX_LEN:
        text = text[: _TAG_MAX_LEN]
    return text


def _sanitize_filename(name: Any) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", str(name)).strip("._")
    return cleaned or "artifact"


class MLflowTracker:
    """Minimal MLflow tracking gateway for a single P0 worker."""

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        if not isinstance(tracking_uri, str) or not tracking_uri.strip():
            raise TrackingError("tracking_uri must be a non-empty string")
        if not isinstance(experiment_name, str) or not experiment_name.strip():
            raise TrackingError("experiment_name must be a non-empty string")
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

    def _client(self) -> MlflowClient:
        return MlflowClient(tracking_uri=self.tracking_uri)

    def _get_or_create_experiment(self, client: MlflowClient) -> str:
        existing = client.get_experiment_by_name(self.experiment_name)
        if existing is not None:
            return existing.experiment_id
        return client.create_experiment(self.experiment_name)

    def _log_tags(
        self,
        client: MlflowClient,
        run_id: str,
        model_name: str,
        task: str,
        tags: dict[str, Any] | None,
    ) -> None:
        merged: dict[str, Any] = dict(tags or {})
        # Required tags always win; callers cannot override the fixed values.
        merged["project"] = PROJECT_TAG
        merged["phase"] = PHASE_TAG
        merged.setdefault("model_family", model_name)
        merged.setdefault("model_name", model_name)
        merged.setdefault("task", task)
        for key, value in merged.items():
            client.set_tag(run_id, str(key), _tag_value(value))

    @staticmethod
    def _validate_pipeline(pipeline: Any) -> None:
        if pipeline is None or not callable(getattr(pipeline, "predict", None)):
            raise TrackingError(
                "pipeline must be a fitted model with a callable predict method "
                f"(got {type(pipeline).__name__})"
            )

    def _log_model_and_artifacts(
        self,
        client: MlflowClient,
        run_id: str,
        pipeline: Any,
        artifacts: dict[str, Any] | None,
        artifact_path: str,
    ) -> str:
        with tempfile.TemporaryDirectory(prefix="maintai_mlops_") as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            mlflow.sklearn.save_model(pipeline, str(model_dir))
            client.log_artifacts(run_id, str(model_dir), artifact_path=artifact_path)
            if artifacts:
                artifacts_dir = root / "json"
                artifacts_dir.mkdir()
                for filename, obj in artifacts.items():
                    target = artifacts_dir / _sanitize_filename(filename)
                    target.write_text(
                        json.dumps(
                            obj,
                            sort_keys=True,
                            default=_json_default,
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                client.log_artifacts(
                    run_id, str(artifacts_dir), artifact_path=_JSON_ARTIFACTS_PATH
                )
        return f"runs:/{run_id}/{artifact_path}"

    def log_model_run(
        self,
        *,
        model_name: str,
        task: str,
        pipeline: Any,
        params: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        artifact_path: str = MODEL_ARTIFACT_PATH,
    ) -> TrackedRun:
        """Create/select an experiment and a single run, log everything, finish it.

        ``params`` values are rendered to stable JSON strings (numpy-aware) and
        truncated. ``metrics`` values of ``None`` are skipped; non-finite or
        non-numeric values raise :class:`TrackingError`. ``artifacts`` is a
        ``filename -> JSON-safe object`` mapping written into the run under the
        ``artifacts/`` prefix. The fitted sklearn pipeline is logged with
        ``mlflow.sklearn`` under ``artifact_path``.
        """
        client = self._client()
        run_id: str | None = None
        try:
            experiment_id = self._get_or_create_experiment(client)
            run = client.create_run(experiment_id)
            run_id = run.info.run_id
            try:
                self._validate_pipeline(pipeline)
                self._log_tags(client, run_id, str(model_name), str(task), tags)
                for key, value in (params or {}).items():
                    client.log_param(run_id, str(key), _param_value(value))
                for key, value in (metrics or {}).items():
                    metric = _metric_value(value)
                    if metric is not None:
                        client.log_metric(run_id, str(key), metric)
                model_uri = self._log_model_and_artifacts(
                    client, run_id, pipeline, artifacts, artifact_path
                )
                client.set_terminated(run_id, _RUN_STATUS_FINISHED)
            except BaseException:
                self._mark_failed(client, run_id)
                raise
            info = client.get_run(run_id).info
            return TrackedRun(
                run_id=run_id,
                experiment_id=experiment_id,
                artifact_uri=info.artifact_uri,
                model_uri=model_uri,
                status=_RUN_STATUS_FINISHED,
            )
        except TrackingError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface any backend failure stably
            raise TrackingError(f"MLflow tracking failed: {type(exc).__name__}") from exc

    @staticmethod
    def _mark_failed(client: MlflowClient, run_id: str) -> None:
        try:
            client.set_terminated(run_id, _RUN_STATUS_FAILED)
        except Exception:  # noqa: BLE001 - best-effort; the original error propagates
            pass

    def get_run(self, run_id: str):
        """Return the raw MLflow :class:`~mlflow.entities.Run` for ``run_id``."""
        try:
            return self._client().get_run(run_id)
        except Exception as exc:  # noqa: BLE001
            raise TrackingError(f"MLflow get_run failed: {type(exc).__name__}") from exc

    def healthcheck(self) -> bool:
        """Return ``True`` when the tracking backend is reachable."""
        try:
            self._client().search_experiments(max_results=1)
        except Exception as exc:  # noqa: BLE001
            raise TrackingError(f"MLflow backend unreachable: {type(exc).__name__}") from exc
        return True

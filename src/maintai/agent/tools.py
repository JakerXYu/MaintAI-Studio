"""P0 copilot tool allowlist.

Only the public *read* methods of the existing application services are exposed:

* dataset profile / quality / task  → :class:`~maintai.application.datasets.DatasetService`
* experiment results / comparison   → :class:`~maintai.application.experiments.ExperimentService`
* model deployment status           → :class:`~maintai.application.models.ModelRegistryService`
* prediction + explanation          → :class:`~maintai.application.predictions.PredictionService`
  (only when an explicit record is supplied; the service itself rejects models
  that are not ``demo_deployed``)

The copilot is read-only: it never trains, registers, or deploys. When a user
asks for such an action the graph returns a proposal/refusal instead of calling
anything here. Tools take only explicit, fixed parameters (never ``**kwargs``),
return JSON-safe, path-free data with a stable ``evidence`` source, and never
surface repositories, sessions, filesystem paths, or SQL. There is no dynamic
import and no arbitrary argument passthrough.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from maintai.application.datasets import DatasetNotFoundError, DatasetService
from maintai.application.experiments import ExperimentNotFoundError, ExperimentService
from maintai.application.models import ModelRegistryService, RegisteredModelNotFoundError
from maintai.application.predictions import (
    InvalidRecordsError,
    ModelNotDeployedError,
    PredictionError,
    PredictionModelNotFoundError,
    PredictionService,
)


@dataclass(frozen=True)
class ToolResult:
    """Outcome of one allowlisted tool call.

    ``data`` is JSON-safe and never contains a filesystem path, repository,
    session, or SQL. ``evidence`` is a small source descriptor (source + ids)
    that tags where the data came from.
    """

    ok: bool
    data: Any = None
    error: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "data": self.data,
            "error": self.error,
            "evidence": self.evidence,
        }


# Internal dataset fields that must never cross the copilot boundary. ``file_path``
# is a controlled storage key and ``file_hash`` is a content digest; neither is a
# user-facing path, but they are still excluded so nothing internal leaks.
_DATASET_HIDDEN_FIELDS = frozenset(
    {"file_path", "file_hash", "original_filename", "size_bytes"}
)


def _public_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in dataset.items() if key not in _DATASET_HIDDEN_FIELDS}


class CopilotTools:
    """Fixed set of read-only tools bound to the application services."""

    def __init__(
        self,
        *,
        dataset_service: DatasetService,
        experiment_service: ExperimentService,
        model_registry_service: ModelRegistryService,
        prediction_service: PredictionService,
    ) -> None:
        self._dataset_service = dataset_service
        self._experiment_service = experiment_service
        self._model_registry_service = model_registry_service
        self._prediction_service = prediction_service

    # -- dataset -----------------------------------------------------------

    def dataset_profile(self, dataset_id: str) -> ToolResult:
        try:
            dataset = self._dataset_service.get(dataset_id)
        except DatasetNotFoundError as exc:
            return ToolResult(ok=False, error=str(exc))
        return ToolResult(
            ok=True,
            data=_public_dataset(dataset),
            evidence={"source": "dataset.profile", "dataset_id": dataset_id},
        )

    def dataset_quality(self, dataset_id: str) -> ToolResult:
        try:
            quality = self._dataset_service.get_quality(dataset_id)
        except DatasetNotFoundError as exc:
            return ToolResult(ok=False, error=str(exc))
        return ToolResult(
            ok=True,
            data=quality,
            evidence={"source": "dataset.quality", "dataset_id": dataset_id},
        )

    def dataset_task(self, dataset_id: str) -> ToolResult:
        try:
            dataset = self._dataset_service.get(dataset_id)
        except DatasetNotFoundError as exc:
            return ToolResult(ok=False, error=str(exc))
        profile = dataset.get("profile") if isinstance(dataset.get("profile"), dict) else {}
        return ToolResult(
            ok=True,
            data={
                "dataset_id": dataset.get("id"),
                "target_column": dataset.get("target_column"),
                "asset_id_column": dataset.get("asset_id_column"),
                "timestamp_column": dataset.get("timestamp_column"),
                "task_type": dataset.get("task_type"),
                "task_recommendation": profile.get("task_recommendation"),
                "leakage_report": profile.get("leakage_report"),
            },
            evidence={"source": "dataset.task", "dataset_id": dataset_id},
        )

    # -- experiment --------------------------------------------------------

    def experiment_results(self, experiment_id: str) -> ToolResult:
        try:
            experiment = self._experiment_service.get(experiment_id)
        except ExperimentNotFoundError as exc:
            return ToolResult(ok=False, error=str(exc))
        runs = [
            {
                key: run.get(key)
                for key in ("model_name", "primary_metric", "primary_metric_value", "metrics")
            }
            for run in (experiment.get("model_runs") or [])
        ]
        return ToolResult(
            ok=True,
            data={
                "experiment_id": experiment.get("id"),
                "task_type": experiment.get("task_type"),
                "status": experiment.get("status"),
                "primary_metric": experiment.get("primary_metric"),
                "value": experiment.get("value"),
                "recommended_model": experiment.get("recommended_model"),
                "recommended_run_id": experiment.get("recommended_run_id"),
                "model_runs": runs,
            },
            evidence={"source": "experiment.results", "experiment_id": experiment_id},
        )

    def experiment_comparison(self, experiment_id: str) -> ToolResult:
        try:
            comparison = self._experiment_service.comparison(experiment_id)
        except ExperimentNotFoundError as exc:
            return ToolResult(ok=False, error=str(exc))
        return ToolResult(
            ok=True,
            data=comparison,
            evidence={"source": "experiment.comparison", "experiment_id": experiment_id},
        )

    # -- model / prediction ------------------------------------------------

    def model_deployment_status(self, model_id: str) -> ToolResult:
        try:
            model = self._model_registry_service.get(model_id)
        except RegisteredModelNotFoundError as exc:
            return ToolResult(ok=False, error=str(exc))
        return ToolResult(
            ok=True,
            data={
                "model_id": model.get("id"),
                "name": model.get("name"),
                "version": model.get("version"),
                "deployment_status": model.get("deployment_status"),
                "alias": model.get("alias"),
                "mlflow_model_uri": model.get("mlflow_model_uri"),
            },
            evidence={"source": "model.deployment_status", "model_id": model_id},
        )

    def prediction_explain(self, model_id: str, record: dict[str, Any] | None) -> ToolResult:
        if not record:
            return ToolResult(
                ok=False,
                error="a record is required to explain a prediction",
            )
        try:
            result = self._prediction_service.predict(model_id, [record], persist=False)
        except PredictionModelNotFoundError as exc:
            return ToolResult(ok=False, error=str(exc))
        except ModelNotDeployedError as exc:
            return ToolResult(ok=False, error=str(exc))
        except (InvalidRecordsError, PredictionError) as exc:
            return ToolResult(ok=False, error=str(exc))
        return ToolResult(
            ok=True,
            data=result,
            evidence={"source": "prediction.explanation", "model_id": model_id},
        )

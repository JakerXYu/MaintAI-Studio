"""Experiment application service (use-case orchestration for P0 training).

Coordinates the full deterministic training loop for a single in-process worker
(synchronous — intended to be wrapped by an upper-layer ``BackgroundTasks``
caller, never a Celery/Redis worker):

    create(experiment) → queued (immutable TrainingPlan snapshot + config hash,
    atomic with audit) → run(experiment_id) → train + recommend → explain →
    package → succeeded / failed (atomic status + audit).

Every successful model is logged to MLflow and persisted as a ``ModelRun`` row.
MLflow success followed by a DB write failure is recorded as a reconciliation
note rather than being forced through a distributed transaction (which P0 does
not implement). No raw filesystem paths or stack traces are ever returned or
persisted.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session, sessionmaker

from maintai import __version__
from maintai.application.datasets import DatasetService, DatasetServiceError
from maintai.audit.service import AuditService
from maintai.data.ingest import DataIngestError
from maintai.data.split import SplitConfig, SplitError
from maintai.data.split import split as split_data
from maintai.db.experiment_repository import ExperimentRepository
from maintai.db.models import (
    EXPERIMENT_STATUS_FAILED,
    EXPERIMENT_STATUS_QUEUED,
    EXPERIMENT_STATUS_RUNNING,
    EXPERIMENT_STATUS_SUCCEEDED,
    MODEL_RUN_STATUS_SUCCESS,
    Experiment,
    ModelRun,
    new_id,
    utcnow,
)
from maintai.ml import catalog, explain, package, preprocess
from maintai.ml.catalog import ModelUnavailableError
from maintai.ml.explain import ExplanationError
from maintai.ml.package import ArtifactError
from maintai.ml.preprocess import PreprocessError
from maintai.ml.recommend import recommend as recommend_model
from maintai.ml.schemas import MLSettings, TrainingPlan
from maintai.ml.train import TrainingError, train
from maintai.mlops.tracker import MLflowTracker, TrackingError

_VALID_TASKS: tuple[str, ...] = (
    "binary_classification",
    "multiclass_classification",
    "regression",
)


class ExperimentServiceError(Exception):
    """Base class for experiment application-service errors."""


class ExperimentNotFoundError(ExperimentServiceError):
    """Raised when an experiment id does not exist."""


class DatasetNotReadyError(ExperimentServiceError):
    """Raised when a dataset lacks a target or a trainable classification/regression task."""


class ExperimentStateError(ExperimentServiceError):
    """Raised when a run is requested for an experiment not in the ``queued`` state."""


# Exceptions whose ``str()`` is already a stable, path-free message produced by
# the deterministic layers. Anything else is rendered by class name only so a
# raw path or traceback can never leak into a persisted error message.
_SAFE_ERROR_TYPES: tuple[type[Exception], ...] = (
    ExperimentServiceError,
    DatasetServiceError,
    DataIngestError,
    SplitError,
    TrainingError,
    TrackingError,
    ExplanationError,
    ArtifactError,
    PreprocessError,
    ModelUnavailableError,
)


def _stable_error_message(exc: Exception) -> str:
    """Return a stable, path-free error message (never a traceback or raw path)."""
    if isinstance(exc, _SAFE_ERROR_TYPES):
        text = str(exc).strip()
    else:
        text = f"{type(exc).__name__}: training run failed"
    if len(text) > 500:
        text = text[:500] + "..."
    return text


def _config_hash(payload: dict[str, Any]) -> str:
    """Deterministic SHA-256 of a stable JSON payload (portable, sort-keyed)."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _iso(value) -> str | None:
    return value.isoformat() if value is not None else None


def _json_scalar(value):
    """Convert numpy scalars to JSON-safe Python builtins."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.item() if value.size == 1 else value.tolist()
    return value


def _feature_baseline(
    train_frame: pd.DataFrame, feature_cols: list[str]
) -> dict[str, Any]:
    """Compute a JSON-safe raw-feature baseline on the train fold.

    Numeric columns use the median, categorical and boolean columns use the mode.
    The result contains one scalar per raw feature and never any raw row, so it
    can be embedded in the package manifest and reused later as a local
    explanation background.
    """
    baseline: dict[str, Any] = {}
    for column in feature_cols:
        series = train_frame[column]
        kind = preprocess.classify_column(series)
        if kind == "numeric":
            value: Any = float(series.median()) if series.notna().any() else 0.0
        else:
            mode = series.mode()
            value = _json_scalar(mode.iloc[0]) if not mode.empty else None
        baseline[column] = value
    return baseline


class ExperimentService:
    """Application service for the P0 experiment lifecycle."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        dataset_service: DatasetService,
        experiment_repository: ExperimentRepository,
        audit: AuditService,
        tracker: MLflowTracker,
        artifact_root: str | Path,
        ml_settings: MLSettings | None = None,
        split_settings: SplitConfig | None = None,
        code_version: str = __version__,
        git_commit: str | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._dataset_service = dataset_service
        self._experiment_repository = experiment_repository
        self._audit = audit
        self._tracker = tracker
        self._artifact_root = Path(artifact_root).resolve()
        self._ml_settings = ml_settings or MLSettings()
        self._split_settings = split_settings or SplitConfig()
        self._code_version = code_version
        self._git_commit = git_commit or os.getenv("MAINTAI_GIT_COMMIT", "unknown")

    # -- helpers -----------------------------------------------------------

    def _get_or_raise(self, experiment_id: str) -> Experiment:
        experiment = self._experiment_repository.get(experiment_id)
        if experiment is None:
            raise ExperimentNotFoundError(f"experiment {experiment_id!r} not found")
        return experiment

    def _experiment_to_dict(
        self, experiment: Experiment, *, include_details: bool
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": experiment.id,
            "dataset_id": experiment.dataset_id,
            "name": experiment.name,
            "task_type": experiment.task_type,
            "status": experiment.status,
            "recommended_model": experiment.recommended_model,
            "recommended_run_id": experiment.recommended_run_id,
            "config_hash": experiment.config_hash,
            "primary_metric": experiment.primary_metric,
            "value": experiment.value,
            "error_message": experiment.error_message,
            "created_at": _iso(experiment.created_at),
            "started_at": _iso(experiment.started_at),
            "completed_at": _iso(experiment.completed_at),
        }
        if include_details:
            result["training_plan"] = experiment.training_plan_json
        return result

    @staticmethod
    def _model_run_to_dict(model_run: ModelRun) -> dict[str, Any]:
        return {
            "id": model_run.id,
            "experiment_id": model_run.experiment_id,
            "mlflow_run_id": model_run.mlflow_run_id,
            "model_name": model_run.model_name,
            "config_hash": model_run.config_hash,
            "status": model_run.status,
            "primary_metric": model_run.primary_metric,
            "primary_metric_value": model_run.primary_metric_value,
            "metrics": model_run.metrics_json,
            "confusion_matrix": model_run.confusion_matrix_json,
            "artifact_uri": model_run.artifact_uri,
            "model_uri": model_run.model_uri,
            "feature_names": model_run.feature_names_json,
            "training_time_seconds": model_run.training_time_seconds,
            "created_at": _iso(model_run.created_at),
        }

    # -- public API --------------------------------------------------------

    def create(
        self,
        dataset_id: str,
        model_names: list[str] | None = None,
        minimum_recall: float | None = None,
    ) -> dict[str, Any]:
        """Create a queued experiment with an immutable TrainingPlan snapshot.

        Requires the dataset to already have a target column and a trainable
        classification/regression task (set by ``DatasetService.recommend_task``).
        The stored leakage-blocked features are read from the dataset profile and
        excluded from training; the deterministic split, resolved model list, and
        settings are frozen into a config hash. The experiment row and its audit
        event commit atomically.
        """
        ds = self._dataset_service.get(dataset_id)
        target = ds.get("target_column")
        task = ds.get("task_type")
        if not target:
            raise DatasetNotReadyError(
                f"dataset {dataset_id!r} has no target column; run task recommendation first"
            )
        if task not in _VALID_TASKS:
            raise DatasetNotReadyError(
                f"dataset {dataset_id!r} task {task!r} is not a trainable "
                "classification/regression task"
            )

        profile = ds.get("profile") if isinstance(ds.get("profile"), dict) else {}
        leakage = profile.get("leakage_report") if isinstance(profile, dict) else None
        excluded = list((leakage or {}).get("excluded_features") or [])
        excluded.extend(
            column
            for column in (ds.get("asset_id_column"), ds.get("timestamp_column"))
            if column
        )
        excluded = sorted(set(excluded))

        minimum_recall = (
            float(minimum_recall)
            if minimum_recall is not None
            else self._ml_settings.minimum_recall
        )
        if not 0.0 <= minimum_recall <= 1.0:
            raise ExperimentServiceError("minimum_recall must be in [0, 1]")

        resolved_models = list(model_names) if model_names else list(
            catalog.model_names_for_task(task)
        )
        allowed_models = set(catalog.model_names_for_task(task))
        invalid_models = sorted(set(resolved_models) - allowed_models)
        if invalid_models:
            raise ExperimentServiceError(
                f"model(s) are not available for task {task!r}: {invalid_models}"
            )
        if len(resolved_models) != len(set(resolved_models)):
            raise ExperimentServiceError("model_names must not contain duplicates")

        frame = self._dataset_service.load_frame(dataset_id)
        split_result = split_data(
            frame,
            target_col=target,
            asset_id_col=ds.get("asset_id_column"),
            timestamp_col=ds.get("timestamp_column"),
            task_type=task,
            config=self._split_settings,
        )

        plan = TrainingPlan(
            target=target,
            task=task,
            features=[],
            excluded=excluded,
            split=split_result,
            model_names=resolved_models,
            primary_metric=None,
            seed=self._ml_settings.seed,
        )

        config_payload: dict[str, Any] = {
            "file_hash": ds.get("file_hash"),
            "task": task,
            "target": target,
            "model_names": sorted(resolved_models),
            "minimum_recall": minimum_recall,
            "seed": self._ml_settings.seed,
            "scale_numeric": self._ml_settings.scale_numeric,
            "n_jobs": self._ml_settings.n_jobs,
            "excluded": sorted(excluded),
            "split": split_result.model_dump(),
            "code_version": self._code_version,
            "git_commit": self._git_commit,
        }
        config_hash = _config_hash(config_payload)

        snapshot: dict[str, Any] = {
            "plan": plan.model_dump(),
            "minimum_recall": minimum_recall,
            "scale_numeric": self._ml_settings.scale_numeric,
            "n_jobs": self._ml_settings.n_jobs,
            "code_version": self._code_version,
            "git_commit": self._git_commit,
        }

        experiment = Experiment(
            id=new_id(),
            dataset_id=dataset_id,
            name=f"exp-{task}-{dataset_id[:8]}",
            task_type=task,
            status=EXPERIMENT_STATUS_QUEUED,
            training_plan_json=snapshot,
            config_hash=config_hash,
        )
        with self._session_factory.begin() as session:
            experiment = self._experiment_repository.create(experiment, session=session)
            self._audit.record(
                actor_type="system",
                action="experiment.create",
                entity_type="experiment",
                entity_id=experiment.id,
                payload={
                    "experiment_id": experiment.id,
                    "dataset_id": dataset_id,
                    "task": task,
                    "target": target,
                    "model_names": resolved_models,
                    "minimum_recall": minimum_recall,
                    "config_hash": config_hash,
                    "split_strategy": split_result.strategy,
                    "excluded_features": excluded,
                },
                session=session,
            )
        return self._experiment_to_dict(experiment, include_details=True)

    def run(self, experiment_id: str) -> dict[str, Any]:
        """Execute a queued experiment synchronously and return its JSON state.

        ``queued → running``, load the frame, train + recommend, then mark
        ``succeeded`` (with the recommended model/run/primary value) or ``failed``
        (with a stable error and audit). Any failure is captured, never raised,
        so an upper-layer ``BackgroundTasks`` wrapper can simply await the result.
        """
        with self._session_factory.begin() as session:
            experiment = session.get(Experiment, experiment_id, with_for_update=True)
            if experiment is None:
                raise ExperimentNotFoundError(f"experiment {experiment_id!r} not found")
            if experiment.status != EXPERIMENT_STATUS_QUEUED:
                raise ExperimentStateError(
                    f"experiment {experiment_id!r} is {experiment.status!r}, not queued"
                )
            experiment.status = EXPERIMENT_STATUS_RUNNING
            experiment.started_at = utcnow()
            experiment.error_message = None
            experiment = self._experiment_repository.update(experiment, session=session)
            self._audit.record(
                actor_type="system",
                action="experiment.run_started",
                entity_type="experiment",
                entity_id=experiment.id,
                payload={"experiment_id": experiment.id, "status": experiment.status},
                session=session,
            )

        try:
            experiment = self._run_training(experiment)
        except Exception as exc:  # noqa: BLE001 - any failure becomes a stable failed state
            experiment = self._mark_failed(experiment, exc)
        return self._experiment_to_dict(experiment, include_details=True)

    def _run_training(self, experiment: Experiment) -> Experiment:
        """Train + recommend + explain + package; persists the succeeded state."""
        dataset_id = experiment.dataset_id
        if not dataset_id:
            raise ExperimentServiceError("experiment has no dataset")

        ds = self._dataset_service.get(dataset_id)
        target = ds.get("target_column")
        task = ds.get("task_type")
        if not target or task not in _VALID_TASKS:
            raise DatasetNotReadyError("dataset is not task-ready; cannot train")

        snapshot = dict(experiment.training_plan_json or {})
        plan = TrainingPlan.model_validate(snapshot["plan"])
        minimum_recall = float(snapshot.get("minimum_recall", self._ml_settings.minimum_recall))
        scale_numeric = bool(snapshot.get("scale_numeric", self._ml_settings.scale_numeric))
        n_jobs = int(snapshot.get("n_jobs", self._ml_settings.n_jobs))

        frame = self._dataset_service.load_frame(dataset_id)

        started = time.perf_counter()
        output = train(frame, plan, n_jobs=n_jobs, scale_numeric=scale_numeric)
        training_seconds = time.perf_counter() - started
        result = output.result
        snapshot["total_training_time_seconds"] = training_seconds
        experiment.training_plan_json = snapshot

        comparison = recommend_model(result, minimum_recall=minimum_recall)

        dataset_hash = ds.get("file_hash")
        schema_json = ds.get("schema")
        quality_report = (ds.get("profile") or {}).get("quality") if isinstance(
            ds.get("profile"), dict
        ) else None
        seed = plan.seed
        split_strategy = plan.split.strategy
        config_hash = experiment.config_hash

        run_ids: dict[str, str] = {}
        persisted_run_names: set[str] = set()
        reconciliation_notes: list[str] = []
        tracked_runs: list[dict[str, str]] = []

        for name, trained in output.models.items():
            evaluation = trained.evaluation
            metrics = dict(evaluation.metrics)
            metrics["inference_latency_ms"] = evaluation.inference_latency_ms
            metrics["training_time_seconds"] = evaluation.training_time_seconds

            try:
                tracked = self._tracker.log_model_run(
                    model_name=name,
                    task=task,
                    pipeline=trained.pipeline,
                    params={
                        "dataset_id": dataset_id,
                        "dataset_hash": dataset_hash,
                        "task": task,
                        "target": target,
                        "split": split_strategy,
                        "model": name,
                        "seed": seed,
                        "config_hash": config_hash,
                        "minimum_recall": minimum_recall,
                        "code_version": self._code_version,
                        "git_commit": self._git_commit,
                    },
                    metrics=metrics,
                    tags={"dataset_id": dataset_id, "experiment_id": experiment.id},
                    artifacts={
                        "training_plan.json": plan.model_dump(),
                        "evaluation.json": evaluation.model_dump(),
                        "schema.json": schema_json,
                        "quality.json": quality_report,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - persist partial tracking evidence
                reconciliation_notes.append(
                    f"model {name!r} was not fully tracked ({type(exc).__name__})"
                )
                snapshot["reconciliation_notes"] = reconciliation_notes
                snapshot["tracked_runs"] = tracked_runs
                experiment.training_plan_json = snapshot
                raise
            run_ids[name] = tracked.run_id
            tracked_runs.append({"model_name": name, "mlflow_run_id": tracked.run_id})
            snapshot["tracked_runs"] = tracked_runs
            experiment.training_plan_json = snapshot

            model_run = ModelRun(
                id=new_id(),
                experiment_id=experiment.id,
                mlflow_run_id=tracked.run_id,
                model_name=name,
                config_hash=config_hash,
                git_commit=self._git_commit,
                status=MODEL_RUN_STATUS_SUCCESS,
                primary_metric=result.primary_metric,
                primary_metric_value=evaluation.metrics.get(result.primary_metric),
                metrics_json=metrics,
                confusion_matrix_json=evaluation.confusion_matrix,
                artifact_uri=f"runs:/{tracked.run_id}",
                model_uri=tracked.model_uri,
                feature_names_json=list(trained.feature_names),
                training_time_seconds=evaluation.training_time_seconds,
            )
            try:
                self._experiment_repository.create_model_run(model_run)
                persisted_run_names.add(name)
            except Exception as exc:  # noqa: BLE001 - MLflow succeeded; note the risk
                reconciliation_notes.append(
                    f"MLflow run {tracked.run_id} for model {name!r} logged but its "
                    f"ModelRun row failed to persist ({type(exc).__name__})"
                )
            snapshot["reconciliation_notes"] = reconciliation_notes
            experiment.training_plan_json = snapshot

        best_name = comparison.best_model
        if best_name is None or best_name not in output.models:
            raise ExperimentServiceError(
                "no model satisfied the primary metric and minimum_recall constraint"
            )
        if best_name not in persisted_run_names:
            raise ExperimentServiceError(
                "recommended MLflow run has no persisted ModelRun row; reconciliation required"
            )
        best = output.models[best_name]
        best_evaluation = best.evaluation
        primary_value = best_evaluation.metrics.get(result.primary_metric)

        # Deterministic global explanation (SHAP with permutation fallback).
        train_frame = frame.iloc[plan.split.train_indices]
        feature_cols = preprocess.resolve_features(
            train_frame, target, plan.features, plan.excluded
        )
        X_test = frame.iloc[plan.split.test_indices].loc[:, feature_cols]
        y_test = frame[target].iloc[plan.split.test_indices]
        global_exp = explain.global_explanation(
            best.pipeline,
            X_test,
            y=y_test,
            task=task,
            positive_label=best_evaluation.positive_label,
            labels=best_evaluation.labels,
            seed=seed,
        )

        # Raw-feature baseline (train-fold median/mode) and, for regression, the
        # empirical absolute-residual quantile are embedded in the manifest so
        # the prediction service can reconstruct local explanations and intervals
        # without any raw training rows.
        feature_baseline = _feature_baseline(train_frame, feature_cols)
        training_summary: dict[str, Any] = {
            "experiment_id": experiment.id,
            "config_hash": config_hash,
            "primary_metric": result.primary_metric,
            "value": primary_value,
            "seed": seed,
            "feature_baseline": feature_baseline,
        }
        if task in ("binary_classification", "multiclass_classification"):
            training_summary["labels"] = [
                _json_scalar(label) for label in best_evaluation.labels
            ]
            training_summary["positive_label"] = _json_scalar(
                best_evaluation.positive_label
            )
        if task == "regression":
            calibration_indices = (
                plan.split.validation_indices
                if plan.split.validation_indices
                else plan.split.test_indices
            )
            X_calibration = frame.iloc[calibration_indices].loc[:, feature_cols]
            y_calibration = frame[target].iloc[calibration_indices]
            y_pred_test = np.asarray(best.pipeline.predict(X_calibration)).ravel()
            residuals = np.abs(
                np.asarray(y_calibration, dtype=float) - y_pred_test.astype(float)
            )
            training_summary["residual_interval"] = {
                "coverage": 0.9,
                "quantile": float(np.quantile(residuals, 0.9)),
            }

        # Save the complete pipeline + manifest (no raw path is persisted — only
        # the controlled, sanitized artifact basename).
        artifact_path = package.save_artifact(
            best.pipeline,
            model_name=best_name,
            version="1",
            task=task,
            features=feature_cols,
            input_schema=dict(schema_json or {}),
            training_summary=training_summary,
            artifact_dir=self._artifact_root,
        )
        artifact_name = Path(artifact_path).name

        new_snapshot = dict(snapshot)
        new_snapshot["result"] = result.model_dump()
        new_snapshot["comparison"] = comparison.model_dump()
        new_snapshot["explanation"] = global_exp.model_dump()
        new_snapshot["package_artifact"] = artifact_name
        new_snapshot["resolved_features"] = feature_cols
        if reconciliation_notes:
            new_snapshot["reconciliation_notes"] = reconciliation_notes

        experiment.training_plan_json = new_snapshot
        experiment.recommended_model = best_name
        experiment.recommended_run_id = run_ids[best_name]
        experiment.primary_metric = result.primary_metric
        experiment.value = primary_value
        experiment.status = EXPERIMENT_STATUS_SUCCEEDED
        experiment.completed_at = utcnow()
        experiment.error_message = None

        with self._session_factory.begin() as session:
            experiment = self._experiment_repository.update(experiment, session=session)
            self._audit.record(
                actor_type="system",
                action="experiment.run_succeeded",
                entity_type="experiment",
                entity_id=experiment.id,
                payload={
                    "experiment_id": experiment.id,
                    "dataset_id": dataset_id,
                    "recommended_model": best_name,
                    "recommended_run_id": run_ids[best_name],
                    "primary_metric": result.primary_metric,
                    "value": primary_value,
                },
                session=session,
            )
        return experiment

    def _mark_failed(self, experiment: Experiment, exc: Exception) -> Experiment:
        """Persist a stable failed state + audit, best-effort even if the DB is down."""
        experiment.status = EXPERIMENT_STATUS_FAILED
        experiment.error_message = _stable_error_message(exc)
        experiment.completed_at = utcnow()
        try:
            with self._session_factory.begin() as session:
                merged = self._experiment_repository.update(experiment, session=session)
                self._audit.record(
                    actor_type="system",
                    action="experiment.run_failed",
                    entity_type="experiment",
                    entity_id=experiment.id,
                    payload={
                        "experiment_id": experiment.id,
                        "status": experiment.status,
                        "error": experiment.error_message,
                    },
                    session=session,
                )
            return merged
        except Exception:  # noqa: BLE001 - DB unavailable; still return a JSON-safe state
            return experiment

    def get(self, experiment_id: str) -> dict[str, Any]:
        """Return the full experiment including its snapshot and model runs."""
        experiment = self._get_or_raise(experiment_id)
        data = self._experiment_to_dict(experiment, include_details=True)
        data["model_runs"] = [
            self._model_run_to_dict(run)
            for run in self._experiment_repository.list_model_runs(experiment_id)
        ]
        return data

    def list(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """Return experiment summaries (no JSON blobs, no model runs)."""
        experiments = self._experiment_repository.list(limit=limit, offset=offset)
        return [self._experiment_to_dict(exp, include_details=False) for exp in experiments]

    def comparison(self, experiment_id: str) -> dict[str, Any]:
        """Return the JSON-safe best-model comparison for an experiment."""
        experiment = self._get_or_raise(experiment_id)
        snapshot = experiment.training_plan_json
        stored = snapshot.get("comparison") if isinstance(snapshot, dict) else None
        if isinstance(stored, dict):
            result = dict(stored)
        else:
            result = {
                "task": experiment.task_type,
                "primary_metric": experiment.primary_metric,
                "best_model": experiment.recommended_model,
                "ranking": [],
                "candidates": [],
                "notes": ["experiment has not produced a comparison yet"],
            }
        result["experiment_id"] = experiment.id
        result["recommended_run_id"] = experiment.recommended_run_id
        result["value"] = experiment.value
        return result

"""Monitoring application service (P1 deterministic monitoring orchestration).

Coordinates a single monitoring run end-to-end: resolve a registered model's
lineage (RegisteredModel → ModelRun → Experiment → baseline Dataset), derive the
raw-feature allowlist from the experiment plan (excluding target/id/timestamp),
obtain one production window (an uploaded dataset or one synthetic replay batch),
then fit/score anomaly detection, detect drift, and build a retraining
recommendation. The results and a single audit event are persisted atomically; a
failed run is persisted with ``status="failed"`` and a stable, path-free error
message.

This service only *observes*. It never trains, deploys, promotes, or creates an
approval request — those transitions stay behind human approval (see AGENTS.md).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy.orm import Session, sessionmaker

from maintai.application.datasets import (
    DatasetNotFoundError,
    DatasetService,
    DatasetServiceError,
    StorageError,
)
from maintai.audit.service import AuditService
from maintai.data.ingest import DataIngestError
from maintai.db.experiment_repository import ExperimentRepository
from maintai.db.model_repository import ModelRepository
from maintai.db.models import (
    DEPLOYMENT_STATUS_CANDIDATE,
    DEPLOYMENT_STATUS_DEMO_DEPLOYED,
    MONITORING_REPLAY_KINDS,
    MONITORING_STATUS_FAILED,
    MONITORING_STATUS_SUCCEEDED,
    MonitoringRun,
    new_id,
    utcnow,
)
from maintai.db.monitoring_repository import MonitoringRepository
from maintai.ml import preprocess
from maintai.ml.preprocess import PreprocessError
from maintai.monitoring import anomaly, drift, recommend, replay
from maintai.monitoring.anomaly import AnomalyConfig
from maintai.monitoring.drift import DriftConfig
from maintai.monitoring.recommend import RecommendConfig

# Registered-model deployment states that may be monitored. Human-approval
# promotion (champion/production) is a later P1 concern and is NOT monitorable
# here; only the two P0-serving states are accepted.
_MONITORABLE_STATUSES = frozenset(
    {DEPLOYMENT_STATUS_CANDIDATE, DEPLOYMENT_STATUS_DEMO_DEPLOYED}
)


class MonitoringServiceError(Exception):
    """Base class for monitoring application-service errors."""


class MonitoringModelNotFoundError(MonitoringServiceError):
    """Raised when a registered-model id does not exist."""


class MonitoringNotDeployableError(MonitoringServiceError):
    """Raised when a registered model is neither candidate nor demo_deployed."""


class MonitoringSourceError(MonitoringServiceError):
    """Raised when the production source is not exactly one of dataset/replay."""


class MonitoringDatasetNotFoundError(MonitoringServiceError):
    """Raised when a referenced production dataset id does not exist."""


class MonitoringRunNotFoundError(MonitoringServiceError):
    """Raised when a monitoring-run id does not exist."""


# Exceptions whose ``str()`` is already a stable, path-free message produced by
# the deterministic layers. Anything else is rendered by class name only so a raw
# path or traceback can never leak into a persisted error message.
_SAFE_ERROR_TYPES: tuple[type[Exception], ...] = (
    MonitoringServiceError,
    DatasetServiceError,
    StorageError,
    PreprocessError,
    drift.DriftError,
    anomaly.AnomalyError,
    DataIngestError,
)


def _stable_error_message(exc: Exception) -> str:
    """Return a stable, path-free error message (never a traceback or raw path)."""
    if isinstance(exc, _SAFE_ERROR_TYPES):
        text = str(exc).strip()
    else:
        text = f"{type(exc).__name__}: monitoring run failed"
    if len(text) > 500:
        text = text[:500] + "..."
    return text


def _iso(value) -> str | None:
    return value.isoformat() if value is not None else None


class MonitoringService:
    """Application service for deterministic monitoring runs (observe only)."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        dataset_service: DatasetService,
        model_repository: ModelRepository,
        experiment_repository: ExperimentRepository,
        repository: MonitoringRepository,
        audit: AuditService,
        anomaly_config: AnomalyConfig | None = None,
        drift_config: DriftConfig | None = None,
        recommend_config: RecommendConfig | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._dataset_service = dataset_service
        self._model_repository = model_repository
        self._experiment_repository = experiment_repository
        self._repository = repository
        self._audit = audit
        self._anomaly_config = anomaly_config or AnomalyConfig()
        self._drift_config = drift_config or DriftConfig()
        self._recommend_config = recommend_config or RecommendConfig()

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _validate_source(
        production_dataset_id: str | None, replay_kind: str | None
    ) -> None:
        """Enforce exactly one production source (uploaded dataset xor replay)."""
        has_production = production_dataset_id is not None
        has_replay = replay_kind is not None
        if has_production == has_replay:
            raise MonitoringSourceError(
                "exactly one of production_dataset_id or replay_kind is required"
            )
        if replay_kind is not None and replay_kind not in MONITORING_REPLAY_KINDS:
            raise MonitoringSourceError(
                f"replay_kind must be one of {sorted(MONITORING_REPLAY_KINDS)}"
            )

    def _require_monitorable(self, registered_id: str):
        model = self._model_repository.get(registered_id)
        if model is None:
            raise MonitoringModelNotFoundError(
                f"registered model {registered_id!r} not found"
            )
        if model.deployment_status not in _MONITORABLE_STATUSES:
            raise MonitoringNotDeployableError(
                f"registered model {registered_id!r} is {model.deployment_status!r}, "
                "not candidate or demo_deployed"
            )
        return model

    def _require_lineage(self, registered) -> tuple[Any, Any, dict[str, Any]]:
        """Resolve RegisteredModel → ModelRun → Experiment → baseline Dataset."""
        model_run = (
            self._model_repository.get_model_run(registered.model_run_id)
            if registered.model_run_id
            else None
        )
        if model_run is None:
            raise MonitoringServiceError("registered model has no model run")
        experiment = (
            self._experiment_repository.get(model_run.experiment_id)
            if model_run.experiment_id
            else None
        )
        if experiment is None:
            raise MonitoringServiceError("model run is not attached to an experiment")
        if not experiment.dataset_id:
            raise MonitoringServiceError("experiment has no baseline dataset")
        try:
            baseline = self._dataset_service.get(experiment.dataset_id)
        except DatasetNotFoundError as exc:
            raise MonitoringServiceError("baseline dataset is missing") from exc
        return model_run, experiment, baseline

    @staticmethod
    def _plan(target: Any, features: Any, excluded: Any) -> tuple[str, list[str], list[str]]:
        """Normalise the immutable plan snapshot's (target, features, excluded)."""
        plan_features = list(features) if isinstance(features, list) else []
        plan_excluded = list(excluded) if isinstance(excluded, list) else []
        return target, plan_features, plan_excluded

    @staticmethod
    def _resolve_feature_allowlist(
        frame: pd.DataFrame, dataset_meta: dict[str, Any], experiment: Any
    ) -> tuple[list[str], str]:
        """Derive the raw-feature allowlist from the experiment plan.

        This mirrors the package ``manifest.features`` used at packaging time:
        raw columns resolved deterministically, excluding the target and the
        declared id/timestamp columns (plus any leakage columns frozen in the
        plan's ``excluded`` list).
        """
        snapshot = (
            experiment.training_plan_json
            if isinstance(experiment.training_plan_json, dict)
            else {}
        )
        plan = snapshot.get("plan") if isinstance(snapshot, dict) else {}
        target, plan_features, plan_excluded = MonitoringService._plan(
            plan.get("target"), plan.get("features"), plan.get("excluded")
        )
        target = target or dataset_meta.get("target_column")
        if not target:
            raise MonitoringServiceError("baseline dataset has no target column")

        excluded = set(plan_excluded)
        for column in (dataset_meta.get("asset_id_column"), dataset_meta.get("timestamp_column")):
            if column:
                excluded.add(column)

        frozen_features = snapshot.get("resolved_features")
        if isinstance(frozen_features, list) and frozen_features:
            missing = [column for column in frozen_features if column not in frame.columns]
            if missing:
                raise MonitoringServiceError(
                    f"baseline is missing frozen model feature(s): {missing}"
                )
            allowlist = [
                column
                for column in frozen_features
                if column != target and column not in excluded
            ]
            if not allowlist:
                raise MonitoringServiceError("no frozen model features remain after exclusions")
            return allowlist, str(target)

        allowlist = preprocess.resolve_features(frame, target, plan_features, sorted(excluded))
        if not allowlist:
            raise MonitoringServiceError("no raw feature columns remain after exclusions")
        return allowlist, str(target)

    def _resolve_production_frame(
        self,
        baseline_frame: pd.DataFrame,
        allowlist: list[str],
        target: str,
        production_dataset_id: str | None,
        replay_kind: str | None,
    ) -> pd.DataFrame:
        """Return the production window: an uploaded dataset or one replay batch."""
        if production_dataset_id is not None:
            return self._dataset_service.load_frame(production_dataset_id)
        replay_input = baseline_frame.loc[:, [*allowlist, target]].copy()
        return replay.generate_batches(
            replay_input, seed=self._anomaly_config.seed, target=target
        ).frames[replay_kind]

    def _compute(
        self,
        baseline_frame: pd.DataFrame,
        production_frame: pd.DataFrame,
        allowlist: list[str],
        target: str,
        performance_drop: float | None,
        baseline_anomaly_rate: float | None,
        schedule_due: bool,
        manual: bool,
    ) -> dict[str, Any]:
        """Fit/score anomaly, detect drift, and recommend (never train/deploy)."""
        numeric_features = [
            column
            for column in allowlist
            if pd.api.types.is_numeric_dtype(baseline_frame[column].dtype)
            and not pd.api.types.is_bool_dtype(baseline_frame[column].dtype)
        ]
        if not numeric_features:
            raise anomaly.AnomalyError(
                "no numeric features available for anomaly detection"
            )

        baseline = anomaly.fit_baseline(
            baseline_frame, numeric_columns=numeric_features, config=self._anomaly_config
        )
        anomaly_result = anomaly.score(baseline, production_frame)
        baseline_anomaly_result = anomaly.score(baseline, baseline_frame)
        drift_report = drift.detect_drift(
            baseline_frame, production_frame, columns=allowlist, config=self._drift_config
        )
        recommendation = recommend.recommend(
            drift_report,
            performance_drop=performance_drop,
            anomaly_rate=anomaly_result.anomaly_rate,
            baseline_anomaly_rate=(
                float(baseline_anomaly_rate)
                if baseline_anomaly_rate is not None
                else baseline_anomaly_result.anomaly_rate
            ),
            schedule_due=bool(schedule_due),
            manual=bool(manual),
            config=self._recommend_config,
        )
        return {
            "drift": drift_report,
            "anomaly": anomaly_result,
            "recommendation": recommendation,
            "allowlist": allowlist,
            "numeric_features": numeric_features,
            "baseline_row_count": int(len(baseline_frame)),
            "production_row_count": int(len(production_frame)),
            "baseline_anomaly_rate": baseline_anomaly_result.anomaly_rate,
        }

    @staticmethod
    def _anomaly_summary(result: Any) -> dict[str, Any]:
        """Return a bounded, value-redacted anomaly report for persistence/API."""
        summary = result.model_dump(exclude={"rows"})
        ranked = sorted(result.rows, key=lambda row: row.combined_score, reverse=True)[:20]
        summary["top_anomalies"] = [
            {
                "row_index": row.row_index,
                "robust_score": row.robust_score,
                "iforest_score": row.iforest_score,
                "combined_score": row.combined_score,
                "flag": row.flag,
                "top_deviating_features": [
                    {"feature": feature.feature, "robust_z": feature.robust_z}
                    for feature in row.top_deviating_features
                ],
            }
            for row in ranked
        ]
        summary["rows_omitted"] = max(0, len(result.rows) - len(ranked))
        return summary

    @staticmethod
    def _to_dict(run: MonitoringRun) -> dict[str, Any]:
        """Render a monitoring run as JSON-safe detail (no paths, no raw rows)."""
        return {
            "id": run.id,
            "registered_model_id": run.registered_model_id,
            "dataset_id": run.dataset_id,
            "production_dataset_id": run.production_dataset_id,
            "replay_kind": run.replay_kind,
            "status": run.status,
            "drift": run.drift_json,
            "anomaly": run.anomaly_json,
            "recommendation": run.recommendation_json,
            "input_summary": run.input_summary_json,
            "error_message": run.error_message,
            "created_at": _iso(run.created_at),
            "completed_at": _iso(run.completed_at),
        }

    def _audit_payload(
        self, run: MonitoringRun, registered, source: str
    ) -> dict[str, Any]:
        """Allowlisted audit metadata only — never raw rows or result blobs."""
        drift_severity = None
        anomaly_rate = None
        recommended = None
        if isinstance(run.drift_json, dict):
            drift_severity = run.drift_json.get("overall_severity")
        if isinstance(run.anomaly_json, dict):
            anomaly_rate = run.anomaly_json.get("anomaly_rate")
        if isinstance(run.recommendation_json, dict):
            recommended = run.recommendation_json.get("recommended")
        return {
            "monitoring_run_id": run.id,
            "registered_model_id": run.registered_model_id,
            "model_name": registered.name,
            "status": run.status,
            "source": source,
            "replay_kind": run.replay_kind,
            "production_dataset_id": run.production_dataset_id,
            "baseline_dataset_id": run.dataset_id,
            "drift_severity": drift_severity,
            "anomaly_rate": anomaly_rate,
            "recommended": recommended,
        }

    def _persist(self, run: MonitoringRun, registered, source: str) -> MonitoringRun:
        """Persist the run and its audit event in a single transaction."""
        payload = self._audit_payload(run, registered, source)
        with self._session_factory.begin() as session:
            run = self._repository.create(run, session=session)
            self._audit.record(
                actor_type="system",
                action="monitoring.run",
                entity_type="registered_model",
                entity_id=registered.id,
                payload=payload,
                session=session,
            )
        return run

    # -- public API --------------------------------------------------------

    def run(
        self,
        model_id: str,
        *,
        production_dataset_id: str | None = None,
        replay_kind: str | None = None,
        performance_drop: float | None = None,
        baseline_anomaly_rate: float | None = None,
        schedule_due: bool = False,
        manual: bool = False,
    ) -> dict[str, Any]:
        """Run one monitoring pass and return its persisted JSON-safe result.

        Synchronous by design (demo scale). Validation errors (not found, not
        deployable, invalid source, missing production dataset) raise before any
        row is written; any compute/storage failure is captured as a persisted
        ``failed`` run with a stable error message.
        """
        self._validate_source(production_dataset_id, replay_kind)
        registered = self._require_monitorable(model_id)
        _, experiment, baseline = self._require_lineage(registered)
        if production_dataset_id == baseline["id"]:
            raise MonitoringSourceError(
                "production_dataset_id must differ from the baseline dataset"
            )

        if production_dataset_id is not None:
            try:
                self._dataset_service.get(production_dataset_id)
            except DatasetNotFoundError as exc:
                raise MonitoringDatasetNotFoundError(
                    f"production dataset {production_dataset_id!r} not found"
                ) from exc

        run = MonitoringRun(
            id=new_id(),
            registered_model_id=registered.id,
            dataset_id=baseline["id"],
            production_dataset_id=production_dataset_id,
            replay_kind=replay_kind,
            status=MONITORING_STATUS_SUCCEEDED,
        )

        try:
            baseline_frame = self._dataset_service.load_frame(baseline["id"])
            allowlist, target = self._resolve_feature_allowlist(
                baseline_frame, baseline, experiment
            )
            production_frame = self._resolve_production_frame(
                baseline_frame,
                allowlist,
                target,
                production_dataset_id,
                replay_kind,
            )
            result = self._compute(
                baseline_frame,
                production_frame,
                allowlist,
                target,
                performance_drop,
                baseline_anomaly_rate,
                schedule_due,
                manual,
            )
            source = "replay" if replay_kind is not None else "uploaded"
            run.drift_json = result["drift"].model_dump()
            run.anomaly_json = self._anomaly_summary(result["anomaly"])
            run.recommendation_json = result["recommendation"].model_dump()
            run.input_summary_json = {
                "source": source,
                "replay_kind": replay_kind,
                "production_dataset_id": production_dataset_id,
                "target": target,
                "feature_allowlist": result["allowlist"],
                "numeric_features": result["numeric_features"],
                "baseline_row_count": result["baseline_row_count"],
                "production_row_count": result["production_row_count"],
                "baseline_anomaly_rate": result["baseline_anomaly_rate"],
            }
            run.status = MONITORING_STATUS_SUCCEEDED
            run.completed_at = utcnow()
        except Exception as exc:  # noqa: BLE001 - any failure becomes a stable failed state
            run.status = MONITORING_STATUS_FAILED
            run.error_message = _stable_error_message(exc)
            run.completed_at = utcnow()

        source = "replay" if replay_kind is not None else "uploaded"
        run = self._persist(run, registered, source)
        return self._to_dict(run)

    def get(self, run_id: str) -> dict[str, Any]:
        """Return a monitoring run by id."""
        run = self._repository.get(run_id)
        if run is None:
            raise MonitoringRunNotFoundError(f"monitoring run {run_id!r} not found")
        return self._to_dict(run)

    def list(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """Return monitoring runs newest-first."""
        runs = self._repository.list(limit=limit, offset=offset)
        return [self._to_dict(run) for run in runs]

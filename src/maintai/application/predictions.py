"""Prediction application service (P0 demo inference).

Loads only a ``demo_deployed`` registered model's controlled package artifact
(cached by id/version), validates a strict raw-feature contract (1..1000 records,
no missing/extra keys), predicts through the stored DataFrame pipeline, and
returns JSON-safe per-record results with a local explanation built from the
manifest's raw-feature baseline background.

Every record becomes an immutable :class:`PredictionEvent`; the single request
audit event stores only counts/model/input hashes (never the raw input payload).
All failures surface as stable, path-free messages.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session, sessionmaker

from maintai.audit.service import AuditService
from maintai.db.model_repository import ModelRepository
from maintai.db.models import (
    DEPLOYMENT_STATUS_DEMO_DEPLOYED,
    PredictionEvent,
    utcnow,
)
from maintai.db.prediction_repository import PredictionRepository
from maintai.ml import package
from maintai.ml.confidence import EMPIRICAL_DISCLAIMER
from maintai.ml.explain import ExplanationError, local_explanation
from maintai.ml.package import ArtifactError, ArtifactManifest
from maintai.ml.schemas import CLASSIFICATION_TASKS

_MAX_RECORDS = 100


class PredictionServiceError(Exception):
    """Base class for prediction application-service errors."""


class PredictionModelNotFoundError(PredictionServiceError):
    """Raised when a registered-model id does not exist."""


class ModelNotDeployedError(PredictionServiceError):
    """Raised when a registered model is not currently ``demo_deployed``."""


class InvalidRecordsError(PredictionServiceError):
    """Raised when the records payload violates the strict feature contract."""


class PredictionError(PredictionServiceError):
    """Raised when a prediction cannot be produced from the stored artifact."""


def _input_hash(record: dict[str, Any]) -> str:
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PredictionService:
    """Application service for deployed-model inference and prediction events."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        model_repository: ModelRepository,
        prediction_repository: PredictionRepository,
        audit: AuditService,
        artifact_root: str | Path,
        top_k: int = 10,
        seed: int = 42,
    ) -> None:
        self._session_factory = session_factory
        self._model_repository = model_repository
        self._prediction_repository = prediction_repository
        self._audit = audit
        self._artifact_root = Path(artifact_root).resolve()
        self._top_k = top_k
        self._seed = seed
        self._cache: OrderedDict[
            tuple[str, str], tuple[Any, ArtifactManifest]
        ] = OrderedDict()

    # -- loading / validation ---------------------------------------------

    def _load_deployed(self, registered) -> tuple[Any, ArtifactManifest]:
        key = (registered.id, registered.version)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        artifact_name = registered.artifact_uri
        if not artifact_name:
            raise PredictionError("registered model has no package artifact")
        try:
            pipeline, manifest = package.load_artifact(self._artifact_root, artifact_name)
        except ArtifactError as exc:
            raise PredictionError(f"package artifact is invalid: {exc}") from exc
        model_run = (
            self._model_repository.get_model_run(registered.model_run_id)
            if registered.model_run_id
            else None
        )
        if model_run is not None and manifest.model_name != model_run.model_name:
            raise PredictionError("package artifact model does not match the model run")
        self._cache[key] = (pipeline, manifest)
        self._cache.move_to_end(key)
        while len(self._cache) > 16:
            self._cache.popitem(last=False)
        return pipeline, manifest

    @staticmethod
    def _validate_records(records, features: list[str]) -> list[dict[str, Any]]:
        if not isinstance(records, (list, tuple)):
            raise InvalidRecordsError("records must be a list of objects")
        if len(records) < 1 or len(records) > _MAX_RECORDS:
            raise InvalidRecordsError(
                f"records must contain between 1 and {_MAX_RECORDS} rows"
            )
        required = set(features)
        cleaned: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise InvalidRecordsError(f"record {index} must be an object")
            keys = set(record.keys())
            missing = sorted(required - keys)
            extra = sorted(keys - required)
            if missing:
                raise InvalidRecordsError(
                    f"record {index} is missing feature(s): {missing}"
                )
            if extra:
                raise InvalidRecordsError(
                    f"record {index} has unexpected feature(s): {extra}"
                )
            cleaned.append(record)
        return cleaned

    @staticmethod
    def _validate_types(
        records: list[dict[str, Any]],
        manifest: ArtifactManifest,
    ) -> None:
        schema = manifest.input_schema
        columns = schema.get("columns") if isinstance(schema, dict) else None
        if not isinstance(columns, list):
            return
        dtype_by_name = {
            item.get("name"): item.get("dtype")
            for item in columns
            if isinstance(item, dict)
        }
        for index, record in enumerate(records):
            for feature in manifest.features:
                value = record[feature]
                if value is None:
                    continue
                dtype = dtype_by_name.get(feature)
                if dtype in {"int", "float"} and (
                    isinstance(value, bool) or not isinstance(value, Real)
                ):
                    raise InvalidRecordsError(
                        f"record {index} feature {feature!r} must be numeric"
                    )
                if dtype == "bool" and not (
                    isinstance(value, bool) or value in (0, 1)
                ):
                    raise InvalidRecordsError(
                        f"record {index} feature {feature!r} must be boolean"
                    )

    @staticmethod
    def _baseline_frame(baseline: Any, features: list[str]) -> pd.DataFrame | None:
        """Build a single-row background frame from the manifest baseline."""
        if not isinstance(baseline, dict) or not baseline:
            return None
        row: dict[str, Any] = {}
        for feature in features:
            if feature not in baseline:
                return None
            row[feature] = baseline[feature]
        return pd.DataFrame([row], columns=features)

    # -- per-record output -------------------------------------------------

    def _explain_record(
        self,
        pipeline,
        manifest: ArtifactManifest,
        index: int,
        frame: pd.DataFrame,
        baseline_frame: pd.DataFrame | None,
        task: str,
        labels: list[Any],
        positive_label: Any,
    ) -> dict[str, Any]:
        try:
            explanation = local_explanation(
                pipeline,
                frame.iloc[[index]],
                background=baseline_frame,
                task=task,
                positive_label=positive_label,
                labels=labels,
                top_k=self._top_k,
                seed=self._seed,
            )
        except ExplanationError as exc:
            raise PredictionError(f"local explanation failed: {exc}") from exc
        return {
            "top_positive": [impact.model_dump() for impact in explanation.top_positive],
            "top_negative": [impact.model_dump() for impact in explanation.top_negative],
            "method": explanation.method,
            "disclaimer": explanation.disclaimer,
        }

    def _classify_record(
        self,
        pipeline,
        manifest: ArtifactManifest,
        encoded: int,
        proba_row: np.ndarray | None,
        index: int,
        frame: pd.DataFrame,
        baseline_frame: pd.DataFrame | None,
        labels: list[Any],
        positive_label: Any,
    ) -> dict[str, Any]:
        decoded = labels[encoded] if labels and 0 <= encoded < len(labels) else encoded
        confidence: float | None = None
        positive_probability: float | None = None
        if proba_row is not None and labels:
            row = np.asarray(proba_row, dtype=float).ravel()
            if row.shape[0] == len(labels):
                if 0 <= encoded < len(row):
                    confidence = float(row[encoded])
                if positive_label is not None and positive_label in labels:
                    positive_probability = float(row[labels.index(positive_label)])
        result: dict[str, Any] = {
            "prediction": decoded,
            "confidence": confidence,
            "positive_probability": positive_probability,
            "positive_label": positive_label,
            "explanation": self._explain_record(
                pipeline,
                manifest,
                index,
                frame,
                baseline_frame,
                manifest.task,
                labels,
                positive_label,
            ),
        }
        return result

    def _regress_record(
        self,
        pipeline,
        manifest: ArtifactManifest,
        prediction: float,
        index: int,
        frame: pd.DataFrame,
        baseline_frame: pd.DataFrame | None,
        labels: list[Any],
        positive_label: Any,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"prediction": float(prediction)}
        residual_interval = manifest.training_summary.get("residual_interval")
        if isinstance(residual_interval, dict) and residual_interval.get("quantile") is not None:
            quantile = float(residual_interval["quantile"])
            coverage = float(residual_interval.get("coverage", 0.9))
            result["interval"] = {
                "lower": result["prediction"] - quantile,
                "upper": result["prediction"] + quantile,
                "coverage": coverage,
                "disclaimer": EMPIRICAL_DISCLAIMER,
            }
        result["explanation"] = self._explain_record(
            pipeline,
            manifest,
            index,
            frame,
            baseline_frame,
            manifest.task,
            labels,
            positive_label,
        )
        return result

    # -- public API --------------------------------------------------------

    def predict(
        self,
        registered_id: str,
        records: list[dict[str, Any]],
        *,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Predict on a deployed model; optionally skip event/audit persistence.

        ``persist=False`` is reserved for read-only previews such as the Copilot
        explanation tool. HTTP inference uses the default and remains audited.
        """
        registered = self._model_repository.get(registered_id)
        if registered is None:
            raise PredictionModelNotFoundError(
                f"registered model {registered_id!r} not found"
            )
        if registered.deployment_status != DEPLOYMENT_STATUS_DEMO_DEPLOYED:
            raise ModelNotDeployedError(
                f"registered model {registered_id!r} is not demo-deployed"
            )

        pipeline, manifest = self._load_deployed(registered)
        features = list(manifest.features)
        task = manifest.task
        summary = manifest.training_summary or {}

        records = self._validate_records(records, features)
        self._validate_types(records, manifest)
        frame = pd.DataFrame(records, columns=features)

        try:
            predictions = np.asarray(pipeline.predict(frame)).ravel()
        except Exception as exc:  # noqa: BLE001 - surface a stable prediction error
            raise PredictionError(f"model prediction failed: {type(exc).__name__}") from exc

        proba: np.ndarray | None = None
        if task in CLASSIFICATION_TASKS and hasattr(pipeline, "predict_proba"):
            try:
                proba = np.asarray(pipeline.predict_proba(frame), dtype=float)
            except Exception:  # noqa: BLE001 - proba is optional
                proba = None

        labels = list(summary.get("labels") or [])
        positive_label = summary.get("positive_label")
        if task in CLASSIFICATION_TASKS:
            estimator = pipeline["model"] if hasattr(pipeline, "named_steps") else pipeline
            model_classes = list(getattr(estimator, "classes_", []))
            if len(labels) != len(model_classes):
                raise PredictionError("stored labels do not match model classes")
        baseline_frame = self._baseline_frame(summary.get("feature_baseline"), features)

        results: list[dict[str, Any]] = []
        input_hashes: list[str] = []
        events: list[PredictionEvent] = []
        event_time = utcnow()
        for index, record in enumerate(records):
            proba_row = proba[index] if proba is not None else None
            if task in CLASSIFICATION_TASKS:
                result = self._classify_record(
                    pipeline,
                    manifest,
                    int(predictions[index]),
                    proba_row,
                    index,
                    frame,
                    baseline_frame,
                    labels,
                    positive_label,
                )
                probability = result.get("positive_probability")
            else:
                result = self._regress_record(
                    pipeline,
                    manifest,
                    float(predictions[index]),
                    index,
                    frame,
                    baseline_frame,
                    labels,
                    positive_label,
                )
                probability = None
            results.append(result)
            input_hash = _input_hash(record)
            input_hashes.append(input_hash)
            events.append(
                PredictionEvent(
                    registered_model_id=registered.id,
                    model_version=registered.version,
                    asset_id=None,
                    event_time=event_time,
                    prediction_json=result,
                    probability=probability,
                    input_hash=input_hash,
                )
            )

        if persist:
            with self._session_factory.begin() as session:
                self._prediction_repository.bulk_create(events, session=session)
                self._audit.record(
                    actor_type="system",
                    action="prediction.predict",
                    entity_type="registered_model",
                    entity_id=registered.id,
                    payload={
                        "registered_model_id": registered.id,
                        "model_name": registered.name,
                        "model_version": registered.version,
                        "count": len(records),
                        "input_hashes": input_hashes,
                    },
                    session=session,
                )

        return {
            "model_id": registered.id,
            "model_version": registered.version,
            "count": len(records),
            "records": results,
        }

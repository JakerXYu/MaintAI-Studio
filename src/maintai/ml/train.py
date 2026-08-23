"""Deterministic training orchestration.

Accepts a DataFrame + TrainingPlan + split indices, builds an independent
preprocessing pipeline per model (fit on train only), trains, and evaluates on
test. Returns runtime sklearn pipeline objects and a JSON-safe Pydantic summary
separately. A failed model is recorded with a stable error while the remaining
models keep training; if nothing succeeds a ``TrainingError`` is raised.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from maintai.ml import catalog, evaluate, preprocess
from maintai.ml.schemas import (
    CLASSIFICATION_TASKS,
    ModelEvaluation,
    TrainingPlan,
    TrainingResult,
)


class TrainingError(Exception):
    """Raised when a training run cannot produce at least one successful model."""


@dataclass
class TrainedModel:
    """Runtime artifacts for a single fitted model (not JSON-safe)."""

    name: str
    pipeline: Pipeline | None
    feature_names: list[str]
    evaluation: ModelEvaluation


@dataclass
class TrainingOutput:
    """Runtime result: JSON-safe summary plus fitted pipeline objects."""

    result: TrainingResult
    models: dict[str, TrainedModel] = field(default_factory=dict)


def _validate_indices(frame: pd.DataFrame, train_idx: list[int], test_idx: list[int]) -> None:
    if not train_idx or not test_idx:
        raise TrainingError("split indices must not be empty")
    if set(train_idx) & set(test_idx):
        raise TrainingError("train and test indices overlap")
    n = len(frame)
    if any(i < 0 or i >= n for i in train_idx + test_idx):
        raise TrainingError("split indices out of range")


def _prepare_classification_target(
    y: pd.Series, train_idx: list[int], test_idx: list[int]
) -> tuple[pd.Series, pd.Series, list[object], dict[object, int], object | None, float | None]:
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    if bool(y_train.isna().any()):
        raise TrainingError("target contains missing values in the train fold")
    if bool(y_test.isna().any()):
        raise TrainingError("target contains missing values in the test fold")
    classes = sorted(pd.unique(y_train.dropna()), key=str)
    if len(classes) < 2:
        raise TrainingError("classification target has fewer than 2 classes in the train fold")
    unseen = set(pd.unique(y_test.dropna())) - set(classes)
    if unseen:
        raise TrainingError("test fold contains classes absent from the train fold")
    label_to_idx = {c: i for i, c in enumerate(classes)}
    if len(classes) == 2:
        counts = y_train.value_counts()
        if counts.iloc[0] == counts.iloc[-1]:
            positive_label = 1 if 1 in classes else classes[-1]
        else:
            positive_label = counts.idxmin()
        positive_rate = float(counts.min() / counts.sum())
    else:
        positive_label = None
        positive_rate = None
    return y_train, y_test, classes, label_to_idx, positive_label, positive_rate


def _prepare_regression_target(
    y: pd.Series, train_idx: list[int], test_idx: list[int]
) -> tuple[pd.Series, pd.Series]:
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    if bool(y_train.isna().any()) or bool(y_test.isna().any()):
        raise TrainingError("regression target contains missing values")
    try:
        return y_train.astype(float), y_test.astype(float)
    except (TypeError, ValueError) as exc:
        raise TrainingError(f"regression target must be numeric: {exc}") from exc


def _stable_error(name: str, exc: Exception) -> str:
    return f"{name}: {type(exc).__name__}: model training failed"


def _json_scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def train(
    frame: pd.DataFrame,
    plan: TrainingPlan,
    *,
    split=None,
    n_jobs: int = 1,
    scale_numeric: bool = True,
) -> TrainingOutput:
    """Train every requested model and return pipelines + a JSON-safe summary."""
    if not isinstance(plan, TrainingPlan):
        raise TrainingError("plan must be a TrainingPlan")
    target = plan.target
    if target not in frame.columns:
        raise TrainingError(f"target column {target!r} not found in the dataset")

    split = split if split is not None else plan.split
    if split is None:
        raise TrainingError("no split indices provided")
    train_idx = [int(i) for i in split.train_indices]
    test_idx = [int(i) for i in split.test_indices]
    _validate_indices(frame, train_idx, test_idx)

    train_frame = frame.iloc[train_idx]
    feature_cols = preprocess.resolve_features(
        train_frame,
        target,
        plan.features,
        plan.excluded,
    )
    names = (
        list(plan.model_names)
        if plan.model_names
        else list(catalog.model_names_for_task(plan.task))
    )

    y = frame[target]
    classes: list[object] = []
    positive_label: object | None = None
    positive_rate: float | None = None
    if plan.task in CLASSIFICATION_TASKS:
        y_train_raw, y_test_raw, classes, label_to_idx, positive_label, positive_rate = (
            _prepare_classification_target(y, train_idx, test_idx)
        )
        y_train = np.array([label_to_idx[c] for c in y_train_raw])
        y_test = y_test_raw.to_numpy()
        primary_metric = plan.primary_metric or evaluate.resolve_primary_metric(
            plan.task, positive_rate=positive_rate
        )
    else:
        y_train_raw, y_test_raw = _prepare_regression_target(y, train_idx, test_idx)
        y_train = y_train_raw.to_numpy(dtype=float)
        y_test = y_test_raw.to_numpy(dtype=float)
        primary_metric = plan.primary_metric or evaluate.resolve_primary_metric(plan.task)

    X_train = frame.iloc[train_idx].loc[:, feature_cols]
    X_test = frame.iloc[test_idx].loc[:, feature_cols]

    evaluations: list[ModelEvaluation] = []
    models: dict[str, TrainedModel] = {}
    errors: list[str] = []

    for name in names:
        try:
            preprocessor = preprocess.build_preprocessor(
                X_train, feature_cols, scale_numeric=scale_numeric
            )
            preprocessor.fit(X_train)
            feature_names = preprocess.get_feature_names(preprocessor)
            estimator = catalog.build_model(name, plan.task, seed=plan.seed, n_jobs=n_jobs)
            pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            if plan.task in CLASSIFICATION_TASKS:
                try:
                    proba = pipeline.predict_proba(X_test)
                except Exception:  # noqa: BLE001
                    proba = None
                y_pred_decoded = np.array([classes[int(i)] for i in y_pred])
                ev = evaluate.evaluate_classification(
                    y_test,
                    y_pred_decoded,
                    y_proba=proba,
                    positive_label=positive_label,
                    labels=classes,
                )
            else:
                ev = evaluate.evaluate_regression(y_test, y_pred)

            latency_ms = evaluate.measure_inference_latency_ms(pipeline, X_test)
            evaluation = ModelEvaluation(
                model_name=name,
                task=plan.task,
                status="success",
                metrics=ev.metrics,
                confusion_matrix=ev.confusion_matrix,
                labels=[_json_scalar(label) for label in classes],
                positive_label=_json_scalar(positive_label),
                inference_latency_ms=latency_ms,
            )
            models[name] = TrainedModel(
                name=name,
                pipeline=pipeline,
                feature_names=feature_names,
                evaluation=evaluation,
            )
            evaluations.append(evaluation)
        except Exception as exc:  # noqa: BLE001
            message = _stable_error(name, exc)
            errors.append(message)
            evaluations.append(
                ModelEvaluation(model_name=name, task=plan.task, status="error", error=message)
            )

    if not any(e.status == "success" for e in evaluations):
        raise TrainingError("no model trained successfully: " + "; ".join(errors))

    if primary_metric == "pr_auc" and not any(
        evaluation.status == "success" and evaluation.metrics.get("pr_auc") is not None
        for evaluation in evaluations
    ):
        primary_metric = "f1"

    result = TrainingResult(
        task=plan.task,
        target=target,
        seed=plan.seed,
        primary_metric=primary_metric,
        evaluations=evaluations,
        errors=errors,
    )
    return TrainingOutput(result=result, models=models)

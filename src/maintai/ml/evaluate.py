"""Deterministic evaluation metrics for classification and regression.

Classification: precision / recall / F1 / ROC-AUC / PR-AUC / confusion matrix.
The imbalance-aware rule makes PR-AUC the primary metric when the positive rate
is below 10%, otherwise F1. Regression: MAE / RMSE / R2 with RMSE as primary.
Missing probability estimates or single-class test sets degrade gracefully to
``None`` instead of crashing.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import (
    confusion_matrix as _confusion_matrix,
)

_PRIMARY_WHEN_IMBALANCED = 0.10


@dataclass
class Evaluation:
    """A deterministic evaluation result (metrics + confusion matrix)."""

    metrics: dict[str, float | None]
    confusion_matrix: list[list[int]] | None = None


def _union_labels(*arrays) -> list[object]:
    values: set[object] = set()
    for array in arrays:
        for value in np.unique(array):
            values.add(value)
    return sorted(values, key=str)


def _positive_probabilities(proba, labels, positive_label) -> np.ndarray | None:
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        return proba
    if proba.ndim == 2 and proba.shape[1] == 1:
        return proba[:, 0]
    if proba.ndim == 2 and positive_label in labels:
        return proba[:, labels.index(positive_label)]
    return None


def evaluate_classification(
    y_true,
    y_pred,
    *,
    y_proba=None,
    positive_label=None,
    labels=None,
) -> Evaluation:
    """Evaluate a classifier; proba-based metrics degrade to None when unavailable."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = _union_labels(y_true, y_pred)
    else:
        labels = list(labels)
    if not labels:
        return Evaluation(metrics={}, confusion_matrix=None)

    n_classes = len(labels)
    with warnings.catch_warnings():
        if n_classes == 1:
            warnings.simplefilter("ignore", UserWarning)
        cm = _confusion_matrix(y_true, y_pred, labels=labels)

    if n_classes == 2:
        positive = positive_label if positive_label in labels else labels[1]
        metrics: dict[str, float | None] = {
            "precision": float(
                precision_score(y_true, y_pred, pos_label=positive, zero_division=0)
            ),
            "recall": float(recall_score(y_true, y_pred, pos_label=positive, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, pos_label=positive, zero_division=0)),
        }
    else:
        metrics = {
            "precision": float(
                precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
            ),
            "f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        }

    roc_auc: float | None = None
    pr_auc: float | None = None
    if n_classes == 2 and y_proba is not None:
        positive = positive_label if positive_label in labels else labels[1]
        pos_proba = _positive_probabilities(y_proba, labels, positive)
        present = set(np.unique(y_true))
        if pos_proba is not None and len(present) == 2:
            y_binary = (y_true == positive).astype(int)
            try:
                roc_auc = float(roc_auc_score(y_binary, pos_proba))
            except ValueError:
                roc_auc = None
            try:
                pr_auc = float(average_precision_score(y_binary, pos_proba))
            except ValueError:
                pr_auc = None

    metrics["roc_auc"] = roc_auc
    metrics["pr_auc"] = pr_auc
    return Evaluation(metrics=metrics, confusion_matrix=cm.tolist())


def evaluate_regression(y_true, y_pred) -> Evaluation:
    """Evaluate a regressor with MAE / RMSE / R2."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return Evaluation(metrics={"mae": mae, "rmse": rmse, "r2": r2})


def resolve_primary_metric(task: str, *, positive_rate=None, metrics=None) -> str:
    """Resolve the primary metric deterministically from the task and imbalance."""
    if task == "regression":
        return "rmse"
    if task == "multiclass_classification":
        return "f1"
    if positive_rate is not None and positive_rate < _PRIMARY_WHEN_IMBALANCED:
        if metrics is None or metrics.get("pr_auc") is not None:
            return "pr_auc"
    return "f1"


def measure_inference_latency_ms(pipeline, X, *, n_repeats: int = 3) -> float:
    """Measure mean inference latency in milliseconds (informational only)."""
    pipeline.predict(X)  # warm-up call outside the timed window
    start = time.perf_counter()
    for _ in range(n_repeats):
        pipeline.predict(X)
    elapsed = time.perf_counter() - start
    return elapsed / n_repeats * 1000.0

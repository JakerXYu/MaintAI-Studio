"""Evaluation metric correctness and graceful single-class / no-proba handling."""

import numpy as np
import pytest

from maintai.ml.evaluate import (
    evaluate_classification,
    evaluate_regression,
    resolve_primary_metric,
)


def test_classification_metrics_correctness():
    y_true = [1, 0, 1, 1, 0, 0]
    y_pred = [1, 0, 0, 1, 0, 1]
    ev = evaluate_classification(y_true, y_pred, positive_label=1)
    assert ev.metrics["precision"] == pytest.approx(2 / 3)
    assert ev.metrics["recall"] == pytest.approx(2 / 3)
    assert ev.metrics["f1"] == pytest.approx(2 / 3)
    assert ev.metrics["roc_auc"] is None
    assert ev.metrics["pr_auc"] is None
    assert ev.confusion_matrix == [[2, 1], [1, 2]]


def test_roc_pr_auc_perfect_separation():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    proba = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    ev = evaluate_classification(y_true, y_pred, y_proba=proba, positive_label=1)
    assert ev.metrics["roc_auc"] == pytest.approx(1.0)
    assert ev.metrics["pr_auc"] == pytest.approx(1.0)


def test_no_probability_returns_none_auc():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    ev = evaluate_classification(y_true, y_pred, y_proba=None, positive_label=1)
    assert ev.metrics["f1"] == pytest.approx(1.0)
    assert ev.metrics["roc_auc"] is None
    assert ev.metrics["pr_auc"] is None


def test_single_class_graceful():
    y_true = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 0]
    ev = evaluate_classification(y_true, y_pred)
    assert ev.metrics["f1"] == pytest.approx(1.0)
    assert ev.metrics["roc_auc"] is None
    assert ev.metrics["pr_auc"] is None


def test_single_class_with_proba_graceful():
    y_true = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 0]
    proba = np.array([[0.9, 0.1]] * 4)
    ev = evaluate_classification(y_true, y_pred, y_proba=proba, positive_label=1)
    assert ev.metrics["roc_auc"] is None
    assert ev.metrics["pr_auc"] is None


def test_regression_metrics_correctness():
    y_true = [3.0, -0.5, 2.0, 7.0]
    y_pred = [2.5, 0.0, 2.0, 8.0]
    ev = evaluate_regression(y_true, y_pred)
    assert ev.metrics["mae"] == pytest.approx(0.5)
    assert ev.metrics["rmse"] == pytest.approx(np.sqrt(0.375))
    assert ev.metrics["r2"] > 0.9


def test_primary_metric_imbalance_rule():
    assert resolve_primary_metric("binary_classification", positive_rate=0.05) == "pr_auc"
    assert resolve_primary_metric("binary_classification", positive_rate=0.5) == "f1"
    assert resolve_primary_metric("multiclass_classification") == "f1"
    assert resolve_primary_metric("regression") == "rmse"

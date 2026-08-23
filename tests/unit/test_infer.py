"""Task-inference tests (synthetic data only)."""

import numpy as np
import pandas as pd

from maintai.tasks.infer import recommend_task


def test_binary_classification():
    frame = pd.DataFrame({"target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})
    rec = recommend_task(frame, target_col="target")
    assert rec.recommended_task == "binary_classification"
    assert rec.trainable is True
    assert 0.0 <= rec.confidence <= 1.0
    assert rec.evidence


def test_multiclass_classification():
    frame = pd.DataFrame({"target": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]})
    rec = recommend_task(frame, target_col="target")
    assert rec.recommended_task == "multiclass_classification"


def test_categorical_string_target_is_classification():
    frame = pd.DataFrame({"target": ["ok", "fail", "ok", "fail", "ok", "fail"]})
    rec = recommend_task(frame, target_col="target")
    assert rec.recommended_task == "binary_classification"


def test_regression():
    frame = pd.DataFrame({"target": np.linspace(0.0, 100.0, 50)})
    rec = recommend_task(frame, target_col="target")
    assert rec.recommended_task == "regression"
    assert rec.trainable is True


def test_regression_offers_multiclass_alternative_for_low_unique():
    frame = pd.DataFrame({"target": np.arange(40, dtype=float)})
    rec = recommend_task(frame, target_col="target")
    assert rec.recommended_task == "regression"
    assert any(a.task == "multiclass_classification" for a in rec.alternatives)


def test_no_target_returns_insufficient_target():
    frame = pd.DataFrame({"feature": [1, 2, 3, 4]})
    rec = recommend_task(frame)
    assert rec.recommended_task == "insufficient_target"
    assert rec.trainable is False


def test_single_class_not_trainable():
    frame = pd.DataFrame({"target": [1, 1, 1, 1]})
    rec = recommend_task(frame, target_col="target")
    assert rec.recommended_task == "single_class"
    assert rec.trainable is False


def test_missing_target_column_is_insufficient():
    frame = pd.DataFrame({"feature": [1, 2, 3, 4]})
    rec = recommend_task(frame, target_col="does_not_exist")
    assert rec.recommended_task == "insufficient_target"

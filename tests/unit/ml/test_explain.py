"""Explanation tests: SHAP (tree/linear) global+local, fallback, disclaimer, JSON-safe."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from maintai.ml import explain
from maintai.ml.preprocess import build_preprocessor

DISCLAIMER = "Model-based explanation, not a verified physical root cause."

CLASSIFICATION_FEATURES = ["num1", "num2", "cat", "flag"]
REGRESSION_FEATURES = ["num1", "num2", "cat"]


def _classification_frame(n=120, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[: n // 2] = 1
    rng.shuffle(y)
    return pd.DataFrame(
        {
            "num1": rng.normal(0, 1, n) + y * 2.0,
            "num2": rng.normal(0, 1, n) - y * 1.0,
            "cat": rng.choice(["a", "b", "c"], n),
            "flag": rng.choice([True, False], n),
            "target": y,
        }
    )


def _regression_frame(n=120, seed=0):
    rng = np.random.default_rng(seed)
    target = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "num1": target + rng.normal(0, 0.1, n),
            "num2": rng.normal(0, 1, n),
            "cat": rng.choice(["a", "b", "c"], n),
            "target": target,
        }
    )


def _fit(frame, estimator, features):
    X = frame[features]
    y = frame["target"]
    preprocessor = build_preprocessor(X, features)
    preprocessor.fit(X)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    pipeline.fit(X, y)
    return pipeline, X, y


def _shap_explodes(*args, **kwargs):
    raise RuntimeError("forced SHAP failure")


def test_tree_classification_global_and_local():
    frame = _classification_frame()
    pipe, X, _ = _fit(
        frame, RandomForestClassifier(n_estimators=20, random_state=42), CLASSIFICATION_FEATURES
    )
    labels = [0, 1]

    global_result = explain.global_explanation(
        pipe,
        X,
        task="binary_classification",
        positive_label=1,
        labels=labels,
        sample_size=60,
        background_size=40,
    )
    assert global_result.method == explain.SHAP_METHOD_TREE
    assert global_result.top_features
    assert all(feature.impact >= 0 for feature in global_result.top_features)
    assert global_result.disclaimer == DISCLAIMER

    local_result = explain.local_explanation(
        pipe,
        X.iloc[[0]],
        background=X,
        task="binary_classification",
        positive_label=1,
        labels=labels,
    )
    assert local_result.method == explain.SHAP_METHOD_TREE
    assert local_result.prediction in (0, 1)
    expected_risk = float(pipe.predict_proba(X.iloc[[0]])[0, 1])
    assert local_result.positive_probability == pytest.approx(expected_risk)
    assert local_result.disclaimer == DISCLAIMER


def test_linear_classification_global_and_local():
    frame = _classification_frame()
    pipe, X, _ = _fit(
        frame, LogisticRegression(max_iter=1000, random_state=42), CLASSIFICATION_FEATURES
    )
    labels = [0, 1]

    global_result = explain.global_explanation(
        pipe,
        X,
        task="binary_classification",
        positive_label=1,
        labels=labels,
        sample_size=60,
        background_size=40,
    )
    assert global_result.method == explain.SHAP_METHOD_LINEAR
    assert global_result.top_features
    assert global_result.disclaimer == DISCLAIMER

    local_result = explain.local_explanation(
        pipe,
        X.iloc[[0]],
        background=X,
        task="binary_classification",
        positive_label=1,
        labels=labels,
    )
    assert local_result.method == explain.SHAP_METHOD_LINEAR
    assert local_result.prediction in (0, 1)
    assert local_result.disclaimer == DISCLAIMER


def test_tree_regression_global_and_local():
    frame = _regression_frame()
    pipe, X, _ = _fit(
        frame, RandomForestRegressor(n_estimators=20, random_state=42), REGRESSION_FEATURES
    )
    global_result = explain.global_explanation(
        pipe, X, task="regression", sample_size=60, background_size=40
    )
    assert global_result.method == explain.SHAP_METHOD_TREE
    assert global_result.top_features

    local_result = explain.local_explanation(
        pipe, X.iloc[[0]], background=X, task="regression"
    )
    assert local_result.method == explain.SHAP_METHOD_TREE
    assert isinstance(local_result.prediction, float)
    assert local_result.positive_probability is None


def test_linear_regression_global_and_local():
    frame = _regression_frame()
    pipe, X, _ = _fit(frame, Ridge(alpha=1.0), REGRESSION_FEATURES)
    global_result = explain.global_explanation(
        pipe, X, task="regression", sample_size=60, background_size=40
    )
    assert global_result.method == explain.SHAP_METHOD_LINEAR
    assert global_result.top_features

    local_result = explain.local_explanation(
        pipe, X.iloc[[0]], background=X, task="regression"
    )
    assert local_result.method == explain.SHAP_METHOD_LINEAR
    assert isinstance(local_result.prediction, float)
    assert local_result.positive_probability is None


def test_global_fallback_to_permutation(monkeypatch):
    frame = _classification_frame()
    pipe, X, y = _fit(
        frame, RandomForestClassifier(n_estimators=20, random_state=42), CLASSIFICATION_FEATURES
    )
    monkeypatch.setattr(explain, "_compute_shap", _shap_explodes)
    result = explain.global_explanation(
        pipe,
        X,
        y=y,
        task="binary_classification",
        positive_label=1,
        labels=[0, 1],
    )
    assert result.method == explain.PERMUTATION_METHOD
    assert result.top_features


def test_local_fallback_to_perturbation(monkeypatch):
    frame = _classification_frame()
    pipe, X, _ = _fit(
        frame, RandomForestClassifier(n_estimators=20, random_state=42), CLASSIFICATION_FEATURES
    )
    monkeypatch.setattr(explain, "_compute_shap", _shap_explodes)
    result = explain.local_explanation(
        pipe,
        X.iloc[[0]],
        background=X,
        task="binary_classification",
        positive_label=1,
        labels=[0, 1],
    )
    assert result.method == explain.PERTURBATION_METHOD


def test_label_count_mismatch_is_rejected():
    frame = _classification_frame()
    pipe, X, _ = _fit(
        frame,
        LogisticRegression(max_iter=1000, random_state=42),
        CLASSIFICATION_FEATURES,
    )
    with pytest.raises(explain.ExplanationError, match="labels length"):
        explain.local_explanation(
            pipe,
            X.iloc[[0]],
            background=X,
            task="binary_classification",
            labels=["only-one-label"],
        )


def test_global_fallback_requires_y(monkeypatch):
    frame = _classification_frame()
    pipe, X, _ = _fit(
        frame, RandomForestClassifier(n_estimators=20, random_state=42), CLASSIFICATION_FEATURES
    )
    monkeypatch.setattr(explain, "_compute_shap", _shap_explodes)
    with pytest.raises(explain.ExplanationError, match="no y"):
        explain.global_explanation(
            pipe, X, task="binary_classification", positive_label=1, labels=[0, 1]
        )


def test_local_explanation_unknown_category():
    frame = pd.DataFrame(
        {
            "num": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "cat": ["a", "b", "a", "b", "a", "b"],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    pipe, X, _ = _fit(frame, LogisticRegression(max_iter=1000), ["num", "cat"])
    unknown = pd.DataFrame({"num": [0.5], "cat": ["c"]})
    result = explain.local_explanation(
        pipe,
        unknown,
        background=X,
        task="binary_classification",
        positive_label=1,
        labels=[0, 1],
    )
    assert result.method == explain.SHAP_METHOD_LINEAR
    assert result.prediction in (0, 1)


def test_disclaimer_constant():
    assert explain.DISCLAIMER == DISCLAIMER


def test_explanations_are_json_safe():
    frame = _classification_frame()
    pipe, X, _ = _fit(
        frame, LogisticRegression(max_iter=1000, random_state=42), CLASSIFICATION_FEATURES
    )
    global_result = explain.global_explanation(
        pipe, X, task="binary_classification", positive_label=1, labels=[0, 1]
    )
    assert isinstance(global_result.model_dump_json(), str)
    data = global_result.model_dump(mode="json")
    for feature in data["top_features"]:
        assert isinstance(feature["feature"], str)
        assert isinstance(feature["impact"], float)

    local_result = explain.local_explanation(
        pipe,
        X.iloc[[0]],
        background=X,
        task="binary_classification",
        positive_label=1,
        labels=[0, 1],
    )
    assert isinstance(local_result.model_dump_json(), str)


def test_feature_name_mismatch_raises():
    frame = _classification_frame()
    pipe, X, _ = _fit(
        frame, LogisticRegression(max_iter=1000), CLASSIFICATION_FEATURES
    )
    with pytest.raises(explain.ExplanationError, match="feature names"):
        explain.global_explanation(pipe, X, feature_names=["only_one"])

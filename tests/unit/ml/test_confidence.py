"""Confidence tests: classification label/proba, unknown categories, regression interval."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from maintai.ml.confidence import (
    EMPIRICAL_DISCLAIMER,
    ClassificationPrediction,
    ConfidenceError,
    EmpiricalIntervalCalibrator,
    RegressionInterval,
    classify,
)
from maintai.ml.preprocess import build_preprocessor


def _fit_classifier(n=160, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[: n // 2] = 1
    rng.shuffle(y)
    frame = pd.DataFrame(
        {
            "num": rng.normal(0, 1, n) + y * 2.0,
            "cat": rng.choice(["a", "b"], n),
            "target": y,
        }
    )
    features = ["num", "cat"]
    X = frame[features]
    preprocessor = build_preprocessor(X, features)
    preprocessor.fit(X)
    model = LogisticRegression(max_iter=1000, random_state=42).fit(
        preprocessor.transform(X), y
    )
    return model, preprocessor, X, y


def test_classification_label_and_proba_mapping():
    model, preprocessor, X, _ = _fit_classifier()
    decoded = ["normal", "anomaly"]  # 0 -> normal, 1 -> anomaly
    sample = preprocessor.transform(X.iloc[[0]])
    result = classify(model, sample, labels=decoded, positive_label="anomaly")

    assert isinstance(result, ClassificationPrediction)
    assert result.prediction in ("normal", "anomaly")
    assert result.confidence is not None and 0.0 <= result.confidence <= 1.0
    assert set(result.probabilities) == {"normal", "anomaly"}
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    # The predicted label's confidence must match its own probability entry.
    assert result.confidence == pytest.approx(result.probabilities[str(result.prediction)])
    assert result.positive_label == "anomaly"


def test_classification_unknown_category_does_not_crash():
    model, preprocessor, X, _ = _fit_classifier()
    decoded = ["normal", "anomaly"]
    unknown = pd.DataFrame({"num": [0.5], "cat": ["zzz"]})
    sample = preprocessor.transform(unknown)
    result = classify(model, sample, labels=decoded)
    assert result.prediction in ("normal", "anomaly")
    assert result.confidence is not None
    assert set(result.probabilities) == {"normal", "anomaly"}


def test_classification_2d_sample_accepted():
    model, preprocessor, X, _ = _fit_classifier()
    sample = preprocessor.transform(X.iloc[[0]])
    # A 1-D row must be accepted too.
    result = classify(model, sample.ravel(), labels=["normal", "anomaly"])
    assert result.prediction in ("normal", "anomaly")


def test_classify_label_length_mismatch_rejected():
    model, preprocessor, X, _ = _fit_classifier()
    sample = preprocessor.transform(X.iloc[[0]])
    with pytest.raises(ConfidenceError, match="labels length"):
        classify(model, sample, labels=["only_one"])


def test_classify_multiple_samples_rejected():
    model, preprocessor, X, _ = _fit_classifier()
    sample = preprocessor.transform(X.iloc[:2])
    with pytest.raises(ConfidenceError, match="exactly one sample"):
        classify(model, sample, labels=["normal", "anomaly"])


def test_regression_interval_empirical_quantile():
    calibrator = EmpiricalIntervalCalibrator(coverage=0.9)
    y_true = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    y_pred = np.zeros(10)
    calibrator.fit(y_true, y_pred)

    quantile = float(np.quantile(np.abs(y_true - y_pred), 0.9))
    interval = calibrator.interval(5.0)
    assert isinstance(interval, RegressionInterval)
    assert interval.prediction == pytest.approx(5.0)
    assert interval.lower == pytest.approx(5.0 - quantile)
    assert interval.upper == pytest.approx(5.0 + quantile)
    assert interval.coverage == 0.9
    assert interval.disclaimer == EMPIRICAL_DISCLAIMER


def test_regression_interval_is_fitted_flag():
    calibrator = EmpiricalIntervalCalibrator(coverage=0.8)
    assert not calibrator.is_fitted()
    calibrator.fit([1.0, 2.0, 3.0], [1.1, 2.2, 2.9])
    assert calibrator.is_fitted()
    assert calibrator.n_ == 3


def test_invalid_coverage_rejected():
    for bad in (0.0, 1.0, -0.5, 1.5, "x", True):
        with pytest.raises(ConfidenceError):
            EmpiricalIntervalCalibrator(coverage=bad)


def test_fit_empty_residuals_rejected():
    with pytest.raises(ConfidenceError, match="empty"):
        EmpiricalIntervalCalibrator().fit([], [])


def test_fit_length_mismatch_rejected():
    with pytest.raises(ConfidenceError, match="same length"):
        EmpiricalIntervalCalibrator().fit([1.0, 2.0, 3.0], [1.0, 2.0])


def test_fit_non_finite_rejected():
    with pytest.raises(ConfidenceError, match="finite"):
        EmpiricalIntervalCalibrator().fit([1.0, np.nan], [1.0, 2.0])


def test_interval_before_fit_rejected():
    with pytest.raises(ConfidenceError, match="must be fitted"):
        EmpiricalIntervalCalibrator().interval(1.0)


def test_interval_non_numeric_prediction_rejected():
    calibrator = EmpiricalIntervalCalibrator()
    calibrator.fit([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ConfidenceError, match="numeric"):
        calibrator.interval("not-a-number")

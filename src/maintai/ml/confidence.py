"""Deterministic confidence outputs and empirical residual intervals.

Classification returns the decoded prediction plus the ``predict_proba``
probability as a confidence score, with correct label mapping. Regression does
**not** fabricate probabilities: it calibrates an empirical prediction interval
from the absolute residuals of a validation/test fold. That interval is always
labelled ``empirical prediction interval; not a statistical guarantee``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

EMPIRICAL_DISCLAIMER = "empirical prediction interval; not a statistical guarantee"


class ConfidenceError(Exception):
    """Raised when confidence inputs are invalid or a calibrator is misused."""


class ClassificationPrediction(BaseModel):
    """Decoded prediction with ``predict_proba`` confidence and label mapping."""

    prediction: str | int | float | bool | None
    confidence: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    positive_label: str | int | float | bool | None = None


class RegressionInterval(BaseModel):
    """Empirical prediction interval from calibrated absolute residuals."""

    prediction: float
    lower: float
    upper: float
    coverage: float
    disclaimer: str = EMPIRICAL_DISCLAIMER


def _as_single_sample(x) -> np.ndarray:
    """Normalise a single sample to a ``(1, n_features)`` float matrix."""
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        arr = x.to_numpy(dtype=float)
    else:
        arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] != 1:
        raise ConfidenceError("confidence requires exactly one sample")
    if not np.all(np.isfinite(arr)):
        raise ConfidenceError("sample contains non-finite values")
    return arr


def classify(
    model,
    x,
    *,
    labels,
    positive_label=None,
) -> ClassificationPrediction:
    """Return the decoded prediction and ``predict_proba`` confidence for one sample.

    ``labels`` must list the decoded class labels in the same order as
    ``model.classes_``. ``x`` is a single preprocessed sample.
    """
    if not hasattr(model, "predict"):
        raise ConfidenceError("model must provide a predict method")
    model_classes = getattr(model, "classes_", None)
    classes = list(model_classes) if model_classes is not None else []
    decoded = list(labels)
    if len(decoded) != len(classes):
        raise ConfidenceError(
            f"labels length {len(decoded)} does not match model classes {len(classes)}"
        )

    sample = _as_single_sample(x)
    try:
        encoded = int(np.asarray(model.predict(sample)).ravel()[0])
    except Exception as exc:  # noqa: BLE001
        raise ConfidenceError(f"model prediction failed: {exc}") from exc

    prediction = decoded[encoded] if 0 <= encoded < len(decoded) else encoded

    confidence: float | None = None
    probabilities: dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        try:
            proba = np.asarray(model.predict_proba(sample), dtype=float)
        except Exception:  # noqa: BLE001 - proba is optional
            proba = None
        if proba is not None:
            if proba.ndim == 1:
                proba = proba.reshape(1, -1)
            if proba.ndim == 2 and proba.shape[1] == len(classes):
                row = proba[0]
                for index, value in enumerate(row):
                    probabilities[str(decoded[index])] = float(value)
                if 0 <= encoded < len(row):
                    confidence = float(row[encoded])

    return ClassificationPrediction(
        prediction=prediction,
        confidence=confidence,
        probabilities=probabilities,
        positive_label=positive_label,
    )


class EmpiricalIntervalCalibrator:
    """Calibrate an empirical prediction interval from absolute residuals.

    ``coverage`` is the target quantile of the absolute residual distribution
    (e.g. ``0.9`` → the 90th percentile). The resulting interval is an empirical
    estimate and is explicitly **not** a statistical guarantee.
    """

    def __init__(self, coverage: float = 0.9):
        if isinstance(coverage, bool) or not isinstance(coverage, (int, float)):
            raise ConfidenceError("coverage must be a number in (0, 1)")
        coverage = float(coverage)
        if not 0.0 < coverage < 1.0:
            raise ConfidenceError("coverage must be in (0, 1)")
        self.coverage = coverage
        self.absolute_error_quantile_: float | None = None
        self.n_: int = 0

    def fit(self, y_true, y_pred) -> EmpiricalIntervalCalibrator:
        """Fit on validation/test residuals (must be 1-D and equal length)."""
        if isinstance(y_true, pd.Series):
            y_true = y_true.to_numpy(dtype=float)
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.to_numpy(dtype=float)
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        if y_true.shape[0] == 0:
            raise ConfidenceError("cannot fit a calibrator on empty residuals")
        if y_true.shape[0] != y_pred.shape[0]:
            raise ConfidenceError("y_true and y_pred must have the same length")
        if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
            raise ConfidenceError("residuals must be finite")
        residuals = np.abs(y_true - y_pred)
        self.absolute_error_quantile_ = float(np.quantile(residuals, self.coverage))
        self.n_ = int(y_true.shape[0])
        return self

    def is_fitted(self) -> bool:
        return self.absolute_error_quantile_ is not None

    def interval(self, prediction) -> RegressionInterval:
        """Return the empirical interval ``[prediction - q, prediction + q]``."""
        if not self.is_fitted():
            raise ConfidenceError("calibrator must be fitted before producing intervals")
        try:
            pred = float(prediction)
        except (TypeError, ValueError) as exc:
            raise ConfidenceError(f"prediction must be numeric: {exc}") from exc
        if not np.isfinite(pred):
            raise ConfidenceError("prediction must be finite")
        q = float(self.absolute_error_quantile_)
        return RegressionInterval(
            prediction=pred,
            lower=pred - q,
            upper=pred + q,
            coverage=self.coverage,
        )

    def predict_interval(self, prediction) -> RegressionInterval:
        """Alias of :meth:`interval` for symmetry with estimator APIs."""
        return self.interval(prediction)

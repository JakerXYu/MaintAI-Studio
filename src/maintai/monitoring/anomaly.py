"""Deterministic anomaly detection: robust median/MAD Z-score + IsolationForest.

The baseline and production paths are separate: robust statistics and the
IsolationForest model are fit only on the baseline, so production data never
leaks into the detectors. Only numeric columns (an explicit allowlist) are
scored. Missing values are imputed deterministically with the baseline median.
MAD=0 columns fall back to the standard deviation; a fully constant column
contributes no signal (constant-safe).

No labels are consumed, so accuracy/precision/recall are deliberately NOT
claimed — the result only carries scores, flags, and the top deviating features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator
from sklearn.ensemble import IsolationForest

from maintai.monitoring.contracts import AnomalyResult, AnomalyRow, DeviatingFeature

_MAD_SCALE = 0.6745  # converts MAD to a standard-deviation-equivalent for N(0,1)
_MIN_BASELINE_ROWS = 4


class AnomalyError(Exception):
    """Raised when a usable numeric baseline cannot be established."""


class AnomalyConfig(BaseModel):
    """Demo-default anomaly thresholds (mirrors ``configs/default.yaml``)."""

    seed: int = 42
    contamination: float | str = "auto"
    threshold: float = 3.0
    top_k_features: int = 3
    n_estimators: int = 100

    @field_validator("threshold")
    @classmethod
    def _positive_threshold(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("threshold must be > 0")
        return value

    @field_validator("contamination")
    @classmethod
    def _valid_contamination(cls, value: float | str) -> float | str:
        if isinstance(value, str):
            if value != "auto":
                raise ValueError("contamination must be 'auto' or a float in (0, 0.5]")
            return value
        if not 0 < value <= 0.5:
            raise ValueError("contamination must be in (0, 0.5]")
        return value

    @field_validator("top_k_features")
    @classmethod
    def _positive_top_k(cls, value: int) -> int:
        if value < 1:
            raise ValueError("top_k_features must be >= 1")
        return value

    @field_validator("n_estimators")
    @classmethod
    def _positive_estimators(cls, value: int) -> int:
        if value < 1:
            raise ValueError("n_estimators must be >= 1")
        return value


@dataclass
class AnomalyBaseline:
    """Fitted state from a baseline window (fit only on baseline data)."""

    numeric_features: list[str]
    medians: dict[str, float]
    mads: dict[str, float]
    stds: dict[str, float]
    model: IsolationForest
    contamination: float | str
    # score-distribution stats used to normalise raw scores into combined units
    robust_score_center: float
    robust_score_scale: float
    iforest_score_center: float
    iforest_score_scale: float
    flag_threshold: float
    threshold_source: str


def _numeric_columns(frame: pd.DataFrame, numeric_columns: list[str] | None) -> list[str]:
    if numeric_columns is not None:
        missing = [c for c in numeric_columns if c not in frame.columns]
        if missing:
            raise AnomalyError(f"numeric column(s) not found in data: {missing}")
        non_numeric = [
            c
            for c in numeric_columns
            if not pd.api.types.is_numeric_dtype(frame[c].dtype)
            or pd.api.types.is_bool_dtype(frame[c].dtype)
        ]
        if non_numeric:
            raise AnomalyError(f"columns are not numeric: {non_numeric}")
        return list(numeric_columns)
    return [
        c
        for c in frame.columns
        if pd.api.types.is_numeric_dtype(frame[c].dtype)
        and not pd.api.types.is_bool_dtype(frame[c].dtype)
    ]


def _to_numeric_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.to_numeric(frame[c], errors="coerce") for c in features})


def _robust_z_matrix(
    X: np.ndarray,
    features: list[str],
    medians: dict[str, float],
    mads: dict[str, float],
    stds: dict[str, float],
) -> np.ndarray:
    """Return the robust Z-score matrix (rows x features), constant-safe."""
    n_rows, n_cols = X.shape
    Z = np.zeros((n_rows, n_cols), dtype=float)
    for j, column in enumerate(features):
        median = medians[column]
        mad = mads[column]
        std = stds[column]
        column_values = X[:, j]
        if mad > 0:
            Z[:, j] = _MAD_SCALE * (column_values - median) / mad
        elif std > 0:
            Z[:, j] = (column_values - median) / std
        # else: constant baseline feature -> contributes no signal (constant-safe)
    return Z


def _score_scale(values: np.ndarray) -> float:
    """A robust scale for a 1-D score distribution (MAD, else std, else 1.0)."""
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    if mad > 0:
        return mad / _MAD_SCALE
    std = float(np.std(values))
    if std > 0:
        return std
    return 1.0


def fit_baseline(
    frame: pd.DataFrame,
    *,
    numeric_columns: list[str] | None = None,
    config: AnomalyConfig | None = None,
) -> AnomalyBaseline:
    """Fit robust statistics and IsolationForest on the baseline window only."""
    config = config or AnomalyConfig()
    features = _numeric_columns(frame, numeric_columns)
    if not features:
        raise AnomalyError("no numeric columns available for anomaly detection")
    if len(frame) < _MIN_BASELINE_ROWS:
        raise AnomalyError(
            f"baseline has {len(frame)} rows; need at least {_MIN_BASELINE_ROWS}"
        )

    numeric = _to_numeric_matrix(frame, features)
    medians: dict[str, float] = {}
    mads: dict[str, float] = {}
    stds: dict[str, float] = {}
    for column in features:
        values = numeric[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            # Entirely-missing column: treat as constant so imputation stays finite.
            medians[column] = 0.0
            mads[column] = 0.0
            stds[column] = 0.0
            continue
        median = float(np.median(finite))
        medians[column] = median
        mads[column] = float(np.median(np.abs(finite - median)))
        stds[column] = float(np.std(finite))

    imputed = numeric.fillna({c: medians[c] for c in features})
    X = imputed.to_numpy(dtype=float)

    model = IsolationForest(
        random_state=config.seed,
        contamination=config.contamination,
        n_estimators=config.n_estimators,
    )
    model.fit(X)

    Z_base = _robust_z_matrix(X, features, medians, mads, stds)
    robust_base = np.max(np.abs(Z_base), axis=1)
    iforest_base = -model.decision_function(X)

    robust_center = float(np.median(robust_base))
    robust_scale = _score_scale(robust_base)
    iforest_center = float(np.median(iforest_base))
    iforest_scale = _score_scale(iforest_base)
    combined_base = 0.5 * (
        (robust_base - robust_center) / robust_scale
        + (iforest_base - iforest_center) / iforest_scale
    )
    if isinstance(config.contamination, float):
        flag_threshold = float(np.quantile(combined_base, 1.0 - config.contamination))
        threshold_source = "baseline contamination quantile"
    else:
        flag_threshold = config.threshold
        threshold_source = "configured combined-score threshold"

    return AnomalyBaseline(
        numeric_features=features,
        medians=medians,
        mads=mads,
        stds=stds,
        model=model,
        contamination=config.contamination,
        robust_score_center=robust_center,
        robust_score_scale=robust_scale,
        iforest_score_center=iforest_center,
        iforest_score_scale=iforest_scale,
        flag_threshold=flag_threshold,
        threshold_source=threshold_source,
    )


def _top_features(
    row: int,
    Z: np.ndarray,
    features: list[str],
    X: np.ndarray,
    raw_X: np.ndarray,
    top_k: int,
) -> list[DeviatingFeature]:
    order = np.argsort(-np.abs(Z[row]), kind="stable")[:top_k]
    return [
        DeviatingFeature(
            feature=features[j],
            value=float(raw_X[row, j]) if np.isfinite(raw_X[row, j]) else None,
            robust_z=float(Z[row, j]),
        )
        for j in order
    ]


def score(
    baseline: AnomalyBaseline,
    frame: pd.DataFrame,
    *,
    config: AnomalyConfig | None = None,
) -> AnomalyResult:
    """Score a production window against a fitted baseline (never refits)."""
    score_config = config or AnomalyConfig()
    features = baseline.numeric_features
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise AnomalyError(f"production frame missing baseline numeric column(s): {missing}")
    if len(frame) == 0:
        raise AnomalyError("production frame has no rows")

    numeric = _to_numeric_matrix(frame, features)
    imputed = numeric.fillna({c: baseline.medians[c] for c in features})
    X = imputed.to_numpy(dtype=float)

    Z = _robust_z_matrix(X, features, baseline.medians, baseline.mads, baseline.stds)
    robust_scores = np.max(np.abs(Z), axis=1)
    iforest_scores = -baseline.model.decision_function(X)

    z_robust = (robust_scores - baseline.robust_score_center) / baseline.robust_score_scale
    z_iforest = (iforest_scores - baseline.iforest_score_center) / baseline.iforest_score_scale
    combined = 0.5 * (z_robust + z_iforest)
    threshold = score_config.threshold if config is not None else baseline.flag_threshold
    flag = combined > threshold

    rows: list[AnomalyRow] = []
    for i in range(X.shape[0]):
        rows.append(
            AnomalyRow(
                row_index=i,
                robust_score=float(robust_scores[i]),
                iforest_score=float(iforest_scores[i]),
                combined_score=float(combined[i]),
                flag=bool(flag[i]),
                top_deviating_features=_top_features(
                    i,
                    Z,
                    features,
                    X,
                    numeric.to_numpy(dtype=float),
                    score_config.top_k_features,
                ),
            )
        )

    flagged_count = int(flag.sum())
    anomaly_rate = flagged_count / len(rows) if rows else 0.0
    return AnomalyResult(
        rows=rows,
        threshold=threshold,
        flagged_count=flagged_count,
        anomaly_rate=round(anomaly_rate, 6),
        contamination=baseline.contamination,
        numeric_features=features,
        notes=[
            "No labels provided: accuracy/precision/recall are NOT claimed.",
            f"flag threshold: combined_score > {threshold} (standardised units).",
            f"threshold source: {baseline.threshold_source}.",
        ],
    )

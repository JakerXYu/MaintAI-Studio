"""Deterministic classification / regression task recommendation (rule engine).

The LLM is only allowed to narrate these deterministic results; the task type
and confidence always come from this rule engine.
"""

from __future__ import annotations

import pandas as pd

from maintai.tasks.schemas import Alternative, TaskRecommendation

_MAX_CLASSES = 30


def _clamp(value: float) -> float:
    return round(max(0.05, min(0.95, value)), 2)


def _binary_confidence(values: pd.Series) -> float:
    counts = values.value_counts()
    minor_rate = float(counts.min() / counts.sum())
    if minor_rate < 0.005:
        base = 0.55
    elif minor_rate < 0.05:
        base = 0.70
    elif minor_rate < 0.40:
        base = 0.85
    else:
        base = 0.75
    return _clamp(base)


def _multiclass_confidence(values: pd.Series) -> float:
    counts = values.value_counts()
    n_classes = int(len(counts))
    min_rate = float(counts.min() / counts.sum())
    base = 0.75 - 0.05 * max(0, n_classes - 2)
    if min_rate < 0.02:
        base -= 0.10
    return _clamp(base)


def _regression_confidence(values: pd.Series) -> float:
    vals = values.astype(float)
    std = float(vals.std())
    if std == 0.0:
        return _clamp(0.30)
    mean = float(vals.mean())
    cv = std / abs(mean) if mean != 0.0 else float("inf")
    base = 0.75
    if cv < 0.05:
        base -= 0.20
    if cv > 3.0:
        base -= 0.10
    return _clamp(base)


def _build_evidence(task: str, target_col: str, values: pd.Series, nunique: int) -> list[str]:
    if task == "binary_classification":
        counts = values.value_counts()
        minor = counts.index[-1]
        minor_rate = float(counts.min() / counts.sum())
        return [
            f"target {target_col!r} has {nunique} discrete classes (binary)",
            f"minority class rate is {minor_rate:.1%} (class {minor!r})",
        ]
    if task == "multiclass_classification":
        counts = values.value_counts()
        return [
            f"target {target_col!r} has {nunique} discrete classes",
            f"smallest class rate is {float(counts.min() / counts.sum()):.1%}",
        ]
    std = float(values.astype(float).std())
    return [
        f"target {target_col!r} is continuous numeric "
        f"({nunique} unique values, std={std:.4g})",
    ]


def _build_alternatives(task: str, values: pd.Series, nunique: int) -> list[Alternative]:
    if task == "regression" and nunique <= 100:
        return [
            Alternative(
                task="multiclass_classification",
                confidence=_multiclass_confidence(values),
                reason=f"target has only {nunique} unique numeric values and could be "
                "ordinal labels",
            )
        ]
    return []


def recommend_task(
    frame: pd.DataFrame,
    target_col: str | None = None,
) -> TaskRecommendation:
    """Recommend a supervised task deterministically from the target column."""
    if target_col is None or target_col not in frame.columns:
        return TaskRecommendation(
            recommended_task="insufficient_target",
            trainable=False,
            confidence=0.0,
            evidence=[
                "no target column was provided; P0 does not recommend unsupervised "
                "anomaly detection"
            ],
        )

    values = frame[target_col].dropna()
    if len(values) < 2:
        return TaskRecommendation(
            recommended_task="insufficient_target",
            trainable=False,
            confidence=0.0,
            target_column=target_col,
            evidence=[f"target {target_col!r} has fewer than 2 non-null values"],
        )

    nunique = int(values.nunique())
    if nunique <= 1:
        return TaskRecommendation(
            recommended_task="single_class",
            trainable=False,
            confidence=0.0,
            target_column=target_col,
            evidence=[
                f"target {target_col!r} has a single class, so no supervised task is "
                "trainable"
            ],
        )

    is_numeric = pd.api.types.is_numeric_dtype(frame[target_col].dtype)
    is_float = pd.api.types.is_float_dtype(frame[target_col].dtype)

    if nunique <= 2:
        task = "binary_classification"
        confidence = _binary_confidence(values)
    elif is_float and nunique <= _MAX_CLASSES:
        task = "multiclass_classification"
        confidence = _multiclass_confidence(values)
    elif is_float:
        task = "regression"
        confidence = _regression_confidence(values)
    elif is_numeric and nunique <= _MAX_CLASSES:
        task = "multiclass_classification"
        confidence = _multiclass_confidence(values)
    elif is_numeric:
        task = "regression"
        confidence = _regression_confidence(values)
    else:
        task = "multiclass_classification"
        confidence = _multiclass_confidence(values)

    return TaskRecommendation(
        recommended_task=task,
        trainable=True,
        confidence=confidence,
        target_column=target_col,
        evidence=_build_evidence(task, target_col, values, nunique),
        alternatives=_build_alternatives(task, values, nunique),
    )

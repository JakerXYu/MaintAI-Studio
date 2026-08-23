"""Deterministic model explanations (SHAP-first with permutation/perturbation fallback).

SHAP is preferred: ``TreeExplainer`` for tree estimators, ``LinearExplainer`` for
linear estimators. Values are always computed on the **preprocessed** feature
matrix (the pipeline's ``preprocess`` output) with the **traceable** output
feature names from :func:`maintai.ml.preprocess.get_feature_names`.

If SHAP is missing, incompatible, or returns an unexpected shape, the global
explanation falls back to permutation importance and the local explanation falls
back to single-feature perturbation (each feature replaced by its background
median/mode). Fallback methods are labelled as such — SHAP values are never
fabricated.

Every output carries the fixed disclaimer
``Model-based explanation, not a verified physical root cause.``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from maintai.ml import preprocess
from maintai.ml.schemas import CLASSIFICATION_TASKS

try:
    import shap

    _SHAP_AVAILABLE = True
except Exception:  # noqa: BLE001 - defensive import; fallback still works without shap
    shap = None  # type: ignore[assignment]
    _SHAP_AVAILABLE = False

DISCLAIMER = "Model-based explanation, not a verified physical root cause."

SHAP_METHOD_TREE = "shap_tree"
SHAP_METHOD_LINEAR = "shap_linear"
PERMUTATION_METHOD = "permutation_importance"
PERTURBATION_METHOD = "perturbation"


class ExplanationError(Exception):
    """Raised when an explanation cannot be produced from the given inputs."""


class FeatureImpact(BaseModel):
    """A single feature's value and its contribution to the model output."""

    feature: str
    value: float | str | bool | int | None
    impact: float


class GlobalExplanation(BaseModel):
    """Model-level explanation: most influential features and the method used."""

    top_features: list[FeatureImpact] = Field(default_factory=list)
    method: str
    disclaimer: str = DISCLAIMER


class LocalExplanation(BaseModel):
    """Sample-level explanation: prediction plus driving / suppressing features."""

    prediction: str | int | float | bool | None
    positive_probability: float | None = None
    top_positive: list[FeatureImpact] = Field(default_factory=list)
    top_negative: list[FeatureImpact] = Field(default_factory=list)
    method: str
    disclaimer: str = DISCLAIMER


def _json_scalar(value):
    """Convert numpy scalars to JSON-safe Python builtins."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_scalar(value.item()) if value.size == 1 else None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return value
    return value


def _final_estimator(pipeline):
    """Return the trained estimator from a ``Pipeline`` (or the object itself)."""
    if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
        return pipeline["model"]
    return pipeline


def _preprocess(pipeline, X) -> np.ndarray:
    """Apply the pipeline's preprocessing step to a raw feature frame."""
    if hasattr(pipeline, "named_steps") and "preprocess" in pipeline.named_steps:
        transformed = pipeline["preprocess"].transform(X)
        if isinstance(transformed, pd.DataFrame):
            return transformed.to_numpy(dtype=float)
        return np.asarray(transformed, dtype=float)
    return np.asarray(X, dtype=float)


def _preprocess_single(pipeline, x) -> np.ndarray:
    """Preprocess exactly one raw sample into a ``(1, n_features)`` matrix."""
    if hasattr(pipeline, "named_steps") and "preprocess" in pipeline.named_steps:
        if isinstance(x, pd.Series):
            frame = x.to_frame().T
        elif isinstance(x, pd.DataFrame):
            if x.shape[0] != 1:
                raise ExplanationError("local explanation requires exactly one sample")
            frame = x
        else:
            raise ExplanationError("local explanation requires a DataFrame or Series sample")
        transformed = pipeline["preprocess"].transform(frame)
        if isinstance(transformed, pd.DataFrame):
            return transformed.to_numpy(dtype=float)
        return np.asarray(transformed, dtype=float)
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] != 1:
        raise ExplanationError("local explanation requires exactly one sample")
    return arr.astype(float)


def _resolve_feature_names(pipeline, feature_names) -> list[str]:
    """Resolve preprocessed feature names from the argument or the pipeline."""
    if feature_names is not None:
        names = [str(name) for name in feature_names]
        if not names:
            raise ExplanationError("feature_names must not be empty")
        return names
    if hasattr(pipeline, "named_steps") and "preprocess" in pipeline.named_steps:
        return preprocess.get_feature_names(pipeline["preprocess"])
    raise ExplanationError("feature_names is required when the pipeline has no preprocess step")


def _validate_width(matrix: np.ndarray, names: list[str]) -> None:
    if matrix.ndim != 2:
        raise ExplanationError("feature matrix must be 2-dimensional")
    if matrix.shape[1] != len(names):
        raise ExplanationError(
            f"feature matrix has {matrix.shape[1]} columns but {len(names)} feature names"
        )


def _sample_indices(n: int, size: int | None, rng: np.random.Generator) -> np.ndarray:
    if size is None or size <= 0 or n <= size:
        return np.arange(n)
    return rng.choice(n, size=size, replace=False)


def _estimator_kind(model) -> str:
    """Classify an estimator as ``tree`` / ``linear`` / ``unknown`` for SHAP."""
    module = type(model).__module__ or ""
    if module.startswith("xgboost"):
        return "tree"
    if module.startswith("sklearn.linear_model"):
        return "linear"
    if module.startswith("sklearn.tree") or module.startswith("sklearn.ensemble"):
        return "tree"
    if hasattr(model, "coef_"):
        return "linear"
    if hasattr(model, "estimators_") or hasattr(model, "tree_") or hasattr(model, "get_booster"):
        return "tree"
    return "unknown"


def _model_classes(model) -> list:
    classes = getattr(model, "classes_", None)
    return list(classes) if classes is not None else []


def _resolve_positive_index(model, labels, positive_label, task) -> int | None:
    """Return the class index whose SHAP output represents the class of interest."""
    classes = _model_classes(model)
    decoded = list(labels) if labels else classes
    if labels is not None and len(decoded) != len(classes):
        raise ExplanationError("labels length does not match model classes")
    if task == "binary_classification":
        if positive_label is not None and positive_label in decoded:
            return decoded.index(positive_label)
        return 1 if len(decoded) >= 2 else 0
    # Multiclass has no single positive class; callers select the predicted class.
    return None


def _compute_shap(model, Xt: np.ndarray, Xt_background: np.ndarray | None, kind: str):
    """Compute raw SHAP values with the appropriate explainer."""
    if not _SHAP_AVAILABLE:
        raise ExplanationError("shap is not available in this environment")
    if kind == "tree":
        explainer = shap.TreeExplainer(model)
    elif kind == "linear":
        if Xt_background is None:
            raise ExplanationError("LinearExplainer requires a background sample")
        explainer = shap.LinearExplainer(model, Xt_background)
    else:
        raise ExplanationError(f"no SHAP explainer for estimator kind {kind!r}")
    return explainer(Xt).values


def _select_shap_values(raw, class_index: int | None) -> np.ndarray:
    """Select the class-of-interest SHAP values from list / 2D / 3D outputs."""
    if raw is None:
        raise ExplanationError("SHAP produced no values")
    if isinstance(raw, list):
        if not raw:
            raise ExplanationError("SHAP produced an empty list of values")
        chosen = (
            raw[class_index]
            if (class_index is not None and class_index < len(raw))
            else raw[0]
        )
        return np.asarray(chosen, dtype=float)
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 3:
        if class_index is None:
            return np.mean(np.abs(arr), axis=2)
        idx = class_index if (class_index is not None and class_index < arr.shape[2]) else 0
        return arr[:, :, idx]
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    raise ExplanationError(f"unexpected SHAP value shape {arr.shape}")


def _validate_values(values: np.ndarray, expected_shape: tuple[int, ...]) -> None:
    if values.shape != expected_shape:
        raise ExplanationError(
            f"SHAP value shape {values.shape} does not match expected {expected_shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ExplanationError("SHAP produced non-finite values")


def _score(model, Xt: np.ndarray, y: np.ndarray, task: str) -> float:
    """Deterministic score for permutation importance (higher is better)."""
    pred = np.asarray(model.predict(Xt)).ravel()
    y = np.asarray(y).ravel()
    if task in CLASSIFICATION_TASKS:
        return float(np.mean(pred == y))
    return -float(np.mean((pred.astype(float) - y.astype(float)) ** 2))


def _permutation_importance(
    model, Xt: np.ndarray, y: np.ndarray, task: str, rng: np.random.Generator, n_repeats: int = 5
) -> np.ndarray:
    baseline = _score(model, Xt, y, task)
    n, p = Xt.shape
    importances = np.zeros(p)
    for _ in range(n_repeats):
        perm = rng.permutation(n)
        for j in range(p):
            Xp = Xt.copy()
            Xp[:, j] = Xt[perm, j]
            importances[j] += baseline - _score(model, Xp, y, task)
    return importances / n_repeats


def _positive_probability(
    model, xt: np.ndarray, class_index: int | None, task: str
) -> float | None:
    """Return the positive-class (or predicted-class) probability for one sample."""
    if task not in CLASSIFICATION_TASKS or not hasattr(model, "predict_proba"):
        return None
    try:
        proba = np.asarray(model.predict_proba(xt), dtype=float)
    except Exception:  # noqa: BLE001 - some estimators raise on proba
        return None
    if proba.ndim == 1:
        proba = proba.reshape(1, -1)
    if proba.ndim != 2 or proba.shape[1] == 0:
        return None
    n_classes = proba.shape[1]
    idx = class_index if (class_index is not None and 0 <= class_index < n_classes) else int(
        np.argmax(proba[0])
    )
    return float(proba[0, idx])


def _score_value(model, xt: np.ndarray, task: str, class_index: int | None) -> float:
    """Scalar model output used as the perturbation baseline."""
    proba = _positive_probability(model, xt, class_index, task)
    if proba is not None:
        return proba
    return float(np.asarray(model.predict(xt)).ravel()[0])


def _background_statistics(X_bg: np.ndarray) -> np.ndarray:
    """Return median (continuous) / mode (binary one-hot/boolean) per column."""
    n_features = X_bg.shape[1]
    stats = np.zeros(n_features)
    for j in range(n_features):
        column = X_bg[:, j]
        unique = np.unique(column)
        if len(unique) <= 2 and np.all(np.isin(unique, [0.0, 1.0])):
            stats[j] = 1.0 if float(np.mean(column)) >= 0.5 else 0.0
        else:
            stats[j] = float(np.median(column))
    return stats


def _perturbation_attribution(
    model, xt: np.ndarray, X_bg: np.ndarray, task: str, class_index: int | None
) -> np.ndarray:
    """Single-feature perturbation delta against background median/mode."""
    baseline = _score_value(model, xt, task, class_index)
    stats = _background_statistics(X_bg)
    impacts = np.zeros(xt.shape[1])
    for j in range(xt.shape[1]):
        xp = xt.copy()
        xp[0, j] = stats[j]
        impacts[j] = baseline - _score_value(model, xp, task, class_index)
    return impacts


def _decode_prediction(model, xt: np.ndarray, labels, task: str):
    """Decode a numeric prediction back into the original label space."""
    pred = np.asarray(model.predict(xt)).ravel()
    encoded = int(pred[0])
    if task in CLASSIFICATION_TASKS:
        classes = list(labels) if labels else _model_classes(model)
        if classes and 0 <= encoded < len(classes):
            return _json_scalar(classes[encoded])
        return encoded
    return float(pred[0])


def _predicted_class_index(model, xt: np.ndarray) -> int:
    return int(np.asarray(model.predict(xt)).ravel()[0])


def _encode_fallback_labels(y, labels, model) -> np.ndarray:
    values = np.asarray(y).ravel()
    model_classes = _model_classes(model)
    if set(values) <= set(model_classes):
        return values
    if labels is not None and set(values) <= set(labels):
        mapping = {label: index for index, label in enumerate(labels)}
        return np.asarray([mapping[value] for value in values])
    raise ExplanationError("fallback labels do not match model classes or decoded labels")


def _build_top_features(
    names: list[str], value_vec: np.ndarray, importance: np.ndarray, top_k: int
) -> list[FeatureImpact]:
    importance = np.asarray(importance, dtype=float)
    order = np.argsort(-np.abs(importance))
    k = min(max(int(top_k), 1), len(names))
    return [
        FeatureImpact(
            feature=names[j],
            value=_json_scalar(float(value_vec[j])),
            impact=float(importance[j]),
        )
        for j in order[:k]
    ]


def _build_local_features(
    names: list[str], row: np.ndarray, impacts: np.ndarray, top_k: int
) -> tuple[list[FeatureImpact], list[FeatureImpact]]:
    impacts = np.asarray(impacts, dtype=float)
    k = min(max(int(top_k), 1), len(names))
    positive = [
        FeatureImpact(feature=names[j], value=_json_scalar(float(row[j])), impact=float(impacts[j]))
        for j in np.argsort(-impacts)[:k]
        if impacts[j] > 0
    ]
    negative = [
        FeatureImpact(feature=names[j], value=_json_scalar(float(row[j])), impact=float(impacts[j]))
        for j in np.argsort(impacts)[:k]
        if impacts[j] < 0
    ]
    return positive, negative


def global_explanation(
    pipeline,
    X,
    *,
    y=None,
    feature_names=None,
    sample_size: int = 200,
    background_size: int = 100,
    seed: int = 42,
    task: str = "binary_classification",
    positive_label=None,
    labels=None,
    top_k: int = 10,
) -> GlobalExplanation:
    """Produce a model-level explanation (SHAP-first, permutation fallback)."""
    model = _final_estimator(pipeline)
    Xt = _preprocess(pipeline, X)
    names = _resolve_feature_names(pipeline, feature_names)
    _validate_width(Xt, names)
    if Xt.shape[0] == 0:
        raise ExplanationError("cannot explain an empty feature matrix")

    rng = np.random.default_rng(seed)
    fg_idx = _sample_indices(Xt.shape[0], sample_size, rng)
    bg_idx = _sample_indices(Xt.shape[0], background_size, rng)
    X_fg = Xt[fg_idx]
    X_bg = Xt[bg_idx]
    kind = _estimator_kind(model)
    class_index = _resolve_positive_index(model, labels, positive_label, task)

    importance: np.ndarray | None = None
    method: str | None = None
    if kind in ("tree", "linear"):
        try:
            raw = _compute_shap(model, X_fg, X_bg, kind)
            values = _select_shap_values(raw, class_index)
            _validate_values(values, X_fg.shape)
            importance = np.mean(np.abs(values), axis=0)
            method = SHAP_METHOD_TREE if kind == "tree" else SHAP_METHOD_LINEAR
        except Exception:  # noqa: BLE001 - any SHAP failure triggers the fallback
            importance = None

    if importance is None:
        if y is None:
            raise ExplanationError(
                "SHAP explanation failed and no y was provided for the "
                "permutation-importance fallback"
            )
        y_arr = np.asarray(y).ravel()
        if y_arr.shape[0] != Xt.shape[0]:
            raise ExplanationError("y length does not match the feature matrix")
        if task in CLASSIFICATION_TASKS:
            y_arr = _encode_fallback_labels(y_arr, labels, model)
        importance = _permutation_importance(model, X_fg, y_arr[fg_idx], task, rng)
        method = PERMUTATION_METHOD

    value_vec = np.mean(X_fg, axis=0)
    return GlobalExplanation(
        method=method,
        top_features=_build_top_features(names, value_vec, importance, top_k),
    )


def local_explanation(
    pipeline,
    x,
    *,
    background=None,
    feature_names=None,
    background_size: int = 100,
    seed: int = 42,
    task: str = "binary_classification",
    positive_label=None,
    labels=None,
    top_k: int = 10,
) -> LocalExplanation:
    """Produce a sample-level explanation (SHAP-first, perturbation fallback)."""
    model = _final_estimator(pipeline)
    xt = _preprocess_single(pipeline, x)
    names = _resolve_feature_names(pipeline, feature_names)
    _validate_width(xt, names)

    rng = np.random.default_rng(seed)
    if background is not None:
        Xb = _preprocess(pipeline, background)
        _validate_width(Xb, names)
        X_bg: np.ndarray | None = Xb[_sample_indices(Xb.shape[0], background_size, rng)]
    else:
        X_bg = None

    kind = _estimator_kind(model)
    class_index = _resolve_positive_index(model, labels, positive_label, task)
    if task == "multiclass_classification":
        class_index = _predicted_class_index(model, xt)
    prediction = _decode_prediction(model, xt, labels, task)
    probability = _positive_probability(model, xt, class_index, task)

    impacts: np.ndarray | None = None
    method: str | None = None
    if kind in ("tree", "linear") and (kind == "tree" or X_bg is not None):
        try:
            raw = _compute_shap(model, xt, X_bg, kind)
            values = _select_shap_values(raw, class_index)
            arr = np.asarray(values, dtype=float).reshape(1, -1)
            _validate_values(arr, (1, len(names)))
            impacts = arr[0]
            method = SHAP_METHOD_TREE if kind == "tree" else SHAP_METHOD_LINEAR
        except Exception:  # noqa: BLE001 - any SHAP failure triggers the fallback
            impacts = None

    if impacts is None:
        if X_bg is None:
            raise ExplanationError(
                "SHAP explanation failed and no background was provided for the "
                "perturbation fallback"
            )
        impacts = _perturbation_attribution(model, xt, X_bg, task, class_index)
        method = PERTURBATION_METHOD

    top_positive, top_negative = _build_local_features(names, xt[0], impacts, top_k)
    return LocalExplanation(
        prediction=prediction,
        positive_probability=probability,
        top_positive=top_positive,
        top_negative=top_negative,
        method=method,
    )

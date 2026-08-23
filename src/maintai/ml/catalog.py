"""Deterministic model catalog with fixed small hyperparameters.

Classification: LogisticRegression, RandomForestClassifier, XGBoostClassifier.
Regression: Ridge, RandomForestRegressor, XGBoostRegressor.

xgboost is a declared P0 dependency; if it is missing at import time the catalog
still loads and reports an explicit ``unavailable`` reason instead of crashing.
LightGBM is intentionally not offered.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception as exc:  # noqa: BLE001
    XGBClassifier = None  # type: ignore[assignment]
    XGBRegressor = None  # type: ignore[assignment]
    _XGB_AVAILABLE = False
    _XGB_IMPORT_ERROR = str(exc)
else:
    _XGB_AVAILABLE = True
    _XGB_IMPORT_ERROR = None

CLASSIFICATION_MODELS: tuple[str, ...] = ("logistic_regression", "random_forest", "xgboost")
REGRESSION_MODELS: tuple[str, ...] = ("ridge", "random_forest", "xgboost")

_CLASSIFICATION_TASKS: tuple[str, ...] = ("binary_classification", "multiclass_classification")

_MODEL_COMPLEXITY: dict[str, int] = {
    "logistic_regression": 0,
    "ridge": 0,
    "random_forest": 1,
    "xgboost": 2,
}


class ModelUnavailableError(Exception):
    """Raised when a requested model cannot be constructed (e.g. missing xgboost)."""


def model_names_for_task(task: str) -> tuple[str, ...]:
    """Return the catalog model names for a supervised task."""
    if task in _CLASSIFICATION_TASKS:
        return CLASSIFICATION_MODELS
    if task == "regression":
        return REGRESSION_MODELS
    raise ModelUnavailableError(f"unsupported task {task!r}")


def is_model_available(name: str) -> bool:
    """Return whether a model can be constructed in this environment."""
    if name != "xgboost":
        return name in CLASSIFICATION_MODELS or name in REGRESSION_MODELS
    return _XGB_AVAILABLE


def unavailable_reason(name: str) -> str | None:
    """Return a human-readable reason when a model is unavailable, else None."""
    if name == "xgboost" and not _XGB_AVAILABLE:
        return (
            "xgboost is unavailable: xgboost could not be imported "
            f"({_XGB_IMPORT_ERROR}); P0 declares xgboost as a dependency"
        )
    return None


def model_complexity(name: str) -> int:
    """Return a deterministic complexity rank (lower = simpler, preferred)."""
    return _MODEL_COMPLEXITY.get(name, 0)


def build_model(name: str, task: str, *, seed: int = 42, n_jobs: int = 1):
    """Build an unfitted estimator with fixed small hyperparameters."""
    if task not in _CLASSIFICATION_TASKS and task != "regression":
        raise ModelUnavailableError(f"unsupported task {task!r}")

    if name == "xgboost":
        if not _XGB_AVAILABLE:
            raise ModelUnavailableError(unavailable_reason(name))
        if task in _CLASSIFICATION_TASKS:
            params: dict = {
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
                "random_state": seed,
                "n_jobs": n_jobs,
            }
            if task == "multiclass_classification":
                params["objective"] = "multi:softprob"
                params["eval_metric"] = "mlogloss"
            else:
                params["objective"] = "binary:logistic"
                params["eval_metric"] = "logloss"
            return XGBClassifier(**params)
        return XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=seed,
            n_jobs=n_jobs,
            objective="reg:squarederror",
        )

    if task in _CLASSIFICATION_TASKS:
        if name == "logistic_regression":
            return LogisticRegression(max_iter=1000, random_state=seed)
        if name == "random_forest":
            return RandomForestClassifier(
                n_estimators=100, max_depth=8, random_state=seed, n_jobs=n_jobs
            )

    if name == "ridge":
        return Ridge(alpha=1.0, random_state=seed)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=100, max_depth=8, random_state=seed, n_jobs=n_jobs
        )

    raise ModelUnavailableError(f"unknown model {name!r} for task {task!r}")

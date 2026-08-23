"""Deterministic preprocessing: imputation, scaling, and one-hot encoding.

Column roles are resolved deterministically and the transformer is fit only on
the train fold, so test data can never leak into imputer/scaler statistics:

- numeric      -> median imputation + optional standard scaling
- categorical  -> most-frequent imputation + one-hot (unknown categories ignored)
- boolean      -> deterministic {0,1} mapping + mode imputation
- identifier / asset / timestamp / explicitly excluded -> never enter features
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_BOOLEAN_TRUE: frozenset[str] = frozenset({"true", "t", "yes", "y", "1", "on", "1.0"})
_BOOLEAN_FALSE: frozenset[str] = frozenset({"false", "f", "no", "n", "0", "off", "0.0"})
_BOOLEAN_VALUES: frozenset[str] = _BOOLEAN_TRUE | _BOOLEAN_FALSE
_IDENTIFIER_UNIQUE_RATIO = 0.9


class PreprocessError(Exception):
    """Raised when a usable feature set cannot be determined."""


def _as_frame(X: object) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X)


def _to_boolean_codes(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("float64")
    if pd.api.types.is_numeric_dtype(series.dtype):
        return series.astype("float64")
    mapped = series.astype(str).str.lower().str.strip()
    return mapped.map(
        lambda value: (
            1.0 if value in _BOOLEAN_TRUE else (0.0 if value in _BOOLEAN_FALSE else np.nan)
        )
    ).astype("float64")


class BooleanEncoder(BaseEstimator, TransformerMixin):
    """Map boolean-like values to {0, 1} and impute missing with the train mode."""

    def fit(self, X, y=None):
        frame = _as_frame(X)
        self.columns_ = list(frame.columns)
        self.mode_: dict[str, float] = {}
        for column in self.columns_:
            codes = _to_boolean_codes(frame[column])
            mode = codes.mode()
            self.mode_[column] = float(mode.iloc[0]) if not mode.empty else 0.0
        return self

    def transform(self, X):
        frame = _as_frame(X)
        out = pd.DataFrame(index=frame.index)
        for column in self.columns_:
            out[column] = _to_boolean_codes(frame[column]).fillna(self.mode_[column]).astype(
                "float64"
            )
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


def is_identifier_like(series: pd.Series) -> bool:
    """Treat datetimes and near-unique object columns as identifiers (not features)."""
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return True
    if pd.api.types.is_numeric_dtype(series.dtype):
        return False
    non_null = int(series.notna().sum())
    if non_null == 0:
        return False
    unique = int(series.nunique(dropna=True))
    return (unique / non_null) >= _IDENTIFIER_UNIQUE_RATIO


def classify_column(series: pd.Series) -> str:
    """Return ``numeric`` / ``boolean`` / ``categorical`` / ``drop`` for a column."""
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "drop"
    if pd.api.types.is_bool_dtype(series.dtype):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series.dtype):
        values = series.dropna().unique()
        if len(values) <= 2 and set(values) <= {0, 1}:
            return "boolean"
        return "numeric"
    values = series.dropna().astype(str).str.lower().str.strip().unique()
    if len(values) <= 2 and all(value in _BOOLEAN_VALUES for value in values):
        return "boolean"
    return "categorical"


def resolve_features(
    frame: pd.DataFrame,
    target: str,
    features: list[str],
    excluded: list[str],
) -> list[str]:
    """Resolve the usable feature columns (excludes target/id/timestamp/excluded)."""
    excluded_set = set(excluded)
    excluded_set.add(target)
    if features:
        missing = [column for column in features if column not in frame.columns]
        if missing:
            raise PreprocessError(f"feature column(s) not found in the dataset: {missing}")
        candidate = list(features)
    else:
        candidate = [column for column in frame.columns if column not in excluded_set]

    resolved: list[str] = []
    seen: set[str] = set()
    for column in candidate:
        if column in excluded_set or column in seen:
            continue
        if is_identifier_like(frame[column]):
            continue
        resolved.append(column)
        seen.add(column)

    if not resolved:
        raise PreprocessError("no usable feature columns remain after exclusions")
    return resolved


def build_preprocessor(
    frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build a ColumnTransformer over numeric/categorical/boolean feature columns."""
    numeric: list[str] = []
    categorical: list[str] = []
    boolean: list[str] = []
    for column in feature_cols:
        kind = classify_column(frame[column])
        if kind == "numeric":
            numeric.append(column)
        elif kind == "categorical":
            categorical.append(column)
        elif kind == "boolean":
            boolean.append(column)

    transformers: list[tuple[str, object, list[str]]] = []
    if numeric:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps), numeric))
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    if boolean:
        transformers.append(("bool", BooleanEncoder(), boolean))

    if not transformers:
        raise PreprocessError("no numeric, categorical, or boolean columns to preprocess")

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return the tracked output feature names of a fitted preprocessor."""
    return [str(name) for name in preprocessor.get_feature_names_out()]

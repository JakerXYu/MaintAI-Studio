"""Deterministic schema inference.

Physical types are derived from pandas dtypes; semantic types are assigned by
deterministic rules (name hints + cardinality), never by an LLM. The result
feeds candidate target / asset-id / timestamp selection.
"""

from __future__ import annotations

import re
import warnings

import pandas as pd

from maintai.data.schemas import ColumnSchema, SchemaInference

_IDENTIFIER_HINTS: tuple[str, ...] = (
    "id",
    "uid",
    "udi",
    "uuid",
    "serial",
    "asset",
    "machine",
    "equipment",
    "device",
    "unit",
    "part",
    "component",
    "tag",
    "code",
    "plate",
    "vin",
    "index",
    "no",
    "number",
)
_SENSOR_HINTS: tuple[str, ...] = (
    "temp",
    "temperature",
    "pressure",
    "vibration",
    "rpm",
    "speed",
    "torque",
    "flow",
    "power",
    "current",
    "voltage",
    "force",
    "load",
    "heat",
    "humidity",
    "level",
    "rotational",
    "air",
    "process",
    "celcius",
    "fahrenheit",
)
_TARGET_HINTS: tuple[str, ...] = (
    "failure",
    "fault",
    "fail",
    "label",
    "target",
    "class",
    "status",
    "condition",
    "rul",
    "ttf",
    "health",
    "remaining",
    "defect",
    "anomaly",
    "error",
    "alarm",
    "maintenance",
    "breakdown",
    "down",
)
_TIMESTAMP_HINTS: tuple[str, ...] = (
    "time",
    "date",
    "datetime",
    "timestamp",
    "created",
    "recorded",
    "logged",
    "day",
    "hour",
    "minute",
)

_BOOLEAN_VALUES = frozenset(
    {"true", "false", "yes", "no", "y", "n", "1", "0", "t", "f", "on", "off"}
)

_MAX_CATEGORICAL_CARDINALITY = 30
_IDENTIFIER_UNIQUE_RATIO = 0.9


def normalize_name(name: str) -> str:
    """Lowercase a name and strip every non-alphanumeric character."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def has_name_hint(name: str, hints: tuple[str, ...]) -> bool:
    """Return True when any hint appears as a substring of the normalized name."""
    norm = normalize_name(name)
    return any(hint in norm for hint in hints)


def coarse_dtype(series: pd.Series) -> str:
    """Map a pandas dtype to a coarse, portable type label."""
    if pd.api.types.is_bool_dtype(series.dtype):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    if pd.api.types.is_integer_dtype(series.dtype):
        return "int"
    if pd.api.types.is_float_dtype(series.dtype):
        return "float"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "category"
    return "string"


def _is_boolean_like(series: pd.Series, dtype: str) -> bool:
    if dtype == "bool":
        return True
    if dtype in ("int", "float"):
        values = series.dropna().unique()
        return len(values) <= 2 and set(values) <= {0, 1}
    if dtype not in ("string", "category"):
        return False
    values = series.dropna().astype(str).str.lower().str.strip().unique()
    if len(values) > 2:
        return False
    return all(value in _BOOLEAN_VALUES for value in values)


def _looks_like_datetime(series: pd.Series, dtype: str) -> bool:
    if dtype == "datetime":
        return True
    if dtype != "string":
        return False
    sample = series.dropna().head(200)
    if sample.empty:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(sample, errors="coerce")
        return bool(parsed.notna().all())
    except (TypeError, ValueError):
        return False


def _semantic_type(
    name: str,
    series: pd.Series,
    dtype: str,
    nunique: int,
    unique_ratio: float,
) -> str:
    if _looks_like_datetime(series, dtype):
        return "datetime"
    if _is_boolean_like(series, dtype):
        return "boolean"
    if dtype in ("int", "float"):
        if has_name_hint(name, _IDENTIFIER_HINTS) and unique_ratio >= _IDENTIFIER_UNIQUE_RATIO:
            return "identifier_like"
        if has_name_hint(name, _SENSOR_HINTS):
            return "sensor_numeric"
        if dtype == "float":
            return "sensor_numeric"
        if nunique <= _MAX_CATEGORICAL_CARDINALITY:
            return "categorical"
        return "numeric"
    if dtype in ("string", "category"):
        if unique_ratio >= _IDENTIFIER_UNIQUE_RATIO:
            return "identifier_like"
        return "categorical"
    return dtype


def _candidate_targets(per_col: dict[str, dict]) -> list[str]:
    candidates: list[tuple[str, int]] = []
    for name, info in per_col.items():
        semantic = info["semantic"]
        nunique = info["nunique"]
        if semantic in ("datetime", "identifier_like"):
            continue
        if nunique < 2 or nunique > 1000:
            continue
        hint = has_name_hint(name, _TARGET_HINTS)
        if semantic == "boolean":
            candidates.append((name, 3 if hint else 2))
        elif semantic in ("categorical", "sensor_numeric", "numeric") and (
            nunique <= _MAX_CATEGORICAL_CARDINALITY
        ):
            candidates.append((name, 3 if hint else 1))
        elif hint:
            candidates.append((name, 2))
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return [name for name, _ in candidates]


def _candidate_asset_ids(per_col: dict[str, dict]) -> list[str]:
    candidates: list[tuple[str, int]] = []
    for name, info in per_col.items():
        if info["semantic"] == "datetime":
            continue
        hint = has_name_hint(name, _IDENTIFIER_HINTS)
        if info["semantic"] == "identifier_like" and hint:
            candidates.append((name, 3))
        elif hint:
            candidates.append((name, 1))
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return [name for name, _ in candidates]


def _candidate_timestamps(per_col: dict[str, dict]) -> list[str]:
    candidates: list[tuple[str, int]] = []
    for name, info in per_col.items():
        semantic = info["semantic"]
        hint = has_name_hint(name, _TIMESTAMP_HINTS)
        if semantic == "datetime":
            candidates.append((name, 3 if hint else 2))
        elif hint and info["dtype"] in ("datetime", "string") and semantic != "identifier_like":
            candidates.append((name, 1))
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return [name for name, _ in candidates]


def infer_schema(frame: pd.DataFrame) -> SchemaInference:
    """Infer physical and semantic types plus candidate columns."""
    columns: list[ColumnSchema] = []
    per_col: dict[str, dict] = {}
    for name in frame.columns:
        series = frame[name]
        dtype = coarse_dtype(series)
        non_null = int(series.notna().sum())
        nunique = int(series.nunique(dropna=True))
        unique_ratio = nunique / non_null if non_null else 0.0
        missing_rate = float(series.isna().mean()) if len(series) else 0.0
        semantic = _semantic_type(name, series, dtype, nunique, unique_ratio)
        per_col[name] = {
            "dtype": dtype,
            "semantic": semantic,
            "nunique": nunique,
            "unique_ratio": unique_ratio,
        }
        columns.append(
            ColumnSchema(
                name=name,
                dtype=dtype,
                semantic_type=semantic,
                missing_rate=round(missing_rate, 6),
                nunique=nunique,
                unique_ratio=round(unique_ratio, 6),
            )
        )
    return SchemaInference(
        columns=columns,
        candidate_targets=_candidate_targets(per_col),
        candidate_asset_ids=_candidate_asset_ids(per_col),
        candidate_timestamps=_candidate_timestamps(per_col),
    )

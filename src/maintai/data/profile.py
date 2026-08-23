"""Deterministic dataset profiling (shape, types, missingness, statistics)."""

from __future__ import annotations

import pandas as pd

from maintai.data.schema import SchemaInference, coarse_dtype
from maintai.data.schemas import (
    CategoryFrequency,
    ColumnProfile,
    NumericStats,
    ProfileReport,
)


def _numeric_stats(series: pd.Series) -> NumericStats | None:
    values = series.dropna().astype(float)
    if values.empty:
        return NumericStats(count=0)
    if len(values) == 1:
        value = float(values.iloc[0])
        return NumericStats(
            count=1,
            min=value,
            max=value,
            mean=value,
            std=0.0,
            median=value,
            q1=value,
            q3=value,
            iqr=0.0,
        )
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    return NumericStats(
        count=int(len(values)),
        min=float(values.min()),
        max=float(values.max()),
        mean=float(values.mean()),
        std=float(values.std(ddof=0)),
        median=float(values.median()),
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
    )


def _top_categories(series: pd.Series, top_n: int) -> list[CategoryFrequency]:
    if pd.api.types.is_numeric_dtype(series.dtype) and series.nunique(dropna=True) > 20:
        return []
    counts = series.dropna().astype(str).value_counts().head(top_n)
    total = int(counts.sum())
    if total == 0:
        return []
    return [
        CategoryFrequency(value=str(value), count=int(count), ratio=float(count / total))
        for value, count in counts.items()
    ]


def profile(
    frame: pd.DataFrame,
    schema: SchemaInference | None = None,
    top_n: int = 10,
) -> ProfileReport:
    """Compute shape, type, missingness, duplicate, cardinality, and stat profiles."""
    row_count, column_count = frame.shape
    cell_count = row_count * column_count
    missing_cells = int(frame.isna().sum().sum())
    missing_rate = missing_cells / cell_count if cell_count else 0.0
    duplicate_rows = int(frame.duplicated().sum())
    duplicate_rate = duplicate_rows / row_count if row_count else 0.0

    type_distribution: dict[str, int] = {}
    for name in frame.columns:
        dtype = coarse_dtype(frame[name])
        type_distribution[dtype] = type_distribution.get(dtype, 0) + 1

    sem_map: dict[str, str] = {}
    if schema is not None:
        sem_map = {column.name: column.semantic_type for column in schema.columns}

    columns: list[ColumnProfile] = []
    for name in frame.columns:
        series = frame[name]
        missing_count = int(series.isna().sum())
        nunique = int(series.nunique(dropna=True))
        numeric = (
            _numeric_stats(series) if pd.api.types.is_numeric_dtype(series.dtype) else None
        )
        columns.append(
            ColumnProfile(
                name=name,
                dtype=coarse_dtype(series),
                semantic_type=sem_map.get(name),
                missing_count=missing_count,
                missing_rate=missing_count / len(series) if len(series) else 0.0,
                unique_count=nunique,
                cardinality=nunique,
                constant=nunique <= 1,
                numeric=numeric,
                top_categories=_top_categories(series, top_n),
            )
        )

    return ProfileReport(
        row_count=row_count,
        column_count=column_count,
        cell_count=cell_count,
        missing_cells=missing_cells,
        missing_rate=missing_rate,
        duplicate_rows=duplicate_rows,
        duplicate_row_rate=duplicate_rate,
        type_distribution=type_distribution,
        columns=columns,
    )

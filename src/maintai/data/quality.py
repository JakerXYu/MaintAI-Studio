"""Deterministic industrial data-quality assessment.

Produces an explainable ``MaintAI heuristic data health score`` (0-100) backed
by named findings and point penalties. Thresholds are demo defaults, not an
industry standard, and are surfaced explicitly in every finding.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from maintai.data.schema import SchemaInference
from maintai.data.schemas import Penalty, QualityFinding, QualityReport


class QualityConfig(BaseModel):
    """Demo-default thresholds for quality checks (mirrors configs/default.yaml)."""

    missing_warning_rate: float = 0.05
    missing_severe_rate: float = 0.20
    duplicate_row_warning_rate: float = 0.01
    duplicate_row_error_rate: float = 0.10
    near_constant_unique_ratio: float = 0.001
    flatline_min_length: int = 10
    flatline_min_ratio: float = 0.05
    outlier_mad_threshold: float = 3.5
    outlier_severe_rate: float = 0.10
    timestamp_gap_factor: float = 5.0
    sampling_cv_threshold: float = 1.0
    asset_imbalance_ratio: float = 10.0
    target_imbalance_ratio: float = 20.0
    min_trainable_samples: int = 100


def _sensor_columns(
    frame: pd.DataFrame,
    schema: SchemaInference | None,
    excluded: set[str],
) -> list[str]:
    sem_map = {c.name: c.semantic_type for c in schema.columns} if schema else {}
    result: list[str] = []
    for name in frame.columns:
        if name in excluded:
            continue
        if not pd.api.types.is_numeric_dtype(frame[name].dtype):
            continue
        if sem_map.get(name) == "identifier_like":
            continue
        result.append(name)
    return result


def _max_flatline_run(values: np.ndarray) -> int:
    """Longest run of consecutive identical (non-null) values."""
    if values.size == 0:
        return 0
    best = 1
    run = 1
    prev = values[0]
    for value in values[1:]:
        if pd.isna(value) or pd.isna(prev):
            run = 1
        elif value == prev:
            run += 1
            best = max(best, run)
        else:
            run = 1
        prev = value
    return best


def assess(
    frame: pd.DataFrame,
    schema: SchemaInference | None = None,
    timestamp_col: str | None = None,
    asset_id_col: str | None = None,
    target_col: str | None = None,
    config: QualityConfig | None = None,
) -> QualityReport:
    """Run all P0 quality checks and return an explainable health report."""
    config = config or QualityConfig()
    findings: list[QualityFinding] = []
    penalties: list[Penalty] = []
    checks: dict[str, Any] = {}

    def add(
        check: str,
        severity: str,
        message: str,
        *,
        column: str | None = None,
        value: Any = None,
        penalty: float = 0.0,
    ) -> None:
        findings.append(
            QualityFinding(
                check=check,
                severity=severity,
                column=column,
                message=message,
                value=value,
                penalty=penalty,
            )
        )
        if penalty > 0:
            penalties.append(
                Penalty(check=check, column=column, points=penalty, reason=message)
            )

    row_count, column_count = frame.shape
    cell_count = row_count * column_count

    # 1. Missingness.
    missing_cells = int(frame.isna().sum().sum())
    missing_rate = missing_cells / cell_count if cell_count else 0.0
    checks["missing_rate"] = round(missing_rate, 6)
    if missing_rate > config.missing_severe_rate:
        add(
            "missing",
            "error",
            f"overall missing cell rate {missing_rate:.1%} exceeds "
            f"{config.missing_severe_rate:.0%}",
            value=missing_rate,
            penalty=15.0,
        )
    elif missing_rate > config.missing_warning_rate:
        add(
            "missing",
            "warning",
            f"overall missing cell rate {missing_rate:.1%} exceeds "
            f"{config.missing_warning_rate:.0%}",
            value=missing_rate,
            penalty=5.0,
        )

    # 2. Duplicate rows.
    duplicate_rows = int(frame.duplicated().sum())
    duplicate_rate = duplicate_rows / row_count if row_count else 0.0
    checks["duplicate_row_rate"] = round(duplicate_rate, 6)
    if duplicate_rate > config.duplicate_row_error_rate:
        add(
            "duplicates",
            "error",
            f"{duplicate_rate:.1%} of rows are exact duplicates",
            value=duplicate_rate,
            penalty=15.0,
        )
    elif duplicate_rate > config.duplicate_row_warning_rate:
        add(
            "duplicates",
            "warning",
            f"{duplicate_rate:.1%} of rows are exact duplicates",
            value=duplicate_rate,
            penalty=5.0,
        )

    sensor_cols = _sensor_columns(
        frame,
        schema,
        {name for name in (timestamp_col, asset_id_col, target_col) if name},
    )

    # 3. Constant / near-constant sensors.
    for column in sensor_cols:
        series = frame[column]
        non_null = int(series.notna().sum())
        nunique = int(series.nunique(dropna=True))
        if nunique <= 1:
            add(
                "constant_sensor",
                "warning",
                f"sensor column {column!r} is constant",
                column=column,
                value=nunique,
                penalty=3.0,
            )
        elif non_null and nunique / non_null <= config.near_constant_unique_ratio:
            add(
                "near_constant_sensor",
                "warning",
                f"sensor column {column!r} is near-constant ({nunique} unique values)",
                column=column,
                value=nunique,
                penalty=1.0,
            )

    # 4. Timestamp-based checks.
    if timestamp_col and timestamp_col in frame.columns:
        ts_series = pd.to_datetime(frame[timestamp_col], errors="coerce", utc=True)
        ts_np = ts_series.to_numpy(dtype="datetime64[ns]")
        valid_mask = ~np.isnat(ts_np)
        valid_positions = np.flatnonzero(valid_mask)

        groups: list[np.ndarray] = []
        if asset_id_col and asset_id_col in frame.columns:
            assets = frame[asset_id_col].to_numpy()
            for asset in pd.unique(assets[valid_positions]):
                if pd.isna(asset):
                    continue
                positions = valid_positions[assets[valid_positions] == asset]
                groups.append(positions[np.argsort(ts_np[positions], kind="stable")])
        elif valid_positions.size:
            groups.append(valid_positions[np.argsort(ts_np[valid_positions], kind="stable")])

        for column in sensor_cols:
            values = frame[column].to_numpy(dtype=float)
            runs = [(_max_flatline_run(values[group]), len(group)) for group in groups]
            run, group_size = max(runs, default=(0, 0), key=lambda item: item[0])
            ratio = run / group_size if group_size else 0.0
            if run >= config.flatline_min_length and ratio >= config.flatline_min_ratio:
                add(
                    "flatline",
                    "warning",
                    f"sensor column {column!r} has a flatline run of {run} rows "
                    f"({ratio:.1%})",
                    column=column,
                    value=run,
                    penalty=3.0,
                )

        if asset_id_col and asset_id_col in frame.columns:
            duplicate_ts = int(
                frame.loc[valid_mask, [asset_id_col, timestamp_col]].duplicated().sum()
            )
        else:
            duplicate_ts = int(ts_series[valid_mask].duplicated().sum())
        checks["duplicate_timestamps"] = duplicate_ts
        if duplicate_ts > 0:
            add(
                "duplicate_timestamps",
                "warning",
                f"{duplicate_ts} duplicate timestamp values",
                value=duplicate_ts,
                penalty=2.0,
            )

        group_gaps = [
            (ts_np[group][1:] - ts_np[group][:-1])
            .astype("timedelta64[s]")
            .astype(np.float64)
            for group in groups
            if len(group) >= 2
        ]
        if group_gaps:
            gaps = np.concatenate(group_gaps)
            median_gap = float(np.median(gaps))
            checks["timestamp_median_gap_seconds"] = median_gap
            if median_gap > 0:
                gap_count = int(np.sum(gaps > config.timestamp_gap_factor * median_gap))
                checks["timestamp_gap_count"] = gap_count
                if gap_count > 0:
                    add(
                        "timestamp_gaps",
                        "warning",
                        f"{gap_count} timestamp gaps exceed "
                        f"{config.timestamp_gap_factor:.1f}x the median gap",
                        value=gap_count,
                        penalty=3.0,
                    )
                std_gap = float(np.std(gaps)) if gaps.size > 1 else 0.0
                cv = std_gap / median_gap
                checks["sampling_cv"] = round(cv, 4)
                if cv > config.sampling_cv_threshold:
                    add(
                        "sampling_irregularity",
                        "warning",
                        f"sampling interval coefficient of variation is {cv:.2f}",
                        value=cv,
                        penalty=2.0,
                    )

    # 5. Robust outlier rate.
    outlier_rates: dict[str, float] = {}
    for column in sensor_cols:
        values = frame[column].dropna().astype(float)
        if len(values) < 2:
            continue
        median = float(values.median())
        mad = float((values - median).abs().median())
        if mad == 0:
            continue
        robust_z = 0.6745 * (values - median) / mad
        rate = float((robust_z.abs() > config.outlier_mad_threshold).mean())
        outlier_rates[column] = round(rate, 6)
        if rate > config.outlier_severe_rate:
            add(
                "outlier",
                "warning",
                f"sensor column {column!r} has {rate:.1%} robust outliers",
                column=column,
                value=rate,
                penalty=5.0,
            )
    checks["outlier_rates"] = outlier_rates

    # 6. Asset record imbalance.
    if asset_id_col and asset_id_col in frame.columns:
        counts = frame[asset_id_col].value_counts(dropna=True)
        if len(counts) > 1:
            ratio = float(counts.iloc[0] / counts.median())
            checks["asset_record_ratio_max_to_median"] = round(ratio, 4)
            checks["asset_record_min"] = int(counts.min())
            checks["asset_record_max"] = int(counts.max())
            if ratio > config.asset_imbalance_ratio:
                add(
                    "asset_imbalance",
                    "warning",
                    f"asset record counts are imbalanced (max/median {ratio:.1f}x)",
                    value=ratio,
                    penalty=3.0,
                )

    # 7. Target imbalance.
    if target_col and target_col in frame.columns:
        counts = frame[target_col].dropna().value_counts()
        checks["target_classes"] = int(len(counts))
        if len(counts) > 1:
            ratio = float(counts.iloc[0] / counts.iloc[-1])
            checks["target_class_ratio"] = round(ratio, 4)
            if ratio > config.target_imbalance_ratio:
                add(
                    "target_imbalance",
                    "warning",
                    f"target classes are imbalanced ({ratio:.1f}:1 majority:minority)",
                    column=target_col,
                    value=ratio,
                    penalty=5.0,
                )
        elif len(counts) == 1:
            add(
                "target_imbalance",
                "error",
                "target has a single class",
                column=target_col,
                value=1,
                penalty=10.0,
            )

    # 8. Trainable sample count.
    if target_col and target_col in frame.columns:
        trainable = int(frame[target_col].notna().sum())
    else:
        trainable = row_count
    checks["trainable_sample_count"] = trainable
    if trainable < config.min_trainable_samples:
        add(
            "trainable_samples",
            "warning",
            f"only {trainable} trainable samples (minimum {config.min_trainable_samples})",
            value=trainable,
            penalty=10.0,
        )

    total = sum(penalty.points for penalty in penalties)
    score = round(max(0.0, min(100.0, 100.0 - total)), 2)
    return QualityReport(
        score=score,
        penalties=penalties,
        findings=findings,
        checks=checks,
        trainable_sample_count=trainable,
    )

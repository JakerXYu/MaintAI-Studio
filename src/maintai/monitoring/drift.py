"""Deterministic drift detection.

Numeric features: quantile-bin PSI (bin edges derived from the BASELINE
quantiles only — production data never defines the bins), a two-sample
Kolmogorov-Smirnov test (statistic + p-value), and standardised mean/std shifts.
Categorical features: PSI + total-variation distance, with unknown production
categories reported. Missingness drift is reported per feature.

Severity thresholds are configurable demo defaults and are explicitly labelled
as heuristics — NOT universal industry standards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator
from scipy import stats as _scipy_stats

from maintai.monitoring.contracts import (
    SEVERITY_ORDER,
    DriftReport,
    FeatureDrift,
    Severity,
)


class DriftError(Exception):
    """Raised when drift cannot be computed (no common columns / empty inputs)."""


class DriftConfig(BaseModel):
    """Demo-default drift thresholds (mirrors ``configs/default.yaml``)."""

    psi_medium: float = 0.10
    psi_high: float = 0.25
    ks_medium: float = 0.10
    ks_high: float = 0.20
    shift_medium: float = 0.50
    shift_high: float = 1.00
    missingness_medium: float = 0.05
    missingness_high: float = 0.15
    n_bins: int = 10
    epsilon: float = 1e-6

    @field_validator("n_bins")
    @classmethod
    def _positive_bins(cls, value: int) -> int:
        if value < 2:
            raise ValueError("n_bins must be >= 2")
        return value

    @field_validator("epsilon")
    @classmethod
    def _positive_epsilon(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("epsilon must be > 0")
        return value


def _severity(value: float | None, medium: float, high: float) -> Severity:
    if value is None:
        return "LOW"
    if value < medium:
        return "LOW"
    if value < high:
        return "MEDIUM"
    return "HIGH"


def _max_severity(severities: list[Severity]) -> Severity:
    return max(severities, key=lambda s: SEVERITY_ORDER[s], default="LOW")


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(
        series.dtype
    )


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _numeric_psi(
    baseline: np.ndarray,
    production: np.ndarray,
    n_bins: int,
    epsilon: float,
) -> float | None:
    """Quantile-bin PSI using baseline-derived bins only (constant-safe)."""
    base = _finite(baseline)
    prod = _finite(production)
    if base.size == 0 or prod.size == 0:
        return None
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    core_edges = np.unique(np.quantile(base, quantiles))
    if core_edges.size < 2:
        return None  # constant baseline: PSI undefined
    # Separate baseline-range overflow bins make extreme production values visible.
    edges = np.concatenate(
        (
            [-np.inf, np.nextafter(core_edges[0], -np.inf)],
            core_edges[1:-1],
            [np.nextafter(core_edges[-1], np.inf), np.inf],
        )
    )
    base_counts, _ = np.histogram(base, bins=edges)
    prod_counts, _ = np.histogram(prod, bins=edges)
    n = base_counts.size
    base_p = (base_counts + epsilon) / (base_counts.sum() + epsilon * n)
    prod_p = (prod_counts + epsilon) / (prod_counts.sum() + epsilon * n)
    return float(np.sum((prod_p - base_p) * np.log(prod_p / base_p)))


def _ks_test(baseline: np.ndarray, production: np.ndarray) -> tuple[float | None, float | None]:
    base = _finite(baseline)
    prod = _finite(production)
    if base.size == 0 or prod.size == 0:
        return None, None
    result = _scipy_stats.ks_2samp(base, prod)
    statistic = float(result.statistic)
    pvalue = float(result.pvalue) if np.isfinite(result.pvalue) else None
    return statistic, pvalue


def _standardized_shifts(
    baseline: np.ndarray,
    production: np.ndarray,
) -> tuple[float | None, float | None]:
    base = _finite(baseline)
    prod = _finite(production)
    if base.size == 0 or prod.size == 0:
        return None, None
    base_mean = float(np.mean(base))
    base_std = float(np.std(base))
    prod_mean = float(np.mean(prod))
    prod_std = float(np.std(prod))
    if base_std <= 0:
        return None, None
    return (prod_mean - base_mean) / base_std, (prod_std - base_std) / base_std


def _categorical_drift(
    baseline: pd.Series,
    production: pd.Series,
    epsilon: float,
) -> tuple[float | None, float | None, list[str]]:
    base = baseline.dropna().astype(str)
    prod = production.dropna().astype(str)
    base_counts = base.value_counts()
    prod_counts = prod.value_counts()
    categories = sorted(set(base_counts.index) | set(prod_counts.index))
    new_categories = sorted(set(prod_counts.index) - set(base_counts.index))
    base_total = int(base_counts.sum())
    prod_total = int(prod_counts.sum())
    if base_total == 0 or prod_total == 0:
        return None, None, new_categories

    n = len(categories)
    base_p = np.array(
        [(base_counts.get(c, 0) + epsilon) / (base_total + epsilon * n) for c in categories]
    )
    prod_p = np.array(
        [(prod_counts.get(c, 0) + epsilon) / (prod_total + epsilon * n) for c in categories]
    )
    psi = float(np.sum((prod_p - base_p) * np.log(prod_p / base_p)))

    base_raw = np.array([base_counts.get(c, 0) / base_total for c in categories])
    prod_raw = np.array([prod_counts.get(c, 0) / prod_total for c in categories])
    total_variation = float(0.5 * np.sum(np.abs(base_raw - prod_raw)))
    return psi, total_variation, new_categories


def _missing_shift(baseline: pd.Series, production: pd.Series) -> float:
    return float(pd.isna(production).mean() - pd.isna(baseline).mean())


def _round(value: float | None, ndigits: int = 6) -> float | None:
    return round(value, ndigits) if value is not None else None


def _numeric_feature_drift(
    feature: str,
    baseline: pd.Series,
    production: pd.Series,
    config: DriftConfig,
) -> FeatureDrift:
    base = pd.to_numeric(baseline, errors="coerce").to_numpy(dtype=float)
    prod = pd.to_numeric(production, errors="coerce").to_numpy(dtype=float)
    psi = _numeric_psi(base, prod, config.n_bins, config.epsilon)
    ks_stat, ks_pvalue = _ks_test(base, prod)
    mean_shift, std_shift = _standardized_shifts(base, prod)
    missing_shift = _missing_shift(baseline, production)

    severity = _max_severity(
        [
            _severity(psi, config.psi_medium, config.psi_high),
            _severity(ks_stat, config.ks_medium, config.ks_high),
            _severity(
                abs(mean_shift) if mean_shift is not None else None,
                config.shift_medium,
                config.shift_high,
            ),
            _severity(
                abs(std_shift) if std_shift is not None else None,
                config.shift_medium,
                config.shift_high,
            ),
            _severity(
                abs(missing_shift) if missing_shift is not None else None,
                config.missingness_medium,
                config.missingness_high,
            ),
        ]
    )
    notes: list[str] = []
    if psi is None:
        notes.append("PSI undefined (constant or near-constant baseline).")
    if float(np.nanstd(base)) == 0.0:
        notes.append("Standardized shifts undefined for a constant baseline; PSI/KS apply.")

    return FeatureDrift(
        feature=feature,
        kind="numeric",
        severity=severity,
        psi=_round(psi),
        ks_stat=_round(ks_stat),
        ks_pvalue=_round(ks_pvalue),
        mean_shift=_round(mean_shift),
        std_shift=_round(std_shift),
        missingness_shift=_round(missing_shift),
        notes=notes,
    )


def _categorical_feature_drift(
    feature: str,
    baseline: pd.Series,
    production: pd.Series,
    config: DriftConfig,
) -> FeatureDrift:
    psi, total_variation, new_categories = _categorical_drift(
        baseline, production, config.epsilon
    )
    missing_shift = _missing_shift(baseline, production)
    severity = _max_severity(
        [
            _severity(psi, config.psi_medium, config.psi_high),
            _severity(total_variation, config.ks_medium, config.ks_high),
            _severity(
                abs(missing_shift) if missing_shift is not None else None,
                config.missingness_medium,
                config.missingness_high,
            ),
        ]
    )
    notes: list[str] = []
    if new_categories:
        notes.append(f"new production categories: {new_categories}")

    return FeatureDrift(
        feature=feature,
        kind="categorical",
        severity=severity,
        psi=_round(psi),
        total_variation=_round(total_variation),
        missingness_shift=_round(missing_shift),
        new_categories=new_categories,
        notes=notes,
    )


def detect_drift(
    baseline: pd.DataFrame,
    production: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    config: DriftConfig | None = None,
) -> DriftReport:
    """Compare a production window against a baseline window, per feature."""
    config = config or DriftConfig()
    if baseline is None or production is None:
        raise DriftError("baseline and production frames are required")
    if len(baseline) == 0:
        raise DriftError("baseline frame has no rows")
    if len(production) == 0:
        raise DriftError("production frame has no rows")

    notes: list[str] = []
    if columns is None:
        columns = [c for c in baseline.columns if c in production.columns]
        only_baseline = [c for c in baseline.columns if c not in production.columns]
        only_production = [c for c in production.columns if c not in baseline.columns]
        if only_baseline:
            notes.append(f"columns only in baseline (skipped): {only_baseline}")
        if only_production:
            notes.append(f"columns only in production (skipped): {only_production}")
    else:
        missing_baseline = [c for c in columns if c not in baseline.columns]
        missing_production = [c for c in columns if c not in production.columns]
        if missing_baseline:
            raise DriftError(f"column(s) not in baseline: {missing_baseline}")
        if missing_production:
            raise DriftError(f"column(s) not in production: {missing_production}")
        columns = list(columns)

    if not columns:
        raise DriftError("no common columns to compare")

    features: list[FeatureDrift] = []
    for column in columns:
        base_series = baseline[column]
        prod_series = production[column]
        if _is_numeric(base_series) and _is_numeric(prod_series):
            features.append(_numeric_feature_drift(column, base_series, prod_series, config))
        else:
            features.append(_categorical_feature_drift(column, base_series, prod_series, config))

    overall = _max_severity([f.severity for f in features])
    thresholds = {
        "psi": [config.psi_medium, config.psi_high],
        "ks_stat": [config.ks_medium, config.ks_high],
        "shift": [config.shift_medium, config.shift_high],
        "missingness": [config.missingness_medium, config.missingness_high],
    }
    notes.append("Severity thresholds are demo heuristics, NOT industry standards.")
    return DriftReport(
        features=features,
        overall_severity=overall,
        thresholds=thresholds,
        notes=notes,
    )

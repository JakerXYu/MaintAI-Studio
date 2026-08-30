"""Synthetic production-replay batches for drift/anomaly testing.

Generates four deterministic batches from a baseline DataFrame — normal, mild,
severe, and increased-failure-risk — with a fixed seed. The input is never
mutated. These are clearly-labelled synthetic transformations for exercising the
monitoring core; they are NOT real production data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from maintai.monitoring.contracts import REPLAY_KINDS, ReplayBatch, ReplayKind

# (mean shift in std units, noise in std units, categorical resample probability)
_SHIFT_SPECS: dict[str, dict[str, float]] = {
    "normal": {"mean_sigma": 0.0, "noise_sigma": 0.05, "categorical_swap": 0.0},
    "mild": {"mean_sigma": 0.3, "noise_sigma": 0.1, "categorical_swap": 0.2},
    "severe": {"mean_sigma": 1.5, "noise_sigma": 0.3, "categorical_swap": 0.6},
    "increased_failure_risk": {
        "mean_sigma": 0.8,
        "noise_sigma": 0.2,
        "categorical_swap": 0.3,
    },
}

_KIND_ORDER = {kind: index for index, kind in enumerate(REPLAY_KINDS)}


@dataclass
class ReplayOutput:
    """Deterministic replay result: frames + JSON-safe metadata per batch."""

    frames: dict[str, pd.DataFrame]
    metadata: dict[str, ReplayBatch]


def _numeric_columns(frame: pd.DataFrame, target: str | None) -> list[str]:
    return [
        c
        for c in frame.columns
        if c != target
        and pd.api.types.is_numeric_dtype(frame[c].dtype)
        and not pd.api.types.is_bool_dtype(frame[c].dtype)
    ]


def _categorical_columns(frame: pd.DataFrame, numeric: list[str], target: str | None) -> list[str]:
    return [c for c in frame.columns if c != target and c not in numeric]


def _shift_numeric(
    series: pd.Series,
    rng: np.random.Generator,
    mean_sigma: float,
    noise_sigma: float,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = float(np.nanstd(values.to_numpy(dtype=float)))
    if not np.isfinite(std) or std <= 0:
        return series  # constant numeric column: leave as-is (safe)
    noise = rng.normal(loc=mean_sigma * std, scale=noise_sigma * std, size=len(series))
    return values + noise


def _shift_categorical(series: pd.Series, rng: np.random.Generator, swap_prob: float) -> pd.Series:
    if swap_prob <= 0:
        return series
    values = series.to_numpy(dtype=object)
    non_null = pd.notna(series).to_numpy()
    pool = values[non_null]
    if pool.size == 0:
        return series
    mask = non_null & (rng.random(len(series)) < swap_prob)
    indices = np.flatnonzero(mask)
    if indices.size:
        values[indices] = rng.choice(pool, size=indices.size, replace=True)
    return pd.Series(values, index=series.index, name=series.name)


def _apply_flatline(frame: pd.DataFrame, numeric: list[str]) -> None:
    if not numeric:
        return
    column = numeric[0]  # deterministic choice
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    constant = float(np.nanmedian(values))  # stall at the column median
    n = len(frame)
    start = n // 3
    length = max(1, n // 3)
    values[start : start + length] = constant
    frame[column] = values


def _shift_target(series: pd.Series, kind: str, rng: np.random.Generator) -> pd.Series:
    """Increase the failure rate for the increased_failure_risk batch (binary only)."""
    if kind != "increased_failure_risk":
        return series
    values = pd.to_numeric(series, errors="coerce")
    unique = np.unique(values.dropna().to_numpy())
    unique_values = set(unique.astype(float).tolist())
    if not unique_values or not unique_values <= {0.0, 1.0} or 0.0 not in unique_values:
        return series  # non-binary target: leave unchanged (synthetic convention)
    low, high = 0.0, 1.0
    negative_mask = (values == low).to_numpy(dtype=bool)
    negative_idx = np.flatnonzero(negative_mask)
    if negative_idx.size == 0:
        return series
    flip_count = max(1, int(0.3 * negative_idx.size))
    flip_idx = rng.choice(negative_idx, size=flip_count, replace=False)
    out = values.copy()
    out.iloc[flip_idx] = high
    return out


def _target_shift_eligible(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    unique = set(np.unique(values).tolist())
    return bool(unique) and unique <= {0.0, 1.0} and 0.0 in unique


def _transformations(
    kind: str,
    *,
    target_shifted: bool,
    flatline: bool,
) -> tuple[list[str], dict[str, object]]:
    spec = _SHIFT_SPECS[kind]
    transforms = ["synthetic replay — NOT real production data"]
    params: dict[str, object] = {
        "numeric_mean_shift_sigma": spec["mean_sigma"],
        "numeric_noise_sigma": spec["noise_sigma"],
        "categorical_swap_probability": spec["categorical_swap"],
        "flatline_injected": flatline,
        "target_shifted": kind == "increased_failure_risk" and target_shifted,
    }
    if kind == "normal":
        transforms.append("numeric: additive Gaussian noise only (no systematic shift)")
        transforms.append("categorical: unchanged")
    else:
        transforms.append(
            f"numeric: mean shift ({spec['mean_sigma']}x std) + Gaussian noise "
            f"({spec['noise_sigma']}x std)"
        )
        transforms.append(
            f"categorical: {spec['categorical_swap']:.0%} of values resampled"
        )
    if flatline:
        transforms.append("flatline: first numeric column stalled over a contiguous block")
    if kind == "increased_failure_risk" and target_shifted:
        transforms.append("target: failure rate increased (binary target)")
    return transforms, params


def _make_batch(
    kind: ReplayKind,
    frame: pd.DataFrame,
    rng: np.random.Generator,
    target: str | None,
    flatline: bool,
) -> pd.DataFrame:
    spec = _SHIFT_SPECS[kind]
    out = frame.copy()
    numeric = _numeric_columns(frame, target)
    categorical = _categorical_columns(frame, numeric, target)

    for column in numeric:
        out[column] = _shift_numeric(
            out[column], rng, spec["mean_sigma"], spec["noise_sigma"]
        )
    for column in categorical:
        out[column] = _shift_categorical(out[column], rng, spec["categorical_swap"])
    if flatline:
        _apply_flatline(out, numeric)
    if target is not None and target in frame.columns:
        out[target] = _shift_target(out[target], kind, rng)
    return out


def generate_batches(
    baseline: pd.DataFrame,
    *,
    seed: int = 42,
    target: str | None = None,
    flatline: bool = False,
) -> ReplayOutput:
    """Generate the four replay batches deterministically without mutating input."""
    if baseline is None or len(baseline) == 0:
        raise ValueError("baseline must be a non-empty DataFrame")

    frames: dict[str, pd.DataFrame] = {}
    metadata: dict[str, ReplayBatch] = {}
    for kind in REPLAY_KINDS:
        batch_seed = seed + _KIND_ORDER[kind]
        rng = np.random.default_rng(batch_seed)
        frames[kind] = _make_batch(kind, baseline, rng, target, flatline)
        target_shifted = (
            target is not None
            and target in baseline.columns
            and _target_shift_eligible(baseline[target])
        )
        transforms, params = _transformations(
            kind,
            target_shifted=target_shifted,
            flatline=flatline,
        )
        metadata[kind] = ReplayBatch(
            kind=kind,
            row_count=len(baseline),
            seed=batch_seed,
            transformations=transforms,
            params=params,
        )
    return ReplayOutput(frames=frames, metadata=metadata)

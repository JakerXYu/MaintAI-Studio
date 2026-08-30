"""Unit tests for deterministic anomaly detection."""

import numpy as np
import pandas as pd
import pytest

from maintai.monitoring.anomaly import (
    AnomalyConfig,
    AnomalyError,
    fit_baseline,
    score,
)


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "sensor_a": rng.normal(0.0, 1.0, n),
            "sensor_b": rng.normal(10.0, 3.0, n),
        }
    )


def test_fit_and_score_basic():
    frame = _frame()
    baseline = fit_baseline(frame)
    result = score(baseline, frame)
    assert len(result.rows) == len(frame)
    assert set(result.numeric_features) == {"sensor_a", "sensor_b"}
    assert all(row.robust_score >= 0 for row in result.rows)
    assert result.anomaly_rate == pytest.approx(result.flagged_count / len(frame))
    assert any("not claimed" in note.lower() for note in result.notes)


def test_fixed_seed_is_reproducible():
    frame = _frame()
    config = AnomalyConfig(seed=7)
    r1 = score(fit_baseline(frame, config=config), frame)
    r2 = score(fit_baseline(frame, config=config), frame)
    assert [row.combined_score for row in r1.rows] == [row.combined_score for row in r2.rows]
    assert [row.flag for row in r1.rows] == [row.flag for row in r2.rows]


def test_contamination_is_configurable_and_passed_through():
    frame = _frame()
    baseline = fit_baseline(frame, config=AnomalyConfig(contamination=0.2))
    result = score(baseline, frame)
    assert result.contamination == 0.2
    assert result.threshold == baseline.flag_threshold
    assert result.anomaly_rate == pytest.approx(0.2, abs=0.02)
    assert "contamination" in baseline.threshold_source


def test_invalid_contamination_is_rejected():
    with pytest.raises(ValueError, match="contamination"):
        AnomalyConfig(contamination=0.8)
    with pytest.raises(ValueError, match="contamination"):
        AnomalyConfig(contamination="estimated")


def test_constant_baseline_is_safe():
    frame = pd.DataFrame({"a": [3.0] * 20, "b": [1.0] * 20})
    baseline = fit_baseline(frame)
    result = score(baseline, frame)
    assert len(result.rows) == 20
    assert all(row.robust_score == 0.0 for row in result.rows)


def test_mad_zero_falls_back_to_std():
    # Median=1.0 and MAD=0 but std>0 -> fallback to std (no divide-by-zero).
    values = [1.0] * 18 + [2.0, 3.0]
    frame = pd.DataFrame({"a": values, "b": np.linspace(0.0, 1.0, 20)})
    baseline = fit_baseline(frame)
    result = score(baseline, frame)
    assert len(result.rows) == 20
    assert baseline.mads["a"] == 0.0
    assert baseline.stds["a"] > 0.0


def test_missing_values_imputed_stably():
    frame = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0, np.nan, 4.0] * 10,
            "b": [np.nan] * 50,  # entirely missing column
        }
    )
    baseline = fit_baseline(frame)
    result = score(baseline, frame)
    assert len(result.rows) == 50
    assert all(np.isfinite(row.combined_score) for row in result.rows)
    missing_row = result.rows[3]
    feature_a = next(
        feature for feature in missing_row.top_deviating_features if feature.feature == "a"
    )
    assert feature_a.value is None


def test_top_deviating_features_order():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "a": rng.normal(0, 1, 50),
            "b": rng.normal(0, 1, 50),
            "c": rng.normal(0, 1, 50),
        }
    )
    baseline = fit_baseline(frame)
    outlier = frame.copy()
    outlier.loc[0, "b"] = 50.0
    result = score(baseline, outlier)
    top = result.rows[0].top_deviating_features
    assert top[0].feature == "b"
    assert len(top) == 3


def test_small_and_invalid_data_errors():
    with pytest.raises(AnomalyError):
        fit_baseline(pd.DataFrame())
    with pytest.raises(AnomalyError):
        fit_baseline(pd.DataFrame({"x": ["a", "b", "c"]}))
    with pytest.raises(AnomalyError):
        fit_baseline(pd.DataFrame({"x": [1.0, 2.0]}))
    baseline = fit_baseline(_frame())
    with pytest.raises(AnomalyError):
        score(baseline, pd.DataFrame())
    with pytest.raises(AnomalyError):
        score(baseline, pd.DataFrame({"other": [1.0, 2.0, 3.0]}))

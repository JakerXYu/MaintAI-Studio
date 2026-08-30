"""Unit tests for deterministic drift detection."""

import numpy as np
import pandas as pd
import pytest

from maintai.monitoring.contracts import SEVERITY_ORDER
from maintai.monitoring.drift import DriftConfig, DriftError, detect_drift


def test_identical_frames_are_low_drift():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(0, 1, 300)})
    report = detect_drift(frame, frame)
    assert report.overall_severity == "LOW"
    feature = report.features[0]
    assert feature.psi is not None and feature.psi < 0.01
    assert feature.ks_stat is not None and feature.ks_stat < 0.05
    assert feature.mean_shift == pytest.approx(0.0, abs=1e-6)


def test_mild_and_severe_numeric_drift():
    rng = np.random.default_rng(0)
    base = pd.DataFrame({"a": rng.normal(0, 1, 500)})
    mild = pd.DataFrame({"a": rng.normal(0.3, 1, 500)})
    severe = pd.DataFrame({"a": rng.normal(2.0, 1.5, 500)})
    mild_report = detect_drift(base, mild)
    severe_report = detect_drift(base, severe)
    assert severe_report.overall_severity != "LOW"
    assert SEVERITY_ORDER[severe_report.overall_severity] >= SEVERITY_ORDER[
        mild_report.overall_severity
    ]


def test_ks_statistic_and_pvalue_reported():
    rng = np.random.default_rng(1)
    base = pd.DataFrame({"a": rng.normal(0, 1, 400)})
    prod = pd.DataFrame({"a": rng.normal(1.0, 1, 400)})
    feature = detect_drift(base, prod).features[0]
    assert feature.ks_stat is not None and 0 <= feature.ks_stat <= 1
    assert feature.ks_pvalue is not None and 0 <= feature.ks_pvalue <= 1


def test_categorical_psi_new_category_and_total_variation():
    base = pd.DataFrame({"c": ["a"] * 40 + ["b"] * 40 + ["c"] * 20})
    prod = pd.DataFrame({"c": ["a"] * 20 + ["b"] * 30 + ["c"] * 10 + ["new"] * 40})
    feature = detect_drift(base, prod).features[0]
    assert feature.kind == "categorical"
    assert "new" in feature.new_categories
    assert feature.psi is not None and feature.psi > 0
    assert feature.total_variation is not None and feature.total_variation > 0
    assert feature.severity != "LOW"


def test_missingness_shift_reported():
    base = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0] * 20})
    prod = pd.DataFrame({"a": [1.0, np.nan, np.nan, 4.0] * 20})
    feature = detect_drift(base, prod).features[0]
    assert feature.missingness_shift is not None and feature.missingness_shift > 0
    assert feature.severity == "HIGH"


def test_constant_numeric_baseline_is_safe():
    base = pd.DataFrame({"a": [5.0] * 30})
    prod = pd.DataFrame({"a": [5.0] * 30})
    feature = detect_drift(base, prod).features[0]
    assert feature.psi is None  # constant baseline -> PSI undefined
    assert feature.severity == "LOW"


def test_psi_stays_finite_for_out_of_range_production():
    base = pd.DataFrame({"a": np.linspace(0, 1, 100)})
    prod = pd.DataFrame({"a": np.linspace(10, 20, 100)})
    feature = detect_drift(base, prod).features[0]
    assert feature.psi is not None and np.isfinite(feature.psi)
    assert feature.psi > 0.25
    assert feature.severity == "HIGH"


def test_shifted_constant_baseline_is_high_without_fake_standard_units():
    base = pd.DataFrame({"a": [5.0] * 40})
    prod = pd.DataFrame({"a": [7.0] * 40})
    feature = detect_drift(base, prod).features[0]
    assert feature.severity == "HIGH"
    assert feature.mean_shift is None
    assert any("constant baseline" in note.lower() for note in feature.notes)


def test_invalid_inputs_raise():
    with pytest.raises(DriftError):
        detect_drift(pd.DataFrame(), pd.DataFrame({"a": [1.0]}))
    with pytest.raises(DriftError):
        detect_drift(pd.DataFrame({"a": [1.0]}), pd.DataFrame())
    with pytest.raises(DriftError):
        detect_drift(pd.DataFrame({"a": [1.0]}), pd.DataFrame({"b": [2.0]}))
    with pytest.raises(DriftError):
        detect_drift(
            pd.DataFrame({"a": [1.0]}),
            pd.DataFrame({"a": [2.0]}),
            columns=["missing"],
        )


def test_configurable_thresholds():
    rng = np.random.default_rng(0)
    base = pd.DataFrame({"a": rng.normal(0, 1, 500)})
    prod = pd.DataFrame({"a": rng.normal(0.2, 1, 500)})
    default = detect_drift(base, prod)
    strict = detect_drift(
        base,
        prod,
        config=DriftConfig(
            psi_medium=0.05, psi_high=0.1, ks_medium=0.05, ks_high=0.1
        ),
    )
    assert SEVERITY_ORDER[strict.overall_severity] >= SEVERITY_ORDER[default.overall_severity]


def test_report_labels_thresholds_as_heuristics():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(0, 1, 100)})
    report = detect_drift(frame, frame)
    assert any("heuristic" in note.lower() for note in report.notes)
    assert set(report.thresholds) == {"psi", "ks_stat", "shift", "missingness"}

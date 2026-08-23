"""Target-leakage tests (synthetic data only)."""

import numpy as np
import pandas as pd

from maintai.data.leakage import detect


def test_exact_duplicate_feature_blocked():
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "leak": [0, 1, 0, 1, 0, 1],
            "feature": np.linspace(0, 1, 6),
        }
    )
    report = detect(frame, target_col="target")
    assert report.verdict == "block"
    assert "leak" in report.excluded_features


def test_near_one_to_one_id_blocked():
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "serial_id": ["s0", "s1", "s2", "s3", "s4", "s5"],
        }
    )
    report = detect(frame, target_col="target")
    assert report.verdict == "block"
    assert "serial_id" in report.excluded_features


def test_label_derived_name_blocked():
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "label": [0, 1, 0, 1, 0, 1],
        }
    )
    report = detect(frame, target_col="target")
    assert "label" in report.excluded_features


def test_post_event_name_warns():
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "failure_note": ["a", "b", "a", "b", "a", "b"],
        }
    )
    report = detect(frame, target_col="target")
    assert report.verdict == "warning"
    kinds = {issue.kind for issue in report.issues}
    assert "post_event_name" in kinds


def test_asset_id_is_not_blocked():
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "asset_id": ["a1", "a2", "a3", "a4", "a5", "a6"],
        }
    )
    report = detect(frame, target_col="target", asset_id_col="asset_id")
    assert "asset_id" not in report.excluded_features


def test_clean_features_are_safe():
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "pressure": np.linspace(1.0, 2.0, 6),
            "temp": np.linspace(20.0, 30.0, 6),
        }
    )
    report = detect(frame, target_col="target")
    assert report.verdict == "safe"
    assert report.excluded_features == []


def test_known_timestamp_is_not_a_leakage_warning():
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "ts": pd.date_range("2024-01-01", periods=6, freq="h"),
            "pressure": np.linspace(1.0, 2.0, 6),
        }
    )
    report = detect(frame, target_col="target", timestamp_col="ts")
    assert report.verdict == "safe"


def test_asset_class_name_is_not_automatically_blocked():
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1],
            "asset_class": ["pump", "motor", "pump", "motor", "pump", "motor"],
        }
    )
    report = detect(frame, target_col="target")
    assert "asset_class" not in report.excluded_features

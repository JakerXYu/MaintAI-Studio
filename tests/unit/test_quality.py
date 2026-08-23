"""Data-quality tests (synthetic data only)."""

import numpy as np
import pandas as pd

from maintai.data.quality import QualityConfig, assess


def _finding_checks(report):
    return {f.check for f in report.findings}


def test_constant_and_near_constant_sensors():
    frame = pd.DataFrame(
        {
            "constant": [5.0] * 2000,
            "near": [1.0] * 1999 + [2.0],
            "healthy": np.linspace(0.0, 1.0, 2000),
        }
    )
    report = assess(frame)
    assert "constant_sensor" in _finding_checks(report)
    assert "near_constant_sensor" in _finding_checks(report)
    assert report.score < 100


def test_flatline_segment_detected_with_timestamp():
    timestamps = pd.date_range("2024-01-01", periods=12, freq="h")
    frame = pd.DataFrame(
        {
            "ts": timestamps,
            "sensor": [1.0] * 11 + [2.0],
        }
    )
    report = assess(frame, timestamp_col="ts")
    assert "flatline" in _finding_checks(report)


def test_robust_outlier_rate():
    base = [8, 9, 10, 11, 12] * 3
    frame = pd.DataFrame({"sensor": base + [1000.0, -500.0]})
    report = assess(frame)
    assert "outlier" in _finding_checks(report)
    assert "sensor" in report.checks["outlier_rates"]


def test_timestamp_gaps_duplicates_irregularity():
    ts = [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-01 00:00:01"),
        pd.Timestamp("2024-01-01 00:00:02"),
        pd.Timestamp("2024-01-01 00:00:03"),
        pd.Timestamp("2024-01-01 00:00:04"),
        pd.Timestamp("2024-01-01 00:00:04"),
        pd.Timestamp("2024-01-01 00:01:39"),
    ]
    frame = pd.DataFrame({"ts": ts, "sensor": np.linspace(0, 1, len(ts))})
    report = assess(frame, timestamp_col="ts")
    checks = _finding_checks(report)
    assert "duplicate_timestamps" in checks
    assert "timestamp_gaps" in checks


def test_same_timestamp_across_assets_is_not_a_duplicate():
    frame = pd.DataFrame(
        {
            "asset": ["A", "B", "A", "B"],
            "ts": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]
            ),
            "sensor": [1.0, 5.0, 2.0, 6.0],
        }
    )
    report = assess(frame, timestamp_col="ts", asset_id_col="asset")
    assert report.checks["duplicate_timestamps"] == 0


def test_flatline_does_not_cross_asset_boundaries():
    rows = []
    for i in range(6):
        timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i)
        rows.append({"asset": "A", "ts": timestamp, "sensor": 1.0})
        rows.append({"asset": "B", "ts": timestamp, "sensor": 1.0})
    frame = pd.DataFrame(rows)
    report = assess(
        frame,
        timestamp_col="ts",
        asset_id_col="asset",
        config=QualityConfig(flatline_min_length=10, flatline_min_ratio=0.5),
    )
    assert "flatline" not in _finding_checks(report)


def test_asset_imbalance():
    frame = pd.DataFrame(
        {
            "asset": ["A"] * 100 + ["B"] + ["C"] + ["D"] + ["E"],
            "sensor": np.linspace(0, 1, 104),
        }
    )
    report = assess(frame, asset_id_col="asset")
    assert "asset_imbalance" in _finding_checks(report)


def test_target_imbalance():
    frame = pd.DataFrame(
        {
            "target": [0] * 99 + [1],
            "sensor": np.linspace(0, 1, 100),
        }
    )
    report = assess(frame, target_col="target")
    assert "target_imbalance" in _finding_checks(report)


def test_trainable_sample_count_too_small():
    frame = pd.DataFrame(
        {
            "target": [0, 1] * 10,
            "sensor": np.linspace(0, 1, 20),
        }
    )
    report = assess(frame, target_col="target")
    assert "trainable_samples" in _finding_checks(report)
    assert report.trainable_sample_count == 20


def test_clean_data_scores_full():
    frame = pd.DataFrame({"sensor": np.arange(200, dtype=float)})
    report = assess(frame)
    assert report.score == 100.0
    assert report.penalties == []


def test_missing_penalty():
    frame = pd.DataFrame(
        {
            "a": [np.nan] * 100,
            "b": np.linspace(0, 1, 100),
        }
    )
    report = assess(frame)
    assert "missing" in _finding_checks(report)
    assert report.score < 100


def test_score_is_bounded_and_explainable():
    frame = pd.DataFrame(
        {
            "a": [np.nan] * 50,
            "b": [1.0] * 50,
        }
    )
    report = assess(frame, config=QualityConfig())
    assert 0.0 <= report.score <= 100.0
    assert report.label == "MaintAI heuristic data health score"
    assert sum(p.points for p in report.penalties) >= 100.0 - report.score - 0.011

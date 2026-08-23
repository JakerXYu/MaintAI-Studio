"""Data-split tests (synthetic data only)."""

import numpy as np
import pandas as pd
import pytest

from maintai.data.split import SplitConfig, SplitError, split


def _assert_no_asset_overlap(frame, result, asset_col):
    train_assets = set(frame.loc[result.train_indices, asset_col])
    test_assets = set(frame.loc[result.test_indices, asset_col])
    assert train_assets.isdisjoint(test_assets)


def test_group_time_no_future_leakage():
    rows = []
    for asset in ("A", "B", "C"):
        for i in range(20):
            rows.append({"asset": asset, "ts": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i)})
    frame = pd.DataFrame(rows)
    frame["target"] = np.arange(len(frame)) % 2
    result = split(
        frame,
        target_col="target",
        asset_id_col="asset",
        timestamp_col="ts",
        task_type="binary_classification",
        config=SplitConfig(test_size=0.3, validation_size=0.0),
    )
    assert result.strategy == "group_time"
    train = frame.loc[result.train_indices]
    test = frame.loc[result.test_indices]
    for asset in frame["asset"].unique():
        train_asset = train[train["asset"] == asset]["ts"]
        test_asset = test[test["asset"] == asset]["ts"]
        if not train_asset.empty and not test_asset.empty:
            assert train_asset.max() <= test_asset.min()


def test_group_split_no_overlap():
    frame = pd.DataFrame(
        {
            "asset": [f"A{i % 8}" for i in range(80)],
            "feature": np.linspace(0, 1, 80),
        }
    )
    result = split(
        frame,
        asset_id_col="asset",
        config=SplitConfig(test_size=0.25, validation_size=0.0),
    )
    assert result.strategy == "group"
    _assert_no_asset_overlap(frame, result, "asset")


def test_chronological_no_future_leakage():
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=50, freq="h"),
            "feature": np.linspace(0, 1, 50),
        }
    )
    result = split(
        frame,
        timestamp_col="ts",
        config=SplitConfig(test_size=0.2, validation_size=0.0),
    )
    assert result.strategy == "chronological"
    train_ts = frame.loc[result.train_indices, "ts"]
    test_ts = frame.loc[result.test_indices, "ts"]
    assert train_ts.max() <= test_ts.min()


def test_classification_uses_stratified():
    frame = pd.DataFrame(
        {
            "target": [0] * 60 + [1] * 40,
            "feature": np.linspace(0, 1, 100),
        }
    )
    result = split(
        frame,
        target_col="target",
        task_type="binary_classification",
        config=SplitConfig(test_size=0.2, validation_size=0.0),
    )
    assert result.strategy == "stratified"
    train = frame.loc[result.train_indices, "target"]
    test = frame.loc[result.test_indices, "target"]
    assert set(train.unique()) == {0, 1}
    assert set(test.unique()) == {0, 1}
    assert train.value_counts(normalize=True)[1] == pytest.approx(0.4, abs=0.05)
    assert test.value_counts(normalize=True)[1] == pytest.approx(0.4, abs=0.05)


def test_regression_uses_fixed_seed_random_and_reproducible():
    frame = pd.DataFrame({"target": np.linspace(0, 100, 200), "feature": np.linspace(0, 1, 200)})
    config = SplitConfig(test_size=0.2, validation_size=0.0, random_state=42)
    a = split(frame, target_col="target", task_type="regression", config=config)
    b = split(frame, target_col="target", task_type="regression", config=config)
    assert a.strategy == "random"
    assert a.train_indices == b.train_indices
    assert a.test_indices == b.test_indices
    assert len(a.test_indices) == pytest.approx(40, abs=2)


def test_validation_split_is_disjoint():
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=100, freq="h"),
            "feature": np.linspace(0, 1, 100),
        }
    )
    result = split(
        frame,
        timestamp_col="ts",
        config=SplitConfig(test_size=0.2, validation_size=0.2),
    )
    assert result.strategy == "chronological"
    assert set(result.train_indices).isdisjoint(result.test_indices)
    assert set(result.train_indices).isdisjoint(result.validation_indices)
    assert set(result.test_indices).isdisjoint(result.validation_indices)


def test_tiny_data_raises():
    frame = pd.DataFrame({"feature": [1, 2, 3]})
    with pytest.raises(SplitError):
        split(frame, config=SplitConfig(validation_size=0.0))


def test_invalid_column_raises():
    frame = pd.DataFrame({"feature": [1, 2, 3, 4]})
    with pytest.raises(SplitError):
        split(frame, asset_id_col="missing", config=SplitConfig(validation_size=0.0))


def test_single_class_stratified_raises():
    frame = pd.DataFrame({"target": [1] * 20, "feature": np.linspace(0, 1, 20)})
    with pytest.raises(SplitError):
        split(
            frame,
            target_col="target",
            task_type="binary_classification",
            config=SplitConfig(validation_size=0.0),
        )


def test_invalid_combined_split_sizes_raise():
    frame = pd.DataFrame({"feature": range(20)})
    with pytest.raises(SplitError, match="less than 1"):
        split(frame, config=SplitConfig(test_size=0.6, validation_size=0.4))


def test_single_group_raises_instead_of_returning_empty_train():
    frame = pd.DataFrame({"asset": ["A"] * 20, "feature": range(20)})
    with pytest.raises(SplitError, match="distinct assets"):
        split(
            frame,
            asset_id_col="asset",
            config=SplitConfig(test_size=0.2, validation_size=0.0),
        )


def test_validation_omission_is_explicit_for_rare_class():
    frame = pd.DataFrame({"target": [0] * 98 + [1] * 2, "feature": range(100)})
    result = split(
        frame,
        target_col="target",
        task_type="binary_classification",
        config=SplitConfig(test_size=0.5, validation_size=0.2),
    )
    assert result.validation_indices == []
    assert any("validation omitted" in item for item in result.evidence)

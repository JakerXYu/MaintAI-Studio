"""Preprocessing tests: roles, unknown categories, train-only fitting, feature names."""

import numpy as np
import pandas as pd
import pytest

from maintai.ml.preprocess import (
    PreprocessError,
    build_preprocessor,
    get_feature_names,
    resolve_features,
)


def test_unknown_category_ignored_at_inference():
    frame = pd.DataFrame({"cat": ["a", "b", "a", "b", "a", "b"]})
    preprocessor = build_preprocessor(frame, ["cat"])
    preprocessor.fit(frame.loc[:3])
    names = get_feature_names(preprocessor)
    assert set(names) == {"cat_a", "cat_b"}

    test = pd.DataFrame({"cat": ["c", "a"]})
    X = preprocessor.transform(test)
    row_c = dict(zip(names, X[0], strict=True))
    row_a = dict(zip(names, X[1], strict=True))
    assert row_c["cat_a"] == 0.0 and row_c["cat_b"] == 0.0  # unknown -> all zeros
    assert row_a["cat_a"] == 1.0 and row_a["cat_b"] == 0.0


def test_preprocessor_fit_uses_only_train_fold():
    frame = pd.DataFrame({"num": [1.0, 2.0, 3.0, np.nan, 1000.0, 1001.0, 1002.0, 1003.0]})
    train_idx = [0, 1, 2, 3]
    preprocessor = build_preprocessor(frame, ["num"])
    preprocessor.fit(frame.loc[train_idx])

    imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    train = frame.loc[train_idx, "num"]
    # Statistics must come from the train fold only, never the test-fold outliers.
    assert imputer.statistics_[0] == pytest.approx(train.median())
    assert scaler.mean_[0] == pytest.approx(2.0)


def test_identifier_and_timestamp_excluded_from_features():
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0],
            "ts": pd.date_range("2024-01-01", periods=5),
            "asset_id": ["a0", "a1", "a2", "a3", "a4"],
            "num": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    assert resolve_features(frame, "target", [], []) == ["num"]


def test_explicitly_excluded_columns_dropped():
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "num": [1.0, 2.0, 3.0, 4.0],
            "leak": [1, 2, 3, 4],
        }
    )
    assert resolve_features(frame, "target", [], ["leak"]) == ["num"]


def test_boolean_mapping_and_mode_impute():
    frame = pd.DataFrame({"flag": ["yes", "no", "yes", "yes", np.nan, "no"]})
    preprocessor = build_preprocessor(frame, ["flag"])
    preprocessor.fit(frame.loc[:3])  # train mode = 1.0 (yes)
    out = preprocessor.transform(frame)
    assert list(out[:, 0]) == [1.0, 0.0, 1.0, 1.0, 1.0, 0.0]


def test_feature_names_are_traceable():
    frame = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0],
            "cat": ["a", "b", "a", "b"],
            "flag": [True, False, True, False],
        }
    )
    preprocessor = build_preprocessor(frame, ["num", "cat", "flag"])
    preprocessor.fit(frame)
    assert get_feature_names(preprocessor) == ["num", "cat_a", "cat_b", "flag"]


def test_resolve_missing_feature_raises():
    frame = pd.DataFrame({"target": [0, 1, 0, 1], "num": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(PreprocessError, match="not found"):
        resolve_features(frame, "target", ["ghost"], [])


def test_resolve_empty_features_raises():
    frame = pd.DataFrame({"target": [0, 1, 0, 1], "num": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(PreprocessError, match="no usable"):
        resolve_features(frame, "target", [], ["num"])

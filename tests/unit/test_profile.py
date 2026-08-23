"""Profiling tests (synthetic data only)."""

import numpy as np
import pandas as pd
import pytest

from maintai.data.profile import profile


def test_profile_basics():
    frame = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0, 4.0],
            "category": ["a", "b", "a", "b"],
        }
    )
    report = profile(frame)
    assert report.row_count == 4
    assert report.column_count == 2
    assert report.cell_count == 8
    assert report.missing_cells == 0
    assert report.duplicate_rows == 0

    numeric = next(c for c in report.columns if c.name == "numeric")
    assert numeric.numeric.mean == 2.5
    assert numeric.numeric.median == 2.5

    category = next(c for c in report.columns if c.name == "category")
    assert category.cardinality == 2
    assert {f.value for f in category.top_categories} == {"a", "b"}


def test_profile_missing_and_duplicates():
    frame = pd.DataFrame(
        {
            "a": [1.0, 1.0, np.nan],
            "b": ["x", "x", "y"],
        }
    )
    report = profile(frame)
    assert report.duplicate_rows == 1  # rows 0 and 1 identical
    assert report.missing_cells == 1
    assert report.missing_rate == pytest.approx(1 / 6)

    col_a = next(c for c in report.columns if c.name == "a")
    assert col_a.missing_count == 1


def test_profile_constant_flag():
    frame = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
    report = profile(frame)
    by_name = {c.name: c for c in report.columns}
    assert by_name["a"].constant is True
    assert by_name["b"].constant is False

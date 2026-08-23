"""Schema inference tests (synthetic data only)."""

import numpy as np
import pandas as pd
import pytest

from maintai.data.schema import infer_schema


def test_schema_infers_core_semantic_types():
    frame = pd.DataFrame(
        {
            "Air temperature [K]": np.linspace(290.0, 310.0, 50),
            "pressure_bar": np.linspace(1.0, 3.0, 50),
            "machine_id": [f"M{i}" for i in range(50)],
            "type": ["L", "M", "H", "L", "M"] * 10,
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="h"),
            "machine_failure": [0, 1, 0, 1, 0] * 10,
        }
    )
    schema = infer_schema(frame)
    by_name = {c.name: c for c in schema.columns}

    assert by_name["Air temperature [K]"].semantic_type == "sensor_numeric"
    assert by_name["pressure_bar"].semantic_type == "sensor_numeric"
    assert by_name["machine_id"].semantic_type == "identifier_like"
    assert by_name["type"].semantic_type == "categorical"
    assert by_name["timestamp"].semantic_type == "datetime"
    assert by_name["machine_failure"].semantic_type == "boolean"

    assert "machine_failure" in schema.candidate_targets
    assert "machine_id" in schema.candidate_asset_ids
    assert "timestamp" in schema.candidate_timestamps


def test_boolean_strings_inferred():
    frame = pd.DataFrame({"flag": ["Yes", "No", "Yes", "No"], "value": [1.0, 2.0, 3.0, 4.0]})
    schema = infer_schema(frame)
    by_name = {c.name: c for c in schema.columns}
    assert by_name["flag"].semantic_type == "boolean"


def test_high_cardinality_integer_is_not_identifier_without_hint():
    frame = pd.DataFrame({"cycles": list(range(100)), "sensor": np.linspace(0, 1, 100)})
    schema = infer_schema(frame)
    by_name = {c.name: c for c in schema.columns}
    assert by_name["cycles"].semantic_type == "numeric"
    assert by_name["sensor"].semantic_type == "sensor_numeric"


def test_missing_rate_reported():
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0]})
    schema = infer_schema(frame)
    assert schema.columns[0].missing_rate == pytest.approx(0.25)

"""Unit tests for the pure UI presentation helpers (no Streamlit / HTTP)."""

from __future__ import annotations

import pytest

from maintai.ui import render as R

# -- constants ----------------------------------------------------------------


def test_quick_prompts_are_seven_with_stable_ids() -> None:
    ids = [prompt_id for prompt_id, _, _ in R.QUICK_PROMPTS]
    assert len(ids) == 7
    assert len(ids) == len(set(ids))
    assert "qp_dataset_problems" in ids
    assert "qp_why_classification" in ids


def test_model_catalog_is_documented() -> None:
    assert R.MODEL_CATALOG["binary_classification"] == (
        "logistic_regression",
        "random_forest",
        "xgboost",
    )
    assert R.MODEL_CATALOG["regression"] == ("ridge", "random_forest", "xgboost")


# -- value formatting ----------------------------------------------------------


def test_fmt_num() -> None:
    assert R.fmt_num(None) == ""
    assert R.fmt_num(3.0) == "3"
    assert R.fmt_num(0.12345678, digits=3) == "0.123"
    assert R.fmt_num("not-a-number") == "not-a-number"


def test_fmt_pct() -> None:
    assert R.fmt_pct(None) == ""
    assert R.fmt_pct(0.052) == "5.2%"


def test_severity_rank() -> None:
    assert R.severity_rank("block") > R.severity_rank("warning") > R.severity_rank("info")
    assert R.severity_rank("error") > R.severity_rank("warning")
    assert R.severity_rank(None) == 0


def test_status_predicates() -> None:
    assert R.is_bad_status("failed")
    assert R.is_bad_status("not_ready")
    assert not R.is_bad_status("succeeded")
    assert R.is_terminal_experiment("succeeded")
    assert R.is_terminal_experiment("failed")
    assert not R.is_terminal_experiment("running")


# -- payload extraction --------------------------------------------------------


def _dataset_payload() -> dict:
    return {
        "id": "ds-1",
        "status": "task_recommended",
        "task_type": "binary_classification",
        "target_column": "failure",
        "quality_score": 87.5,
        "schema": {
            "columns": [
                {
                    "name": "failure",
                    "dtype": "int",
                    "semantic_type": "binary",
                    "missing_rate": 0.0,
                    "nunique": 2,
                },
                {
                    "name": "torque",
                    "dtype": "float",
                    "semantic_type": "numeric",
                    "missing_rate": 0.1,
                    "nunique": 100,
                },
            ],
            "candidate_targets": ["failure"],
            "candidate_asset_ids": ["asset_id"],
            "candidate_timestamps": ["timestamp"],
        },
        "profile": {
            "profile": {
                "row_count": 10,
                "columns": [
                    {
                        "name": "torque",
                        "dtype": "float",
                        "semantic_type": "numeric",
                        "missing_count": 1,
                        "missing_rate": 0.1,
                        "unique_count": 9,
                        "constant": False,
                        "numeric": {"min": 0.0, "median": 5.0, "max": 10.0},
                    }
                ],
            },
            "quality": {
                "label": R.HEALTH_SCORE_LABEL,
                "max_score": 100.0,
                "score": 87.5,
                "trainable_sample_count": 10,
                "findings": [
                    {
                        "check": "missing",
                        "severity": "warning",
                        "column": "torque",
                        "message": "missing",
                        "penalty": 5.0,
                    },
                    {
                        "check": "outlier",
                        "severity": "error",
                        "column": None,
                        "message": "outlier",
                        "penalty": 10.0,
                    },
                ],
                "checks": {"outlier_rates": {"torque": 0.11}},
            },
            "leakage_report": {
                "verdict": "block",
                "excluded_features": ["serial_no"],
                "issues": [
                    {
                        "feature": "serial_no",
                        "kind": "near_one_to_one_id",
                        "severity": "block",
                        "reason": "id",
                        "evidence": {"unique_ratio": 1.0},
                    },
                    {
                        "feature": "post",
                        "kind": "post_event_name",
                        "severity": "warning",
                        "reason": "post-event",
                        "evidence": {},
                    },
                ],
            },
        },
    }


def test_extract_schema_columns() -> None:
    columns = R.extract_schema_columns(_dataset_payload())
    assert [c["name"] for c in columns] == ["failure", "torque"]


def test_extract_profile_quality_leakage() -> None:
    payload = _dataset_payload()
    assert R.extract_profile(payload)["row_count"] == 10
    assert R.extract_quality(payload)["score"] == 87.5
    assert R.extract_leakage(payload)["verdict"] == "block"
    assert R.extract_task(payload) == {}


def test_schema_candidate_columns() -> None:
    payload = _dataset_payload()
    assert R.schema_candidate_columns(payload, "candidate_targets") == ["failure"]
    assert R.schema_candidate_columns(payload, "candidate_asset_ids") == ["asset_id"]


def test_quality_finding_rows_sort_most_severe_first() -> None:
    rows = R.quality_finding_rows(R.extract_quality(_dataset_payload()))
    assert rows[0]["severity"] == "error"
    assert rows[1]["severity"] == "warning"


def test_leakage_issue_rows_sort_block_first() -> None:
    rows = R.leakage_issue_rows(R.extract_leakage(_dataset_payload()))
    assert rows[0]["severity"] == "block"
    assert rows[1]["severity"] == "warning"


def test_profile_column_rows() -> None:
    rows = R.profile_column_rows(R.extract_profile(_dataset_payload()))
    assert rows[0]["column"] == "torque"
    assert rows[0]["median"] == "5"


def test_model_run_rows_flatten_metrics() -> None:
    runs = [
        {
            "id": "run-1",
            "mlflow_run_id": "mlflow-1",
            "model_name": "random_forest",
            "status": "success",
            "primary_metric": "f1",
            "primary_metric_value": 0.9,
            "metrics": {"f1": 0.9, "recall": 0.95, "training_time_seconds": 1.5},
            "training_time_seconds": 1.5,
        }
    ]
    rows = R.model_run_rows(runs)
    assert rows[0]["model"] == "random_forest"
    assert rows[0]["f1"] == "0.9"
    assert rows[0]["recall"] == "0.95"


def test_explanation_impacts_sorted_by_abs_impact() -> None:
    explanation = {
        "top_features": [
            {"feature": "a", "value": 1.0, "impact": 0.1},
            {"feature": "b", "value": 2.0, "impact": -0.5},
            {"feature": "c", "value": 3.0, "impact": 0.2},
        ]
    }
    impacts = R.explanation_impacts(explanation)
    assert [row["feature"] for row in impacts] == ["b", "c", "a"]


def test_registered_model_rows() -> None:
    rows = R.registered_model_rows(
        [{"id": "rm-1", "name": "random_forest", "version": "1", "deployment_status": "candidate"}]
    )
    assert rows[0]["name"] == "random_forest"
    assert rows[0]["deployment"] == "candidate"


# -- CSV parse / format --------------------------------------------------------


def test_parse_csv_records_coerces_types() -> None:
    text = "a,b,c\n1,2.5,true\n3,4.5,false\n"
    records = R.parse_csv_records(text)
    assert records == [
        {"a": 1, "b": 2.5, "c": True},
        {"a": 3, "b": 4.5, "c": False},
    ]


def test_parse_csv_records_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        R.parse_csv_records("   ")


def test_parse_csv_records_no_rows_raises() -> None:
    with pytest.raises(ValueError, match="no data"):
        R.parse_csv_records("a,b\n")


def test_records_to_csv_round_trip() -> None:
    records = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    text = R.records_to_csv(records)
    assert text.splitlines()[0] == "a,b"
    assert R.parse_csv_records(text) == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]


def test_records_to_csv_empty() -> None:
    assert R.records_to_csv([]) == ""

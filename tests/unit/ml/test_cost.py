"""Cost-aware selection tests: FN/FP localisation, formula, ranking, and errors."""

import pytest

from maintai.ml.cost import CostError, compare_costs
from maintai.ml.schemas import (
    CostAssumptions,
    CostComparison,
    ModelEvaluation,
    TrainingResult,
)

_CLS_METRICS = {"precision": 0.5, "recall": 0.5, "f1": 0.5, "roc_auc": None, "pr_auc": None}


def _eval(
    name, *, cm, labels, positive_label, f1=0.5, status="success", task="binary_classification"
):
    metrics = dict(_CLS_METRICS)
    metrics["f1"] = f1
    return ModelEvaluation(
        model_name=name,
        task=task,
        status=status,
        metrics=metrics,
        confusion_matrix=cm,
        labels=labels,
        positive_label=positive_label,
    )


def _result(evaluations, *, task="binary_classification", primary="f1"):
    return TrainingResult(
        task=task,
        target="target",
        seed=42,
        primary_metric=primary,
        evaluations=evaluations,
    )


def test_positive_label_at_index_zero():
    # cm[i][j] = true label labels[i] predicted labels[j]; labels[0] is the positive class.
    ev = _eval("m", cm=[[3, 2], [4, 1]], labels=[0, 1], positive_label=0)
    comp = compare_costs(_result([ev]), CostAssumptions(fn_cost=10, fp_cost=5))
    row = comp.rows[0]
    assert row.fn == 2
    assert row.fp == 4
    assert row.expected_error_cost == 2 * 10 + 4 * 5


def test_positive_label_at_index_one():
    ev = _eval("m", cm=[[3, 2], [4, 1]], labels=[0, 1], positive_label=1)
    comp = compare_costs(_result([ev]), CostAssumptions(fn_cost=10, fp_cost=5))
    row = comp.rows[0]
    assert row.fn == 4
    assert row.fp == 2
    assert row.expected_error_cost == 4 * 10 + 2 * 5


def test_positive_label_defaults_to_last_label():
    ev = _eval("m", cm=[[3, 2], [4, 1]], labels=[0, 1], positive_label=None)
    comp = compare_costs(_result([ev]), CostAssumptions())
    row = comp.rows[0]
    assert row.fn == 4
    assert row.fp == 2


def test_expected_error_cost_formula():
    ev = _eval("m", cm=[[7, 4], [3, 6]], labels=[0, 1], positive_label=1)
    comp = compare_costs(_result([ev]), CostAssumptions(fn_cost=100, fp_cost=25))
    row = comp.rows[0]
    assert row.fn == 3
    assert row.fp == 4
    assert row.expected_error_cost == 3 * 100 + 4 * 25
    assert row.expected_error_cost == 400


def test_zero_costs_tie_break_by_fn():
    a = _eval("a", cm=[[0, 0], [5, 0]], labels=[0, 1], positive_label=1, f1=0.9)
    b = _eval("b", cm=[[0, 0], [2, 0]], labels=[0, 1], positive_label=1, f1=0.5)
    comp = compare_costs(_result([a, b]), CostAssumptions(fn_cost=0, fp_cost=0))
    assert comp.cost_best == "b"
    assert comp.rows[0].expected_error_cost == 0
    assert comp.rows[1].expected_error_cost == 0


def test_cost_tie_break_prefers_simpler_model():
    a = _eval("xgboost", cm=[[5, 0], [0, 5]], labels=[0, 1], positive_label=1)
    b = _eval("logistic_regression", cm=[[5, 0], [0, 5]], labels=[0, 1], positive_label=1)
    comp = compare_costs(_result([a, b]), CostAssumptions(fn_cost=1, fp_cost=1))
    assert comp.cost_best == "logistic_regression"


def test_metric_best_differs_from_cost_best():
    accurate = _eval(
        "accurate", cm=[[0, 100], [0, 50]], labels=[0, 1], positive_label=1, f1=0.9
    )
    cheap = _eval("cheap", cm=[[50, 0], [0, 0]], labels=[0, 1], positive_label=1, f1=0.5)
    comp = compare_costs(_result([cheap, accurate]), CostAssumptions(fn_cost=50, fp_cost=2))
    assert comp.metric_best == "accurate"
    assert comp.cost_best == "cheap"
    assert any("differs from cost_best" in note for note in comp.notes)


def test_failed_model_unavailable_and_excluded():
    good = _eval("good", cm=[[1, 0], [0, 1]], labels=[0, 1], positive_label=1, f1=0.8)
    bad = ModelEvaluation(
        model_name="bad", task="binary_classification", status="error", error="boom"
    )
    comp = compare_costs(_result([good, bad]), CostAssumptions())
    assert comp.cost_best == "good"
    bad_row = next(row for row in comp.rows if row.model_name == "bad")
    assert bad_row.status == "unavailable"
    assert bad_row.fn is None
    assert bad_row.expected_error_cost is None
    assert comp.rows[-1].model_name == "bad"


def test_missing_confusion_matrix_raises():
    ev = ModelEvaluation(
        model_name="m", task="binary_classification", status="success", metrics=dict(_CLS_METRICS)
    )
    with pytest.raises(CostError, match="confusion_matrix"):
        compare_costs(_result([ev]))


def test_missing_labels_raises():
    ev = _eval("m", cm=[[1, 0], [0, 1]], labels=[], positive_label=1)
    with pytest.raises(CostError, match="labels"):
        compare_costs(_result([ev]))


def test_multiclass_rejected():
    with pytest.raises(CostError, match="binary_classification"):
        compare_costs(_result([], task="multiclass_classification"))


def test_regression_rejected():
    with pytest.raises(CostError, match="binary_classification"):
        compare_costs(_result([], task="regression"))


def test_json_safe_roundtrip():
    ev = _eval("m", cm=[[3, 2], [4, 1]], labels=[0, 1], positive_label=1)
    comp = compare_costs(_result([ev]), CostAssumptions(fn_cost=10, fp_cost=5))
    assert isinstance(comp.model_dump_json(), str)
    assert CostComparison.model_validate(comp.model_dump()) == comp


def test_fixed_deterministic():
    ev = _eval("m", cm=[[3, 2], [4, 1]], labels=[0, 1], positive_label=1)
    result = _result([ev])
    first = compare_costs(result, CostAssumptions(fn_cost=10, fp_cost=5))
    second = compare_costs(result, CostAssumptions(fn_cost=10, fp_cost=5))
    assert first.model_dump() == second.model_dump()

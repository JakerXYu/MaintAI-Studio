"""Deterministic cost-aware model selection (P1).

Cost-aware selection extends the deterministic recommendation with a maintenance
cost lens. For each successful binary classifier it reads the confusion matrix and
labels, locates the positive class (even when it is at index 0), and computes the
expected error cost as ``FN * fn_cost + FP * fp_cost``.

The metric-best model is reused from the deterministic recommendation
(``recommend``). The cost-best model ranks by cost, then FN, then FP, then model
complexity, then model name. Failed models are reported as ``unavailable`` rows
and excluded from ranking. No LLM and no automatic deployment.

The cost numbers are demo assumptions, not actual maintenance economics.
"""

from __future__ import annotations

from maintai.ml import catalog
from maintai.ml.recommend import recommend
from maintai.ml.schemas import (
    CostAssumptions,
    CostComparison,
    ModelCostRow,
    ModelEvaluation,
    TrainingResult,
)


class CostError(Exception):
    """Raised when cost-aware selection cannot be computed from the inputs."""


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _count_fn_fp(evaluation: ModelEvaluation) -> tuple[int, int]:
    """Return ``(FN, FP)`` by locating the positive class within ``labels``."""
    matrix = evaluation.confusion_matrix
    labels = evaluation.labels
    name = evaluation.model_name
    if matrix is None:
        raise CostError(f"model {name!r}: missing confusion_matrix for cost calculation")
    if not labels:
        raise CostError(f"model {name!r}: missing labels for cost calculation")
    if len(labels) != 2:
        raise CostError(
            f"model {name!r}: cost selection requires 2 binary labels, got {len(labels)}"
        )
    if len(set(labels)) != 2:
        raise CostError(f"model {name!r}: binary labels must be two distinct classes")
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        raise CostError(f"model {name!r}: confusion_matrix must be 2x2")

    if evaluation.positive_label is not None and evaluation.positive_label not in labels:
        raise CostError(f"model {name!r}: positive_label is absent from labels")
    positive = evaluation.positive_label if evaluation.positive_label is not None else labels[1]
    positive_index = labels.index(positive)
    negative_index = 1 - positive_index
    fn = matrix[positive_index][negative_index]
    fp = matrix[negative_index][positive_index]
    return int(fn), int(fp)


def _cost_key(row: ModelCostRow) -> tuple:
    return (
        row.expected_error_cost,
        row.fn,
        row.fp,
        catalog.model_complexity(row.model_name),
        row.model_name,
    )


def compare_costs(
    result: TrainingResult,
    assumptions: CostAssumptions | None = None,
    *,
    minimum_recall: float = 0.0,
) -> CostComparison:
    """Compare models by maintenance cost alongside the deterministic metric-best."""
    if assumptions is None:
        assumptions = CostAssumptions()

    if result.task != "binary_classification":
        raise CostError(
            f"cost-aware selection requires binary_classification, got {result.task!r}"
        )

    recommendation = recommend(result, minimum_recall=minimum_recall)
    primary = recommendation.primary_metric
    metric_best = recommendation.best_model

    notes: list[str] = [
        "costs are demo assumptions, not actual maintenance economics",
        f"expected_error_cost = FN*{_fmt(assumptions.fn_cost)} + "
        f"FP*{_fmt(assumptions.fp_cost)} ({assumptions.currency_label})",
    ]

    ok_rows: list[ModelCostRow] = []
    unavailable_rows: list[ModelCostRow] = []

    for evaluation in result.evaluations:
        if evaluation.status != "success":
            unavailable_rows.append(
                ModelCostRow(model_name=evaluation.model_name, status="unavailable")
            )
            continue
        if evaluation.task != "binary_classification":
            raise CostError(
                f"model {evaluation.model_name!r}: cost selection is binary-only, "
                f"got task {evaluation.task!r}"
            )
        fn, fp = _count_fn_fp(evaluation)
        cost = fn * assumptions.fn_cost + fp * assumptions.fp_cost
        ok_rows.append(
            ModelCostRow(
                model_name=evaluation.model_name,
                fn=fn,
                fp=fp,
                expected_error_cost=cost,
                primary_metric=primary,
                value=evaluation.metrics.get(primary) if evaluation.metrics else None,
                status="ok",
            )
        )

    ok_rows.sort(key=_cost_key)
    unavailable_rows.sort(key=lambda row: row.model_name)

    if unavailable_rows:
        notes.append(
            f"{len(unavailable_rows)} failed model(s) marked unavailable "
            "and excluded from ranking"
        )
    if not ok_rows:
        notes.append("no successful binary model with a confusion matrix to compare")

    cost_best = ok_rows[0].model_name if ok_rows else None

    if metric_best is not None and cost_best is not None:
        if metric_best == cost_best:
            notes.append(f"metric_best and cost_best agree on {metric_best!r}")
        else:
            metric_row = next(row for row in ok_rows if row.model_name == metric_best)
            cost_row = next(row for row in ok_rows if row.model_name == cost_best)
            notes.append(
                f"metric_best {metric_best!r} ({primary}={_fmt(float(metric_row.value))}) "
                f"differs from cost_best {cost_best!r} "
                f"(expected_error_cost={_fmt(float(cost_row.expected_error_cost))}): "
                "cost-aware selection favours fewer FN/FP; "
                "validate against real maintenance economics"
            )

    return CostComparison(
        metric_best=metric_best,
        cost_best=cost_best,
        rows=ok_rows + unavailable_rows,
        notes=notes,
        assumptions=assumptions,
    )

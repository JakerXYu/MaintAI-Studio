"""Pure presentation helpers for the MaintAI Streamlit UI.

No Streamlit, no HTTP, no filesystem access, and no ML/database imports. These
functions transform API JSON payloads into display-ready values so they can be
unit-tested without a running server or the Streamlit runtime.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

# -- fixed, user-facing strings ------------------------------------------------

PRODUCT_DISCLAIMER = (
    "This prototype provides model-based decision support only; it does not "
    "replace qualified maintenance, safety, or engineering judgment."
)
HEALTH_SCORE_LABEL = "MaintAI heuristic data health score"
EXPLANATION_DISCLAIMER = "Model-based explanation, not a verified physical root cause."
DEMO_DEPLOY_DISCLAIMER = (
    "Demo deployment is a P0 demo-serving convenience, not production promotion."
)
FUTURE_P1_NOTE = "P1 feature (monitoring / approvals / CMMS) — not implemented in this build."

# Deterministic model catalog (mirrors the backend ML catalog; the UI never
# imports the ML package, it only describes the fixed catalog for the plan page).
MODEL_CATALOG: dict[str, tuple[str, ...]] = {
    "binary_classification": ("logistic_regression", "random_forest", "xgboost"),
    "multiclass_classification": ("logistic_regression", "random_forest", "xgboost"),
    "regression": ("ridge", "random_forest", "xgboost"),
}

# Canonical copilot quick prompts: stable id, short label, full prompt.
QUICK_PROMPTS: tuple[tuple[str, str, str], ...] = (
    (
        "qp_dataset_problems",
        "What problems do you see in this dataset?",
        "What problems do you see in this dataset?",
    ),
    (
        "qp_why_classification",
        "Why did you recommend classification?",
        "Why did you recommend classification?",
    ),
    (
        "qp_which_model",
        "Which model should I deploy and why?",
        "Which model should I deploy and why?",
    ),
    (
        "qp_explain_prediction",
        "Explain the most recent prediction",
        "Explain the most recent prediction and its driving features.",
    ),
    (
        "qp_fix_before_retraining",
        "What should I fix before retraining?",
        "What should I fix before retraining?",
    ),
    (
        "qp_experiment_results",
        "Summarize the current experiment",
        "Summarize the current experiment results, metrics, and recommendation.",
    ),
    (
        "qp_health_summary",
        "Summarize dataset health",
        "Summarize the dataset health score and its main findings.",
    ),
)

# Statuses that the UI treats as "bad" / non-operational.
_BAD_STATUSES = frozenset({"failed", "error", "not_ready", "unavailable", "down"})
# Experiment statuses that will not change again (terminal states).
_TERMINAL_EXPERIMENT_STATUSES = frozenset({"succeeded", "failed"})

# Severity ordering for findings / leakage issues (higher is more severe).
_SEVERITY_RANK = {"info": 0, "safe": 0, "warning": 1, "error": 2, "block": 3}


# -- small value helpers -------------------------------------------------------


def severity_rank(severity: Any) -> int:
    """Return a numeric rank for a severity string (higher = more severe)."""
    if not isinstance(severity, str):
        return 0
    return _SEVERITY_RANK.get(severity.lower(), 0)


def is_bad_status(status: Any) -> bool:
    """Return True for statuses that should be highlighted as unhealthy/failed."""
    return isinstance(status, str) and status.lower() in _BAD_STATUSES


def is_terminal_experiment(status: Any) -> bool:
    """Return True when an experiment has reached a final state."""
    return isinstance(status, str) and status in _TERMINAL_EXPERIMENT_STATUSES


def fmt_num(value: Any, digits: int = 4) -> str:
    """Format a numeric value compactly; empty string for non-numeric values."""
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number == int(number):
        return str(int(number))
    return f"{number:.{digits}g}"


def fmt_pct(value: Any) -> str:
    """Format a 0..1 ratio as a percentage string."""
    if value is None:
        return ""
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


# -- payload extraction helpers ------------------------------------------------


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def extract_schema_columns(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the schema column list from a dataset payload."""
    schema = _dict(dataset.get("schema"))
    columns = schema.get("columns")
    return list(columns) if isinstance(columns, list) else []


def extract_profile(dataset: dict[str, Any]) -> dict[str, Any]:
    """Return the stored profile report from a dataset payload."""
    return _dict(_dict(dataset.get("profile")).get("profile"))


def extract_quality(dataset: dict[str, Any]) -> dict[str, Any]:
    """Return the stored quality report from a dataset payload."""
    return _dict(_dict(dataset.get("profile")).get("quality"))


def extract_leakage(dataset: dict[str, Any]) -> dict[str, Any]:
    """Return the stored leakage report from a dataset payload."""
    return _dict(_dict(dataset.get("profile")).get("leakage_report"))


def extract_task(dataset: dict[str, Any]) -> dict[str, Any]:
    """Return the stored task recommendation from a dataset payload."""
    return _dict(_dict(dataset.get("profile")).get("task_recommendation"))


def schema_candidate_columns(dataset: dict[str, Any], kind: str) -> list[str]:
    """Return candidate target / asset / timestamp columns from schema inference."""
    schema = _dict(dataset.get("schema"))
    values = schema.get(kind)
    return [str(v) for v in values] if isinstance(values, list) else []


def profile_column_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a profile report's per-column data into table rows."""
    rows: list[dict[str, Any]] = []
    for column in profile.get("columns") or []:
        if not isinstance(column, dict):
            continue
        numeric = _dict(column.get("numeric"))
        rows.append(
            {
                "column": column.get("name"),
                "dtype": column.get("dtype"),
                "semantic": column.get("semantic_type"),
                "missing": column.get("missing_count"),
                "missing_rate": fmt_pct(column.get("missing_rate")),
                "unique": column.get("unique_count"),
                "constant": bool(column.get("constant")),
                "min": fmt_num(numeric.get("min")),
                "median": fmt_num(numeric.get("median")),
                "max": fmt_num(numeric.get("max")),
            }
        )
    return rows


def quality_finding_rows(quality: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten quality findings into sortable table rows (most severe first)."""
    rows: list[dict[str, Any]] = []
    for finding in quality.get("findings") or []:
        if not isinstance(finding, dict):
            continue
        rows.append(
            {
                "severity": finding.get("severity"),
                "check": finding.get("check"),
                "column": finding.get("column"),
                "message": finding.get("message"),
                "penalty": fmt_num(finding.get("penalty")),
            }
        )
    rows.sort(key=lambda row: severity_rank(row["severity"]), reverse=True)
    return rows


def leakage_issue_rows(leakage: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten leakage issues into sortable table rows (block first)."""
    rows: list[dict[str, Any]] = []
    for issue in leakage.get("issues") or []:
        if not isinstance(issue, dict):
            continue
        evidence = _dict(issue.get("evidence"))
        rows.append(
            {
                "feature": issue.get("feature"),
                "kind": issue.get("kind"),
                "severity": issue.get("severity"),
                "reason": issue.get("reason"),
                "evidence": json.dumps(evidence, sort_keys=True) if evidence else "",
            }
        )
    rows.sort(key=lambda row: severity_rank(row["severity"]), reverse=True)
    return rows


def model_run_rows(model_runs: Any) -> list[dict[str, Any]]:
    """Flatten experiment model runs into a metrics table."""
    rows: list[dict[str, Any]] = []
    for run in model_runs or []:
        if not isinstance(run, dict):
            continue
        metrics = _dict(run.get("metrics"))
        rows.append(
            {
                "model": run.get("model_name"),
                "status": run.get("status"),
                "primary_metric": run.get("primary_metric"),
                "value": fmt_num(run.get("primary_metric_value")),
                "precision": fmt_num(metrics.get("precision")),
                "recall": fmt_num(metrics.get("recall")),
                "f1": fmt_num(metrics.get("f1")),
                "roc_auc": fmt_num(metrics.get("roc_auc")),
                "pr_auc": fmt_num(metrics.get("pr_auc")),
                "mae": fmt_num(metrics.get("mae")),
                "rmse": fmt_num(metrics.get("rmse")),
                "r2": fmt_num(metrics.get("r2")),
                "train_s": fmt_num(run.get("training_time_seconds")),
                "infer_ms": fmt_num(metrics.get("inference_latency_ms")),
                "run_id": run.get("id"),
                "mlflow_run_id": run.get("mlflow_run_id"),
            }
        )
    return rows


def explanation_impacts(explanation: dict[str, Any]) -> list[dict[str, Any]]:
    """Return global-explanation feature impacts sorted by absolute impact."""
    top = explanation.get("top_features")
    features = list(top) if isinstance(top, list) else []
    rows = [
        {
            "feature": item.get("feature"),
            "value": fmt_num(item.get("value")),
            "impact": item.get("impact"),
        }
        for item in features
        if isinstance(item, dict)
    ]
    rows.sort(key=lambda row: abs(float(row["impact"] or 0.0)), reverse=True)
    return rows


def registered_model_rows(models: Any) -> list[dict[str, Any]]:
    """Flatten registered models into a table for the registry page."""
    rows: list[dict[str, Any]] = []
    for model in models or []:
        if not isinstance(model, dict):
            continue
        rows.append(
            {
                "id": model.get("id"),
                "name": model.get("name"),
                "version": model.get("version"),
                "deployment": model.get("deployment_status"),
                "alias": model.get("alias"),
                "experiment_id": model.get("experiment_id"),
                "model_run_id": model.get("model_run_id"),
                "uri": model.get("mlflow_model_uri"),
                "created_at": model.get("created_at"),
            }
        )
    return rows


# -- CSV parse / format (stdlib only; no filesystem) ---------------------------


def parse_csv_records(text: str) -> list[dict[str, Any]]:
    """Parse CSV text into a list of records (JSON-typed values).

    Raises ``ValueError`` with a readable message when the CSV is empty or
    malformed so the UI can surface a clear input error.
    """
    text = text.strip()
    if not text:
        raise ValueError("CSV is empty")
    try:
        reader = csv.DictReader(io.StringIO(text))
    except csv.Error as exc:
        raise ValueError(f"malformed CSV: {exc}") from exc
    if reader.fieldnames is None:
        raise ValueError("CSV has no header row")
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(reader):
        if raw is None:
            continue
        record: dict[str, Any] = {}
        for key, value in raw.items():
            record[key] = _coerce_scalar(value)
        records.append(record)
        if index >= 9999:
            raise ValueError("CSV exceeds the supported row count")
    if not records:
        raise ValueError("CSV contains no data rows")
    return records


def _coerce_scalar(value: Any) -> Any:
    """Best-effort coercion of a CSV cell to a JSON scalar."""
    if value is None:
        return None
    text = str(value)
    if text == "":
        return None
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." in text or "e" in lowered or "E" in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def records_to_csv(records: Any) -> str:
    """Render prediction results (a list of dicts) as CSV text for download."""
    rows = [r for r in (records or []) if isinstance(r, dict)]
    if not rows:
        return ""
    keys = list(rows[0].keys())
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _csv_scalar(row.get(k)) for k in keys})
    return output.getvalue()


def _csv_scalar(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return "" if value is None else value

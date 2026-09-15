"""Deterministic Scania APS benchmark runner.

This slice loads the normalised prepared train/test CSVs produced by the
``scania_aps`` adapter, validates that target + features are identical, and
reassembles the official UCI holdout split by concatenating the two frames
without reshuffling and calling ``official_split(train_rows, ...)``. It then runs
the frozen P0 vertical slices unchanged (profile -> task -> train -> recommend ->
cost) and serialises deterministic, JSON-safe artifacts under
``artifacts/benchmarks/scania_aps/`` plus a Markdown summary that reflects only
the actual computed results.

The runner never optimises a decision threshold: classifiers use the default 0.5
probability cut-off from ``sklearn`` ``predict``, and that fact is recorded
explicitly in every artifact. Cost numbers are the official IDA 2016 challenge
economics (FP=10, FN=500), not MaintAI production economics.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from maintai.benchmarks.scania_aps import (
    FN_COST,
    FP_COST,
    LABEL_ENCODING,
    N_FEATURES,
    PREPARED_DIR,
    PREPARED_TEST_FILENAME,
    PREPARED_TRAIN_FILENAME,
    SOURCE_METADATA,
    SUMMARY_FILENAME,
    TARGET_COLUMN,
    TEST_ROWS,
    TRAIN_ROWS,
    build_split_summary,
    official_split,
    project_root,
)
from maintai.data.profile import profile
from maintai.data.quality import assess
from maintai.data.schema import infer_schema
from maintai.ml import catalog
from maintai.ml.cost import compare_costs
from maintai.ml.recommend import recommend
from maintai.ml.schemas import CostAssumptions, TrainingPlan
from maintai.ml.train import train
from maintai.tasks.infer import recommend_task

SEED = 42
PRIMARY_METRIC = "pr_auc"
TASK = "binary_classification"
DEFAULT_THRESHOLD = 0.5
MINIMUM_RECALL = 0.0

THRESHOLD_NOTE = (
    "classifier predictions use the default 0.5 decision threshold on the "
    "positive-class probability; no threshold optimisation is performed"
)

ARTIFACTS_DIR = project_root() / "artifacts" / "benchmarks" / "scania_aps"
CONFIG_PATH = project_root() / "configs" / "benchmarks" / "scania_aps.yaml"

DATASET_SUMMARY_FILENAME = "dataset_summary.json"
MODEL_METRICS_FILENAME = "model_metrics.json"
MODEL_METRICS_CSV_FILENAME = "model_metrics.csv"
CONFUSION_MATRICES_FILENAME = "confusion_matrices.json"
COST_COMPARISON_FILENAME = "cost_comparison.json"
BENCHMARK_SUMMARY_FILENAME = "benchmark_summary.md"

CSV_COLUMNS = [
    "model_name",
    "status",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "tn",
    "fp",
    "fn",
    "tp",
    "threshold",
    "training_time_seconds",
    "inference_latency_ms",
    "primary_metric",
    "primary_metric_value",
]


class BenchmarkError(Exception):
    """Base class for benchmark runner errors."""


class BenchmarkConfigError(BenchmarkError):
    """Raised when the benchmark config is invalid or mismatches adapter constants."""


@dataclass(frozen=True)
class BenchmarkConfig:
    """Parsed benchmark configuration (strictly validated against adapter constants)."""

    target: str
    n_features: int
    train_rows: int
    test_rows: int
    fp_cost: float
    fn_cost: float
    currency_label: str
    threshold: float
    task: str
    model_names: tuple[str, ...]


def _config_node(mapping: dict, *keys: str):
    node: object = mapping
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            raise BenchmarkConfigError(
                f"config is missing required key {'.'.join(keys)!r}"
            )
        node = node[key]
    return node


def parse_benchmark_config(config_path: Path | str) -> BenchmarkConfig:
    """Load the benchmark YAML and reject any mismatch with adapter constants."""
    path = Path(config_path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BenchmarkConfigError(f"benchmark config not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise BenchmarkConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise BenchmarkConfigError(f"config root must be a mapping, got {type(raw).__name__}")

    target = _config_node(raw, "dataset", "target")
    n_features = _config_node(raw, "dataset", "n_features")
    train_rows = _config_node(raw, "dataset", "train_rows")
    test_rows = _config_node(raw, "dataset", "test_rows")
    fp_cost = _config_node(raw, "costs", "fp_cost")
    fn_cost = _config_node(raw, "costs", "fn_cost")
    currency_label = _config_node(raw, "costs", "currency_label")
    threshold = _config_node(raw, "threshold")
    task = _config_node(raw, "models", "task")
    model_names = _config_node(raw, "models", "catalog")

    errors: list[str] = []
    if _config_node(raw, "dataset", "name") != "scania_aps":
        errors.append("dataset.name must be 'scania_aps'")
    if target != TARGET_COLUMN:
        errors.append(f"dataset.target must be {TARGET_COLUMN!r}, got {target!r}")
    if int(n_features) != N_FEATURES:
        errors.append(f"dataset.n_features must be {N_FEATURES}, got {n_features!r}")
    if int(train_rows) != TRAIN_ROWS:
        errors.append(f"dataset.train_rows must be {TRAIN_ROWS}, got {train_rows!r}")
    if int(test_rows) != TEST_ROWS:
        errors.append(f"dataset.test_rows must be {TEST_ROWS}, got {test_rows!r}")
    if float(fp_cost) != FP_COST:
        errors.append(f"costs.fp_cost must be {FP_COST}, got {fp_cost!r}")
    if float(fn_cost) != FN_COST:
        errors.append(f"costs.fn_cost must be {FN_COST}, got {fn_cost!r}")
    if float(threshold) != DEFAULT_THRESHOLD:
        errors.append(f"threshold must be {DEFAULT_THRESHOLD}, got {threshold!r}")
    if task != TASK:
        errors.append(f"models.task must be {TASK!r}, got {task!r}")
    if list(model_names) != list(catalog.CLASSIFICATION_MODELS):
        errors.append(
            f"models.catalog must be {list(catalog.CLASSIFICATION_MODELS)!r}, "
            f"got {list(model_names)!r}"
        )
    if errors:
        raise BenchmarkConfigError(
            "config mismatch with adapter constants: " + "; ".join(errors)
        )

    return BenchmarkConfig(
        target=str(target),
        n_features=int(n_features),
        train_rows=int(train_rows),
        test_rows=int(test_rows),
        fp_cost=float(fp_cost),
        fn_cost=float(fn_cost),
        currency_label=str(currency_label),
        threshold=float(threshold),
        task=str(task),
        model_names=tuple(model_names),
    )


def _load_prepared_frame(path: Path, expected_rows: int) -> pd.DataFrame:
    if not path.is_file():
        raise BenchmarkError(f"prepared CSV not found: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - normalize parser failures
        raise BenchmarkError(f"failed to read {path.name!r}: {exc}") from exc
    if len(frame) != expected_rows:
        raise BenchmarkError(
            f"expected {expected_rows} rows in {path.name!r}, got {len(frame)}"
        )
    return frame


def _validate_columns(
    train_df: pd.DataFrame, test_df: pd.DataFrame, config: BenchmarkConfig
) -> None:
    for label, frame in (("train", train_df), ("test", test_df)):
        if TARGET_COLUMN not in frame.columns:
            raise BenchmarkError(f"prepared {label} is missing target column {TARGET_COLUMN!r}")
    train_cols = list(train_df.columns)
    test_cols = list(test_df.columns)
    if train_cols != test_cols:
        raise BenchmarkError("prepared train and test have different columns")
    if train_cols[0] != TARGET_COLUMN:
        raise BenchmarkError(
            f"target column must be first, got {train_cols[0]!r}"
        )
    if len(train_cols) != config.n_features + 1:
        raise BenchmarkError(
            f"expected {config.n_features + 1} columns (target + {config.n_features} "
            f"features), got {len(train_cols)}"
        )


def _json_safe(value):
    """Recursively coerce non-finite floats to None for strict JSON output."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _serialize_json(obj) -> bytes:
    return (
        json.dumps(_json_safe(obj), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = -1  # ownership moved to the file object
            handle.write(data)
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001 - clean up then re-raise
        if fd != -1:
            os.close(fd)
        tmp.unlink(missing_ok=True)
        raise


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _synthesize_dataset_summary(
    train_df: pd.DataFrame, test_df: pd.DataFrame, config: BenchmarkConfig
) -> dict:
    return {
        "dataset": "scania_aps",
        "source": SOURCE_METADATA,
        "target": TARGET_COLUMN,
        "label_encoding": LABEL_ENCODING,
        "n_features": config.n_features,
        "prepared_at": None,
        "splits": {
            "train": build_split_summary(train_df, _csv_bytes(train_df), PREPARED_TRAIN_FILENAME),
            "test": build_split_summary(test_df, _csv_bytes(test_df), PREPARED_TEST_FILENAME),
        },
    }


def _load_prepared_summary(prepared_dir: Path) -> dict | None:
    summary_path = prepared_dir / SUMMARY_FILENAME
    if not summary_path.is_file():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _relativize_path(path_str: str) -> str:
    try:
        return str(Path(path_str).relative_to(project_root()))
    except ValueError:
        return Path(path_str).name


def _enrich_dataset_summary(base: dict, config: BenchmarkConfig) -> dict:
    enriched = dict(base)
    splits = enriched.setdefault("splits", {})
    for key in ("train", "test"):
        entry = splits.get(key)
        if isinstance(entry, dict) and isinstance(entry.get("path"), str):
            entry["path"] = _relativize_path(entry["path"])
    enriched["benchmark"] = {
        "name": "scania_aps",
        "task": config.task,
        "primary_metric": PRIMARY_METRIC,
        "seed": SEED,
        "threshold": config.threshold,
        "threshold_optimized": False,
        "threshold_note": THRESHOLD_NOTE,
        "models": list(config.model_names),
        "official_split": {
            "train_rows": config.train_rows,
            "test_rows": config.test_rows,
        },
    }
    return enriched


def _confusion_counts(evaluation) -> tuple[int | None, int | None, int | None, int | None]:
    matrix = evaluation.confusion_matrix
    labels = list(evaluation.labels)
    if (
        matrix is None
        or len(labels) != 2
        or len(matrix) != 2
        or any(len(row) != 2 for row in matrix)
    ):
        return None, None, None, None
    positive = evaluation.positive_label if evaluation.positive_label is not None else labels[1]
    if positive not in labels:
        return None, None, None, None
    pos_idx = labels.index(positive)
    neg_idx = 1 - pos_idx
    return (
        int(matrix[neg_idx][neg_idx]),
        int(matrix[neg_idx][pos_idx]),
        int(matrix[pos_idx][neg_idx]),
        int(matrix[pos_idx][pos_idx]),
    )


def _model_records(result, config: BenchmarkConfig) -> list[dict]:
    primary = result.primary_metric
    records: list[dict] = []
    for evaluation in result.evaluations:
        tn, fp, fn, tp = _confusion_counts(evaluation)
        metrics = evaluation.metrics or {}
        records.append(
            {
                "model_name": evaluation.model_name,
                "task": evaluation.task,
                "status": evaluation.status,
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "threshold": config.threshold,
                "training_time_seconds": evaluation.training_time_seconds,
                "inference_latency_ms": evaluation.inference_latency_ms,
                "primary_metric": primary,
                "primary_metric_value": metrics.get(primary) if primary else None,
            }
        )
    return records


def _confusion_payload(result) -> dict:
    models: dict[str, dict] = {}
    for evaluation in result.evaluations:
        tn, fp, fn, tp = _confusion_counts(evaluation)
        models[evaluation.model_name] = {
            "status": evaluation.status,
            "labels": list(evaluation.labels),
            "positive_label": evaluation.positive_label,
            "matrix": evaluation.confusion_matrix,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
    return {"models": models}


def _fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _render_benchmark_markdown(
    config: BenchmarkConfig,
    profile_report,
    quality_report,
    task,
    result,
    comparison,
    cost,
    records: list[dict],
) -> str:
    lines: list[str] = []
    lines.append("# Scania APS benchmark — official holdout")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- target: `{config.target}`")
    lines.append(f"- task: `{config.task}`")
    lines.append(f"- primary metric: `{result.primary_metric}`")
    lines.append(f"- seed: {SEED}")
    lines.append(
        f"- decision threshold: {config.threshold} (default; no threshold optimisation)"
    )
    lines.append(f"- models: {', '.join(config.model_names)}")
    lines.append(
        f"- official split: {config.train_rows} train / {config.test_rows} test rows"
    )
    lines.append("")
    lines.append("## Data (official training split only)")
    lines.append(f"- rows: {profile_report.row_count}")
    lines.append(f"- columns: {profile_report.column_count}")
    lines.append(f"- recommended task: {task.recommended_task} (confidence {task.confidence:.2f})")
    lines.append(f"- data quality score: {quality_report.score:.2f} / {quality_report.max_score}")
    lines.append("")
    lines.append("## Models (official test split)")
    lines.append("")
    lines.append(
        "| model | status | precision | recall | f1 | roc_auc | pr_auc | "
        "threshold | train_s | infer_ms |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for record in records:
        lines.append(
            f"| {record['model_name']} | {record['status']} | "
            f"{_fmt(record['precision'])} | {_fmt(record['recall'])} | "
            f"{_fmt(record['f1'])} | {_fmt(record['roc_auc'])} | "
            f"{_fmt(record['pr_auc'])} | {record['threshold']} | "
            f"{_fmt(record['training_time_seconds'])} | "
            f"{_fmt(record['inference_latency_ms'])} |"
        )
    lines.append("")
    lines.append("## Confusion counts (TN / FP / FN / TP, official test split)")
    for record in records:
        lines.append(
            f"- {record['model_name']}: TN={record['tn']} FP={record['fp']} "
            f"FN={record['fn']} TP={record['tp']}"
        )
    lines.append("")
    lines.append("## Selection")
    if comparison.best_model is not None:
        best = next(r for r in records if r["model_name"] == comparison.best_model)
        lines.append(
            f"- metric-best ({result.primary_metric}): {comparison.best_model} "
            f"({_fmt(best['primary_metric_value'])})"
        )
    else:
        lines.append(f"- metric-best ({result.primary_metric}): none")
    cost_best = next((r for r in cost.rows if r.model_name == cost.cost_best), None)
    if cost_best is not None and cost_best.expected_error_cost is not None:
        lines.append(
            f"- cost-best (FN*{config.fn_cost:g} + FP*{config.fp_cost:g}): "
            f"{cost_best.model_name} ({_fmt(cost_best.expected_error_cost)})"
        )
    else:
        lines.append("- cost-best: none")
    lines.append("")
    lines.append("## Disclaimers")
    lines.append(f"- {THRESHOLD_NOTE}.")
    lines.append(
        f"- Cost assumptions are the official IDA 2016 challenge economics "
        f"(FP={config.fp_cost:g}, FN={config.fn_cost:g}, {config.currency_label}), "
        "not MaintAI production economics."
    )
    return "\n".join(lines) + "\n"


def run_benchmark(
    *,
    prepared_dir: Path | str = PREPARED_DIR,
    artifacts_dir: Path | str = ARTIFACTS_DIR,
    config: BenchmarkConfig | None = None,
    config_path: Path | str = CONFIG_PATH,
) -> dict:
    """Run the full Scania APS benchmark slice and write deterministic artifacts."""
    if config is None:
        config = parse_benchmark_config(config_path)
    prepared_dir = Path(prepared_dir)
    artifacts_dir = Path(artifacts_dir)

    train_df = _load_prepared_frame(
        prepared_dir / PREPARED_TRAIN_FILENAME, config.train_rows
    )
    test_df = _load_prepared_frame(
        prepared_dir / PREPARED_TEST_FILENAME, config.test_rows
    )
    _validate_columns(train_df, test_df, config)

    # Concatenate without reshuffling, then use the official UCI holdout split.
    frame = pd.concat([train_df, test_df], ignore_index=True)
    split = official_split(frame, train_rows=config.train_rows)
    if len(split.test_indices) != config.test_rows:
        raise BenchmarkError(
            f"official split produced {len(split.test_indices)} test rows, "
            f"expected {config.test_rows}"
        )

    # Profiling / task framing / quality run on the official training split only.
    train_frame = frame.iloc[split.train_indices]
    schema = infer_schema(train_frame)
    profile_report = profile(train_frame, schema)
    quality_report = assess(train_frame, schema, target_col=config.target)
    task = recommend_task(train_frame, target_col=config.target)

    plan = TrainingPlan(
        target=config.target,
        task=config.task,
        split=split,
        model_names=list(config.model_names),
        primary_metric=PRIMARY_METRIC,
        seed=SEED,
    )
    output = train(frame, plan)
    result = output.result

    comparison = recommend(result, minimum_recall=MINIMUM_RECALL)
    cost = compare_costs(
        result,
        CostAssumptions(
            fn_cost=config.fn_cost,
            fp_cost=config.fp_cost,
            currency_label=config.currency_label,
        ),
        minimum_recall=MINIMUM_RECALL,
    )

    base_summary = _load_prepared_summary(prepared_dir) or _synthesize_dataset_summary(
        train_df, test_df, config
    )
    dataset_summary = _enrich_dataset_summary(base_summary, config)
    records = _model_records(result, config)
    model_metrics = {
        "dataset": "scania_aps",
        "target": config.target,
        "task": config.task,
        "primary_metric": result.primary_metric,
        "seed": SEED,
        "threshold": config.threshold,
        "threshold_optimized": False,
        "threshold_note": THRESHOLD_NOTE,
        "recommended_model": comparison.best_model,
        "models": records,
    }
    confusion_matrices = _confusion_payload(result)
    cost_payload = cost.model_dump(mode="json")
    cost_payload["notes"] = [
        (
            "costs use the official IDA 2016 challenge economics, not MaintAI "
            "production maintenance economics"
            if note == "costs are demo assumptions, not actual maintenance economics"
            else note
        )
        for note in cost_payload.get("notes", [])
    ]
    markdown = _render_benchmark_markdown(
        config, profile_report, quality_report, task, result, comparison, cost, records
    )

    metrics_frame = pd.DataFrame(records, columns=CSV_COLUMNS)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        artifacts_dir / DATASET_SUMMARY_FILENAME, _serialize_json(dataset_summary)
    )
    _atomic_write(
        artifacts_dir / MODEL_METRICS_FILENAME, _serialize_json(model_metrics)
    )
    _atomic_write(
        artifacts_dir / MODEL_METRICS_CSV_FILENAME,
        metrics_frame.to_csv(index=False, lineterminator="\n").encode("utf-8"),
    )
    _atomic_write(
        artifacts_dir / CONFUSION_MATRICES_FILENAME, _serialize_json(confusion_matrices)
    )
    _atomic_write(
        artifacts_dir / COST_COMPARISON_FILENAME, _serialize_json(cost_payload)
    )
    _atomic_write(artifacts_dir / BENCHMARK_SUMMARY_FILENAME, markdown.encode("utf-8"))

    return {
        "dataset": "scania_aps",
        "target": config.target,
        "task": config.task,
        "primary_metric": result.primary_metric,
        "seed": SEED,
        "threshold": config.threshold,
        "threshold_optimized": False,
        "official_split": {
            "train_rows": config.train_rows,
            "test_rows": config.test_rows,
        },
        "task_recommendation": task.recommended_task,
        "task_confidence": task.confidence,
        "quality_score": quality_report.score,
        "recommended_model": comparison.best_model,
        "metric_best_model": cost.metric_best,
        "cost_best_model": cost.cost_best,
        "models": [
            {
                "model_name": record["model_name"],
                "status": record["status"],
                "primary_metric_value": record["primary_metric_value"],
            }
            for record in records
        ],
        "artifacts": {
            "dir": str(artifacts_dir),
            DATASET_SUMMARY_FILENAME: str(artifacts_dir / DATASET_SUMMARY_FILENAME),
            MODEL_METRICS_FILENAME: str(artifacts_dir / MODEL_METRICS_FILENAME),
            MODEL_METRICS_CSV_FILENAME: str(artifacts_dir / MODEL_METRICS_CSV_FILENAME),
            CONFUSION_MATRICES_FILENAME: str(
                artifacts_dir / CONFUSION_MATRICES_FILENAME
            ),
            COST_COMPARISON_FILENAME: str(artifacts_dir / COST_COMPARISON_FILENAME),
            BENCHMARK_SUMMARY_FILENAME: str(
                artifacts_dir / BENCHMARK_SUMMARY_FILENAME
            ),
        },
    }

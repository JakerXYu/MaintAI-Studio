"""Deterministic target-leakage detection.

Flags features that leak the target (exact duplicates, near-identical columns,
label-derived names, post-event names, near one-to-one IDs). ``block`` features
are excluded from training by default; overriding requires an audit record.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from maintai.data.schema import SchemaInference, has_name_hint
from maintai.data.schemas import LeakageIssue, LeakageReport

_ID_HINTS: tuple[str, ...] = (
    "id",
    "uid",
    "udi",
    "uuid",
    "serial",
    "asset",
    "machine",
    "equipment",
    "device",
    "unit",
    "part",
    "component",
    "tag",
    "code",
    "plate",
    "vin",
    "index",
    "no",
    "number",
)
_POST_EVENT_HINTS: tuple[str, ...] = (
    "failure",
    "fault",
    "repair",
    "downtime",
    "maintenance",
    "breakdown",
    "error",
    "alarm",
    "incident",
    "workorder",
)
def _name_tokens(name: str) -> set[str]:
    return {token for token in pd.Series([name]).str.lower().str.findall(r"[a-z0-9]+")[0]}


def _is_label_derived_name(name: str) -> bool:
    tokens = _name_tokens(name)
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    if tokens & {"label", "target", "groundtruth"}:
        return True
    return normalized in {"class", "predictedclass", "outputclass", "groundtruth"}


class LeakageError(Exception):
    """Raised when leakage detection cannot run (e.g. missing target column)."""


def _agreement(series: pd.Series, target: pd.Series) -> float:
    both = series.notna() & target.notna()
    if not bool(both.any()):
        return 0.0
    equal = series[both].astype(str) == target[both].astype(str)
    return float(equal.mean())


def _numeric_correlation(series: pd.Series, target: pd.Series) -> float | None:
    both = series.notna() & target.notna()
    if int(both.sum()) < 2:
        return None
    x = series[both].astype(float)
    y = target[both].astype(float)
    if x.nunique() < 2 or y.nunique() < 2:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _check_feature(
    name: str,
    series: pd.Series,
    target: pd.Series,
    schema: SchemaInference | None,
) -> LeakageIssue | None:
    sem_map = {c.name: c.semantic_type for c in schema.columns} if schema else {}
    semantic = sem_map.get(name)

    agreement = _agreement(series, target)
    if agreement >= 0.999:
        return LeakageIssue(
            feature=name,
            kind="target_duplicate",
            severity="block",
            reason=f"feature is identical to the target ({agreement:.1%} agreement)",
            evidence={"agreement": agreement},
        )

    if pd.api.types.is_numeric_dtype(series.dtype) and pd.api.types.is_numeric_dtype(
        target.dtype
    ):
        corr = _numeric_correlation(series, target)
        if corr is not None and abs(corr) >= 0.99:
            return LeakageIssue(
                feature=name,
                kind="target_correlated",
                severity="block",
                reason=f"feature is almost perfectly correlated with the target (r={corr:.3f})",
                evidence={"correlation": corr},
            )

    if agreement >= 0.90:
        return LeakageIssue(
            feature=name,
            kind="highly_identical",
            severity="warning",
            reason=f"feature is highly identical to the target ({agreement:.1%} agreement)",
            evidence={"agreement": agreement},
        )

    if _is_label_derived_name(name):
        return LeakageIssue(
            feature=name,
            kind="label_derived_name",
            severity="block",
            reason="feature name suggests a label/target-derived column",
        )

    if has_name_hint(name, _POST_EVENT_HINTS):
        return LeakageIssue(
            feature=name,
            kind="post_event_name",
            severity="warning",
            reason="feature name suggests a post-event / outcome column",
        )

    non_null = int(series.notna().sum())
    nunique = int(series.nunique(dropna=True))
    unique_ratio = nunique / non_null if non_null else 0.0
    if semantic == "identifier_like" or (
        unique_ratio >= 0.99 and has_name_hint(name, _ID_HINTS)
    ):
        return LeakageIssue(
            feature=name,
            kind="near_one_to_one_id",
            severity="block",
            reason=f"identifier-like column with {unique_ratio:.1%} unique ratio",
            evidence={"unique_ratio": unique_ratio},
        )

    if unique_ratio >= 0.99 and not pd.api.types.is_numeric_dtype(series.dtype):
        return LeakageIssue(
            feature=name,
            kind="high_cardinality",
            severity="warning",
            reason=f"high-cardinality column ({unique_ratio:.1%} unique)",
            evidence={"unique_ratio": unique_ratio},
        )

    return None


def detect(
    frame: pd.DataFrame,
    target_col: str,
    timestamp_col: str | None = None,
    asset_id_col: str | None = None,
    schema: SchemaInference | None = None,
) -> LeakageReport:
    """Detect target-leakage features and return a safe/warning/block verdict."""
    if target_col not in frame.columns:
        raise LeakageError(f"target column {target_col!r} not found in the dataset")
    target = frame[target_col]
    issues: list[LeakageIssue] = []
    for name in frame.columns:
        if name in (target_col, asset_id_col, timestamp_col):
            continue
        issue = _check_feature(name, frame[name], target, schema)
        if issue is not None:
            issues.append(issue)

    if timestamp_col and timestamp_col in frame.columns:
        for name in frame.columns:
            if name in (timestamp_col, target_col):
                continue
            if pd.api.types.is_datetime64_any_dtype(frame[name].dtype):
                issues.append(
                    LeakageIssue(
                        feature=name,
                        kind="extra_datetime",
                        severity="warning",
                        reason="datetime feature besides the known timestamp may encode an "
                        "event time",
                    )
                )

    blocked = [issue.feature for issue in issues if issue.severity == "block"]
    verdict: str = "block" if blocked else ("warning" if issues else "safe")
    return LeakageReport(
        verdict=verdict,
        issues=issues,
        excluded_features=blocked,
    )

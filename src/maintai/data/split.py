"""Deterministic, leakage-aware data split.

Strategy priority (spec §9.6):
1. asset ID + timestamp -> per-asset chronological holdout (no future leakage)
2. asset ID only       -> group split (no asset appears in train and test)
3. timestamp only      -> chronological split
4. neither             -> classification: stratified; regression: fixed-seed random
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel

from maintai.data.schemas import SplitResult

CLASSIFICATION_TASKS = ("binary_classification", "multiclass_classification")


class SplitError(Exception):
    """Raised when a deterministic split cannot be produced."""


class SplitConfig(BaseModel):
    """Demo-default split sizes (mirrors configs/default.yaml)."""

    test_size: float = 0.20
    validation_size: float = 0.20
    random_state: int = 42


def _validate_columns(
    frame: pd.DataFrame,
    target_col: str | None,
    asset_id_col: str | None,
    timestamp_col: str | None,
) -> None:
    for column, label in (
        (asset_id_col, "asset_id_col"),
        (timestamp_col, "timestamp_col"),
        (target_col, "target_col"),
    ):
        if column is not None and column not in frame.columns:
            raise SplitError(f"{label} {column!r} not found in the dataset")


def _select_strategy(
    asset_id_col: str | None,
    timestamp_col: str | None,
    task_type: str | None,
    target_col: str | None,
) -> str:
    if asset_id_col and timestamp_col:
        return "group_time"
    if asset_id_col:
        return "group"
    if timestamp_col:
        return "chronological"
    if task_type in CLASSIFICATION_TASKS and target_col:
        return "stratified"
    return "random"


def _timestamps(frame: pd.DataFrame, timestamp_col: str) -> np.ndarray:
    ts = pd.to_datetime(frame[timestamp_col], errors="coerce").to_numpy(dtype="datetime64[ns]")
    if np.isnat(ts).any():
        raise SplitError("timestamp column contains unparseable or missing values")
    return ts


def _group_time(
    frame: pd.DataFrame,
    test_size: float,
    asset_id_col: str,
    timestamp_col: str,
    scope: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scope_idx = np.flatnonzero(scope)
    ts = _timestamps(frame, timestamp_col)
    assets = frame[asset_id_col].to_numpy()
    if pd.isna(assets).any():
        raise SplitError("asset_id column must not contain missing values")
    ts_sub = ts[scope_idx]
    perm = np.argsort(ts_sub, kind="stable")
    sorted_pos = scope_idx[perm]
    sorted_assets = assets[sorted_pos]
    train_pos: list[np.ndarray] = []
    test_pos: list[np.ndarray] = []
    multi_record = False
    for group in np.unique(sorted_assets):
        positions = np.flatnonzero(sorted_assets == group)
        if len(positions) >= 2:
            multi_record = True
            k = max(1, int(np.ceil(len(positions) * test_size)))
            test_pos.append(sorted_pos[positions[-k:]])
            train_pos.append(sorted_pos[positions[:-k]])
        else:
            train_pos.append(sorted_pos[positions])
    if not multi_record:
        k = max(1, int(np.ceil(len(sorted_pos) * test_size)))
        return sorted_pos[:-k], sorted_pos[-k:]
    return np.concatenate(train_pos), np.concatenate(test_pos)


def _group(
    frame: pd.DataFrame,
    test_size: float,
    random_state: int,
    asset_id_col: str,
    scope: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scope_idx = np.flatnonzero(scope)
    assets = frame[asset_id_col].to_numpy()[scope_idx]
    if pd.isna(assets).any():
        raise SplitError("asset_id column must not contain missing values")
    uniq = np.unique(assets)
    if len(uniq) < 2:
        raise SplitError("group split requires at least 2 distinct assets")
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(uniq)
    n_test = max(1, int(np.ceil(len(uniq) * test_size)))
    test_groups = perm[:n_test]
    test_mask = np.isin(assets, test_groups)
    return scope_idx[~test_mask], scope_idx[test_mask]


def _chronological(
    frame: pd.DataFrame,
    test_size: float,
    timestamp_col: str,
    scope: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scope_idx = np.flatnonzero(scope)
    ts = _timestamps(frame, timestamp_col)
    perm = np.argsort(ts[scope_idx], kind="stable")
    sorted_pos = scope_idx[perm]
    k = max(1, int(np.ceil(len(sorted_pos) * test_size)))
    return sorted_pos[:-k], sorted_pos[-k:]


def _stratified(
    frame: pd.DataFrame,
    test_size: float,
    random_state: int,
    target_col: str,
    scope: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    scope_idx = np.flatnonzero(scope)
    stratify = frame[target_col].to_numpy()[scope_idx]
    if pd.isna(stratify).any():
        raise SplitError("cannot stratify: target contains missing values")
    counts = pd.Series(stratify).value_counts()
    if len(counts) < 2:
        raise SplitError("cannot stratify: target has a single class")
    if int(counts.min()) < 2:
        raise SplitError("cannot stratify: a target class has fewer than 2 samples")
    try:
        train, test = train_test_split(
            scope_idx,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError as exc:
        raise SplitError(f"stratified split failed: {exc}") from exc
    return train, test


def _random(
    frame: pd.DataFrame,
    test_size: float,
    random_state: int,
    scope: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    scope_idx = np.flatnonzero(scope)
    return train_test_split(
        scope_idx,
        test_size=test_size,
        random_state=random_state,
    )


def _evidence_for(
    strategy: str,
    asset_id_col: str | None,
    timestamp_col: str | None,
    target_col: str | None,
    task_type: str | None,
) -> list[str]:
    if strategy == "group_time":
        return [
            "asset+timestamp split: within each asset the latest records are held out "
            "as test, so no future record leaks into training",
            f"group key {asset_id_col!r}, ordering key {timestamp_col!r}",
        ]
    if strategy == "group":
        return [
            "group split: assets are partitioned so no asset appears in both train and test",
            f"group key {asset_id_col!r}",
        ]
    if strategy == "chronological":
        return [
            f"chronological split: rows ordered by {timestamp_col!r}, latest held out as test"
        ]
    if strategy == "stratified":
        return [f"stratified split: class proportions preserved via {target_col!r}"]
    return [f"fixed-seed random split (task={task_type})"]


def split(
    frame: pd.DataFrame,
    target_col: str | None = None,
    asset_id_col: str | None = None,
    timestamp_col: str | None = None,
    task_type: str | None = None,
    config: SplitConfig | None = None,
) -> SplitResult:
    """Return train/test (and optional validation) indices with strategy evidence."""
    config = config or SplitConfig()
    if len(frame) < 4:
        raise SplitError("cannot split: fewer than 4 rows")
    if not 0.0 < config.test_size < 1.0:
        raise SplitError("test_size must be in (0, 1)")
    if config.validation_size and not 0.0 < config.validation_size < 1.0:
        raise SplitError("validation_size must be in (0, 1)")
    if config.test_size + config.validation_size >= 1.0:
        raise SplitError("test_size + validation_size must be less than 1")
    _validate_columns(frame, target_col, asset_id_col, timestamp_col)

    strategy = _select_strategy(asset_id_col, timestamp_col, task_type, target_col)
    scope = np.ones(len(frame), dtype=bool)
    train_idx, test_idx = _apply_strategy(
        frame, strategy, config.test_size, config.random_state, target_col,
        asset_id_col, timestamp_col, task_type, scope,
    )
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise SplitError("split produced an empty train or test fold")

    validation_idx = np.array([], dtype=int)
    validation_note: str | None = None
    if config.validation_size > 0 and len(train_idx) >= 4:
        val_frac = config.validation_size / (1.0 - config.test_size)
        if 0.0 < val_frac < 1.0:
            train_scope = np.zeros(len(frame), dtype=bool)
            train_scope[train_idx] = True
            try:
                train_idx, validation_idx = _apply_strategy(
                    frame, strategy, val_frac, config.random_state, target_col,
                    asset_id_col, timestamp_col, task_type, train_scope,
                )
            except SplitError as exc:
                # Not enough structure left in the train fold to further split.
                validation_idx = np.array([], dtype=int)
                validation_note = f"validation omitted: {exc}"

    evidence = _evidence_for(
        strategy,
        asset_id_col,
        timestamp_col,
        target_col,
        task_type,
    )
    if validation_note:
        evidence.append(validation_note)

    return SplitResult(
        strategy=strategy,
        train_indices=sorted(int(i) for i in train_idx),
        test_indices=sorted(int(i) for i in test_idx),
        validation_indices=sorted(int(i) for i in validation_idx),
        evidence=evidence,
        params={
            "test_size": config.test_size,
            "validation_size": config.validation_size,
            "random_state": config.random_state,
        },
    )


def _apply_strategy(
    frame: pd.DataFrame,
    strategy: str,
    test_size: float,
    random_state: int,
    target_col: str | None,
    asset_id_col: str | None,
    timestamp_col: str | None,
    task_type: str | None,
    scope: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if strategy == "group_time":
        return _group_time(frame, test_size, asset_id_col, timestamp_col, scope)
    if strategy == "group":
        return _group(frame, test_size, random_state, asset_id_col, scope)
    if strategy == "chronological":
        return _chronological(frame, test_size, timestamp_col, scope)
    if strategy == "stratified":
        return _stratified(frame, test_size, random_state, target_col, scope)
    return _random(frame, test_size, random_state, scope)

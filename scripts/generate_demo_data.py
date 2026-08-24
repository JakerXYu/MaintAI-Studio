"""Generate a deterministic AI4I-style synthetic predictive-maintenance CSV.

The generated file mirrors the *shape* of the public AI4I 2020 Predictive
Maintenance dataset (UCI Machine Learning Repository) — a compact 10-column
sensor/telemetry analogue with similar product-type vocabulary — but every value is
synthesised here from a fixed seed (42). No UCI row is copied and no real asset
or private data is included. See ``data/README.md`` for provenance, privacy and
licence notes.

Because the seed and the generation algorithm are fixed, repeated runs produce
byte-identical output. The last stdout line is a single-line JSON summary::

    {"path": "...", "sha256": "...", "rows": 401}

Usage (from the repo root)::

    python scripts/generate_demo_data.py
    python scripts/generate_demo_data.py --output /tmp/other.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUTPUT = "data/synthetic/maintai_ai4i_style_demo.csv"
SEED = 42

COLUMNS: tuple[str, ...] = (
    "timestamp",
    "machine_id",
    "serial_no",
    "type",
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "machine_failure",
)

_N_MACHINES = 20
_RECORDS_PER_MACHINE = 20
_TARGET_FAILURE_RATE = 0.20

_NUMERIC_COLUMNS: tuple[str, ...] = (
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
)

# The exact anomaly assets/columns are fixed so the demo deterministically
# demonstrates a flatline, outliers, missing values and a duplicate row.
_FLATLINE_ASSET = "M0001"
_FLATLINE_COLUMN = "rotational_speed"
_FLATLINE_VALUE = 2200.0
_OUTLIER_ASSET = "M0002"
_OUTLIER_COLUMN = "torque"
_OUTLIER_VALUE = 250.0
_MISSING_COLUMNS = ("air_temperature", "process_temperature", "tool_wear")
_N_MISSING_CELLS = 15
_DUPLICATE_ROW = 0


def generate_dataframe(seed: int = SEED) -> pd.DataFrame:
    """Return the deterministic synthetic dataframe (401 rows, 10 columns)."""
    rng = np.random.default_rng(seed)
    n = _N_MACHINES * _RECORDS_PER_MACHINE

    machine_ids = [f"M{i:04d}" for i in range(_N_MACHINES)]
    machine = np.repeat(np.asarray(machine_ids, dtype=object), _RECORDS_PER_MACHINE)
    serials = [f"SN-{i:06d}" for i in range(1, n + 1)]

    # Product type is fixed per asset (L/M/H), mirroring AI4I's product quality.
    types = np.repeat(
        np.asarray([["L", "M", "H"][i % 3] for i in range(_N_MACHINES)], dtype=object),
        _RECORDS_PER_MACHINE,
    )

    # Per-asset contiguous hourly timestamps (each asset is its own time series).
    base = pd.Timestamp("2025-01-01 00:00:00")
    hour_offsets = (
        np.repeat(np.arange(_N_MACHINES) * _RECORDS_PER_MACHINE, _RECORDS_PER_MACHINE)
        + np.tile(np.arange(_RECORDS_PER_MACHINE), _N_MACHINES)
    )
    timestamps = [
        (base + pd.Timedelta(hours=int(h))).strftime("%Y-%m-%d %H:%M:%S")
        for h in hour_offsets
    ]

    # Sensors in roughly AI4I ranges (K / K / rpm / Nm / min).
    air_temperature = 299.0 + 2.0 * rng.normal(0.0, 1.0, n)
    process_temperature = air_temperature + 9.0 + 1.5 * rng.normal(0.0, 1.0, n)
    rotational_speed = np.clip(1800.0 + 350.0 * rng.normal(0.0, 1.0, n), 1200.0, 2900.0)
    torque = np.clip(40.0 + 25.0 * rng.normal(0.0, 1.0, n), 5.0, 85.0)
    tool_wear = np.clip(
        np.tile(np.arange(_RECORDS_PER_MACHINE), _N_MACHINES) * 12.0
        + rng.normal(0.0, 8.0, n),
        0.0,
        250.0,
    )

    # Deterministic anomalies.
    rotational_speed = rotational_speed.copy()
    torque = torque.copy()
    rotational_speed[machine == _FLATLINE_ASSET] = _FLATLINE_VALUE
    torque[np.flatnonzero(machine == _OUTLIER_ASSET)[:2]] = _OUTLIER_VALUE

    # Failure signal: a learnable health-risk score over sensor deviations plus
    # noise, labelled as the top ~20% so the signal is strong but spread evenly
    # over time (which keeps the per-asset chronological holdout balanced).
    risk = (
        0.12 * (air_temperature - 299.0) / 3.0
        + 0.12 * (process_temperature - 309.0) / 3.0
        + 0.25 * np.abs(rotational_speed - 1500.0) / 400.0
        + 0.25 * (torque - 40.0) / 30.0
        + 0.10 * (tool_wear / 200.0)
        + rng.normal(0.0, 0.25, n)
    )
    threshold = float(np.quantile(risk, 1.0 - _TARGET_FAILURE_RATE))
    machine_failure = (risk >= threshold).astype(int)

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "machine_id": machine,
            "serial_no": serials,
            "type": types,
            "air_temperature": air_temperature,
            "process_temperature": process_temperature,
            "rotational_speed": rotational_speed,
            "torque": torque,
            "tool_wear": tool_wear,
            "machine_failure": machine_failure,
        }
    )

    # A small amount of missing sensor data (never the target column).
    missing_rows = rng.integers(0, n, size=_N_MISSING_CELLS)
    missing_cols = rng.choice(np.asarray(_MISSING_COLUMNS), size=_N_MISSING_CELLS)
    for row, col in zip(missing_rows, missing_cols, strict=True):
        frame.loc[int(row), str(col)] = np.nan

    # One exact duplicate row (a real-world asset-log artifact).
    duplicate = frame.iloc[[_DUPLICATE_ROW]]
    frame = pd.concat([frame, duplicate], ignore_index=True)

    for column in _NUMERIC_COLUMNS:
        frame[column] = frame[column].round(6)
    frame["machine_failure"] = frame["machine_failure"].astype(int)

    return frame[list(COLUMNS)]


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize the frame to UTF-8 CSV bytes (LF line endings, no index)."""
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _write_output(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"output CSV path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args(argv)

    frame = generate_dataframe(SEED)
    data = to_csv_bytes(frame)
    output = Path(args.output)
    _write_output(output, data)

    summary = {
        "path": str(output.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "rows": int(len(frame)),
    }
    # Last stdout line: the single-line JSON summary.
    print(json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

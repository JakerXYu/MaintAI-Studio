"""Prepare the Scania APS dataset into normalised train/test CSVs.

Parses the raw UCI CSVs (train + test), normalises the target to the integer
``aps_failure`` column (neg=0, pos=1), and writes two CSV files plus a
``dataset_summary.json`` under ``data/raw/scania_aps/prepared/`` (git-ignored).

The raw CSVs must already exist (run ``scripts/download_scania_aps.py`` first,
or place ``aps_failure_training_set.csv`` / ``aps_failure_test_set.csv`` in the
raw directory manually). No training or evaluation is performed here.

Usage (from the repo root)::

    python scripts/prepare_scania_aps.py
    python scripts/prepare_scania_aps.py --raw-dir data/raw/scania_aps
"""

# isort: skip_file
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# -- bootstrap: allow running from the repo root without an installed package --
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from maintai.benchmarks.scania_aps import (  # noqa: E402
    PREPARED_DIR,
    RAW_DIR,
    TEST_MEMBER,
    TEST_ROWS,
    TRAIN_MEMBER,
    TRAIN_ROWS,
    ScaniaApsError,
    read_scania_aps_csv,
    write_prepared_dataset,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw-dir",
        default=str(RAW_DIR),
        help=f"directory holding the raw CSVs (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--prepared-dir",
        default=str(PREPARED_DIR),
        help=f"output directory for the prepared files (default: {PREPARED_DIR})",
    )
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    train_path = raw_dir / TRAIN_MEMBER
    test_path = raw_dir / TEST_MEMBER
    for path, name in ((train_path, TRAIN_MEMBER), (test_path, TEST_MEMBER)):
        if not path.is_file():
            print(
                f"ERROR: missing {name!r} in {raw_dir}. Run "
                "scripts/download_scania_aps.py first, or place the file manually.",
                file=sys.stderr,
            )
            return 1

    try:
        train = read_scania_aps_csv(train_path, expected_rows=TRAIN_ROWS)
        test = read_scania_aps_csv(test_path, expected_rows=TEST_ROWS)
        summary = write_prepared_dataset(train, test, Path(args.prepared_dir))
    except ScaniaApsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Last stdout line: the single-line JSON summary.
    print(json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

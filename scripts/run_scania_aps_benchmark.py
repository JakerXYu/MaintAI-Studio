"""Run the Scania APS benchmark slice end-to-end.

Loads the normalised prepared train/test CSVs (written by
``scripts/prepare_scania_aps.py``), validates the official split, trains the P0
model catalog on the official holdout, computes cost-aware comparison with the
official IDA 2016 economics (FP=10, FN=500), and writes deterministic artifacts
under ``artifacts/benchmarks/scania_aps/``.

No threshold optimisation is performed: classifiers use the default 0.5
probability cut-off, which is recorded in every artifact. No MLflow and no API
are required.

Usage (from the repo root)::

    python scripts/run_scania_aps_benchmark.py
    python scripts/run_scania_aps_benchmark.py --prepared-dir data/raw/scania_aps/prepared
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

from maintai.benchmarks.runner import (  # noqa: E402
    ARTIFACTS_DIR,
    CONFIG_PATH,
    run_benchmark,
)
from maintai.benchmarks.scania_aps import PREPARED_DIR  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--prepared-dir",
        default=str(PREPARED_DIR),
        help=f"directory holding the prepared CSVs (default: {PREPARED_DIR})",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=str(ARTIFACTS_DIR),
        help=f"output directory for benchmark artifacts (default: {ARTIFACTS_DIR})",
    )
    parser.add_argument(
        "--config",
        default=str(CONFIG_PATH),
        help=f"benchmark YAML config (default: {CONFIG_PATH})",
    )
    args = parser.parse_args(argv)

    try:
        summary = run_benchmark(
            prepared_dir=Path(args.prepared_dir),
            artifacts_dir=Path(args.artifacts_dir),
            config_path=Path(args.config),
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary: clean exit code
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Last stdout line: the single-line JSON summary.
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

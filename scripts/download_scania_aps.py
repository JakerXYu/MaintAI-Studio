"""Download and extract the official Scania APS archive (stdlib only).

Downloads the UCI archive (dataset 421) to ``data/raw/scania_aps/`` with an
atomic temp-file write, then extracts exactly the two CSVs (train + test). The
description text file is left in the archive. No training or evaluation.

On download failure a clear manual-download fallback is printed: fetch the
archive from the UCI URL yourself, place it at the destination, and re-run.

Usage (from the repo root)::

    python scripts/download_scania_aps.py
    python scripts/download_scania_aps.py --dest data/raw/scania_aps
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
    RAW_DIR,
    SCANIA_APS_URL,
    ScaniaApsDownloadError,
    ScaniaApsError,
    UnsafeZipMemberError,
    download_zip,
    extract_zip_members,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--url",
        default=SCANIA_APS_URL,
        help="archive URL to download (default: the official UCI URL)",
    )
    parser.add_argument(
        "--dest",
        default=str(RAW_DIR),
        help=f"destination directory for the raw files (default: {RAW_DIR})",
    )
    args = parser.parse_args(argv)

    dest = Path(args.dest)
    try:
        zip_path = download_zip(url=args.url, dest_dir=dest)
        extracted = extract_zip_members(zip_path, dest)
    except (ScaniaApsDownloadError, ScaniaApsError, UnsafeZipMemberError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = {
        "zip": str(zip_path.resolve()),
        "extracted": {name: str(path.resolve()) for name, path in extracted.items()},
    }
    # Last stdout line: the single-line JSON summary.
    print(json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

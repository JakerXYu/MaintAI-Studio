"""Evaluate the Scania APS copilot against the benchmark artifacts.

Replays the 12 fixed deterministic scenarios through the real P0 copilot
(offline mock provider) reading the benchmark artifacts and writes
``copilot_evaluation.json`` plus ``copilot_evaluation.md`` under the artifacts
directory. No live LLM, training, or benchmark re-run is involved.

With ``--deepseek-live`` the same 12 frozen probes are replayed plus 10
live-only probes through the DeepSeek provider (``https://api.deepseek.com``,
model ``deepseek-v4-flash``, thinking disabled). Results are written to
``copilot_evaluation_deepseek.json`` — never the mock artifacts. The API key is
read only from the generic settings key ``llm_api_key``; when it is missing the
command prints a clear SKIPPED JSON and performs no network call.

Usage (from the repo root)::

    python scripts/evaluate_scania_copilot.py
    python scripts/evaluate_scania_copilot.py --artifacts-dir artifacts/benchmarks/scania_aps
    python scripts/evaluate_scania_copilot.py --deepseek-live
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

from maintai.benchmarks.copilot_eval import (  # noqa: E402
    build_deepseek_provider,
    deepseek_provider_config,
    evaluate_deepseek,
    smoke_provider,
    write_deepseek_evaluation,
    write_evaluation,
)
from maintai.benchmarks.runner import ARTIFACTS_DIR  # noqa: E402
from maintai.config import get_settings  # noqa: E402


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def _run_deepseek_live(artifacts_dir: Path) -> int:
    settings = get_settings()
    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    if not api_key:
        _print_json(
            {
                "status": "skipped",
                "reason": (
                    "missing DeepSeek API key: set the generic settings key "
                    "`llm_api_key` (env LLM_API_KEY) and retry"
                ),
                "provider": deepseek_provider_config(),
                "artifacts_dir": str(artifacts_dir),
                "scenarios": [],
                "metrics": {},
            }
        )
        return 0

    provider = build_deepseek_provider(api_key)
    smoke = smoke_provider(provider)
    if smoke["status"] != "ok":
        _print_json(
            {
                "status": "error",
                "reason": smoke["error"],
                "provider": deepseek_provider_config(),
                "smoke": smoke,
                "artifacts_dir": str(artifacts_dir),
            }
        )
        return 1

    result = evaluate_deepseek(artifacts_dir, provider)
    result["smoke"] = smoke
    write_deepseek_evaluation(artifacts_dir, result)
    _print_json(
        {
            "status": "ok",
            "meta": result["meta"],
            "provider": result["provider"],
            "smoke": smoke,
            "mock_baseline": result["mock_baseline"],
            "replay": result["replay"],
            "live": result["live"],
        }
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--artifacts-dir",
        default=str(ARTIFACTS_DIR),
        help=f"directory holding the benchmark artifacts (default: {ARTIFACTS_DIR})",
    )
    parser.add_argument(
        "--deepseek-live",
        action="store_true",
        help="run the DeepSeek live evaluation (12 frozen probes + 10 live-only probes)",
    )
    args = parser.parse_args(argv)

    if args.deepseek_live:
        return _run_deepseek_live(Path(args.artifacts_dir))

    try:
        result = write_evaluation(Path(args.artifacts_dir))
    except Exception as exc:  # noqa: BLE001 - CLI boundary: clean exit code
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = {
        "meta": result["meta"],
        "metrics": result["metrics"],
        "na_scenarios": result["na_scenarios"],
    }
    _print_json(summary)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

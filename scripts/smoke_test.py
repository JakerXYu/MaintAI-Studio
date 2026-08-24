"""Read-only HTTP smoke test for a running MaintAI API (and optional Streamlit UI).

Checks the health/live/ready endpoints and the list endpoints for datasets,
experiments and models, all with GET requests only — nothing is mutated. On
success it prints a single-line JSON summary (last stdout line) and exits 0; on
any failure it prints the JSON summary with ``status: "failed"`` and exits 1.

Usage (from the repo root)::

    python scripts/smoke_test.py --api-url http://127.0.0.1:8000
    python scripts/smoke_test.py --api-url http://127.0.0.1:8000 --ui-url http://127.0.0.1:8501
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import httpx

_HEALTH_CHECKS: tuple[tuple[str, str, int], ...] = (
    ("health", "/health", 200),
    ("live", "/health/live", 200),
    ("ready", "/health/ready", 200),
)

_RESOURCE_CHECKS: tuple[tuple[str, str], ...] = (
    ("datasets", "/api/v1/datasets"),
    ("experiments", "/api/v1/experiments"),
    ("models", "/api/v1/models"),
)


def _check(client: httpx.Client, name: str, url: str, expected: int) -> dict[str, Any]:
    try:
        response = client.get(url)
    except httpx.HTTPError as exc:
        return {
            "name": name,
            "url": url,
            "ok": False,
            "error": type(exc).__name__,
        }
    ok = response.status_code == expected
    result: dict[str, Any] = {
        "name": name,
        "url": url,
        "status_code": response.status_code,
        "ok": ok,
    }
    if not ok:
        result["expected"] = expected
    return result


def _resource_check(client: httpx.Client, name: str, url: str) -> dict[str, Any]:
    try:
        response = client.get(url)
    except httpx.HTTPError as exc:
        return {
            "name": name,
            "url": url,
            "ok": False,
            "error": type(exc).__name__,
        }
    result: dict[str, Any] = {
        "name": name,
        "url": url,
        "status_code": response.status_code,
        "ok": response.status_code == 200,
    }
    if response.status_code == 200:
        try:
            payload = response.json()
        except ValueError:
            result["ok"] = False
            result["error"] = "non-JSON response"
        else:
            if isinstance(payload, list):
                result["count"] = len(payload)
    return result


def _ui_check(client: httpx.Client, url: str) -> dict[str, Any]:
    try:
        response = client.get(url)
    except httpx.HTTPError as exc:
        return {"name": "ui", "url": url, "ok": False, "error": type(exc).__name__}
    return {
        "name": "ui",
        "url": url,
        "status_code": response.status_code,
        "ok": response.status_code == 200,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="base URL of the FastAPI service (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="per-request timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--ui-url",
        default=None,
        help="optional Streamlit UI base URL to check",
    )
    args = parser.parse_args(argv)

    api_url = args.api_url.rstrip("/")
    checks: list[dict[str, Any]] = []

    with httpx.Client(timeout=args.timeout) as client:
        for name, path, expected in _HEALTH_CHECKS:
            checks.append(_check(client, name, f"{api_url}{path}", expected))
        for name, path in _RESOURCE_CHECKS:
            checks.append(_resource_check(client, name, f"{api_url}{path}"))
        if args.ui_url:
            checks.append(_ui_check(client, args.ui_url.rstrip("/")))

    failures = [check for check in checks if not check["ok"]]
    summary: dict[str, Any] = {
        "status": "ok" if not failures else "failed",
        "api_url": api_url,
        "ui_url": args.ui_url,
        "checks": checks,
        "failures": [failure["name"] for failure in failures],
    }

    # Last stdout line: the single-line JSON summary.
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))

    if failures:
        for failure in failures:
            print(f"smoke check failed: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

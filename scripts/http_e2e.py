"""Run the complete P0 workflow over a live MaintAI HTTP API.

This script is intentionally separate from the throwaway local demo. It proves
the deployed Postgres, HTTP MLflow, API, registry, artifact volume, inference,
and Copilot path used by Docker Compose.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx


class LiveE2EError(RuntimeError):
    """Raised when a live workflow step returns an unexpected response."""


def _json(response: httpx.Response, expected: set[int]) -> Any:
    if response.status_code not in expected:
        raise LiveE2EError(
            f"{response.request.method} {response.request.url.path} returned "
            f"{response.status_code}: {response.text[:500]}"
        )
    return response.json()


def run(
    *,
    api_url: str,
    dataset_path: Path,
    poll_timeout: float,
) -> dict[str, Any]:
    base = api_url.rstrip("/")
    with httpx.Client(base_url=base, timeout=60.0) as client:
        ready = _json(client.get("/health/ready"), {200})

        content = dataset_path.read_bytes()
        upload_response = client.post(
            "/api/v1/datasets/upload",
            files={"file": (dataset_path.name, content, "text/csv")},
        )
        if upload_response.status_code == 409:
            dataset_id = upload_response.json()["existing_dataset_id"]
        else:
            dataset_id = _json(upload_response, {201})["id"]

        profile = _json(
            client.post(f"/api/v1/datasets/{dataset_id}/profile"),
            {200},
        )
        task = _json(
            client.post(
                f"/api/v1/datasets/{dataset_id}/task-recommendation",
                json={
                    "target_column": "machine_failure",
                    "asset_id_column": "machine_id",
                    "timestamp_column": "timestamp",
                },
            ),
            {200},
        )

        experiment = _json(
            client.post(
                "/api/v1/experiments",
                json={"dataset_id": dataset_id, "minimum_recall": 0.8},
            ),
            {202},
        )
        deadline = time.monotonic() + poll_timeout
        while time.monotonic() < deadline:
            experiment = _json(
                client.get(f"/api/v1/experiments/{experiment['id']}"),
                {200},
            )
            if experiment["status"] in {"succeeded", "failed"}:
                break
            time.sleep(2.0)
        else:
            raise LiveE2EError("experiment polling timed out")
        if experiment["status"] != "succeeded":
            raise LiveE2EError(
                f"experiment failed: {experiment.get('error_message', 'unknown error')}"
            )

        comparison = _json(
            client.get(f"/api/v1/experiments/{experiment['id']}/comparison"),
            {200},
        )
        recommended = experiment["recommended_model"]
        model_run = next(
            run for run in experiment["model_runs"] if run["model_name"] == recommended
        )
        registered = _json(
            client.post(
                f"/api/v1/models/{model_run['id']}/register",
                json={"name": "machine-failure-compose"},
            ),
            {200, 201},
        )
        deployed = _json(
            client.post(f"/api/v1/models/{registered['id']}/deploy-demo"),
            {200},
        )

        record = {
            "type": "L",
            "air_temperature": 305.0,
            "process_temperature": 317.0,
            "rotational_speed": 1050.0,
            "torque": 92.0,
            "tool_wear": 245.0,
        }
        prediction = _json(
            client.post(
                "/api/v1/predict",
                json={"model_id": registered["id"], "records": [record]},
            ),
            {200},
        )

        copilot_requests = [
            {
                "user_request": "What data-quality problems do you see?",
                "dataset_id": dataset_id,
            },
            {
                "user_request": "What is the deployment status?",
                "model_id": registered["id"],
            },
            {
                "user_request": "Explain this prediction",
                "model_id": registered["id"],
                "record": record,
            },
        ]
        copilot = [
            _json(client.post("/api/v1/copilot/chat", json=request), {200})
            for request in copilot_requests
        ]

    return {
        "status": "ok",
        "ready": ready,
        "dataset_id": dataset_id,
        "rows": profile["row_count"],
        "quality_score": profile["quality_score"],
        "task": task["task"]["recommended_task"],
        "leakage_excluded": task["leakage"]["excluded_features"],
        "experiment_id": experiment["id"],
        "experiment_status": experiment["status"],
        "model_runs": len(experiment["model_runs"]),
        "recommended_model": recommended,
        "comparison_best": comparison["best_model"],
        "registered_model_id": registered["id"],
        "registry_version": registered["version"],
        "registry_alias": registered["alias"],
        "deployment_status": deployed["deployment_status"],
        "prediction": prediction["records"][0]["prediction"],
        "positive_probability": prediction["records"][0].get(
            "positive_probability"
        ),
        "explanation_method": prediction["records"][0]["explanation"]["method"],
        "copilot_intents": [response["intent"] for response in copilot],
        "copilot_grounded": all(response["evidence"] for response in copilot),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/synthetic/maintai_ai4i_style_demo.csv"),
    )
    parser.add_argument("--poll-timeout", type=float, default=300.0)
    args = parser.parse_args()
    try:
        summary = run(
            api_url=args.api_url,
            dataset_path=args.dataset.resolve(),
            poll_timeout=args.poll_timeout,
        )
    except (httpx.HTTPError, OSError, LiveE2EError, KeyError, StopIteration) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 1
    print(json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

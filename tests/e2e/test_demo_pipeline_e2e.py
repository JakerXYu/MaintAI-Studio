"""End-to-end tests for the P0 demo scripts (``scripts/**``).

These run the scripts as real subprocesses from the repository root (so the
scripts' own bootstrap path is exercised), parse the single-line JSON summary
each script prints on the last stdout line, and assert the demo loop closes
end-to-end. They use the venv's interpreter via ``sys.executable`` and never
write outside ``tmp_path`` or the repository's own ``data/synthetic`` default.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"

EXPECTED_COLUMNS = [
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
]


def _last_json(stdout: str) -> dict:
    """Parse the last non-empty stdout line as a single-line JSON object."""
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    assert lines, "script printed no stdout output"
    return json.loads(lines[-1])


def _run(script: Path, *args: str, timeout: int = 240) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def test_generator_is_byte_identical(tmp_path):
    out1 = tmp_path / "run1.csv"
    out2 = tmp_path / "run2.csv"
    script = SCRIPTS / "generate_demo_data.py"

    first = _run(script, "--output", str(out1), timeout=60)
    second = _run(script, "--output", str(out2), timeout=60)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr

    summary1 = _last_json(first.stdout)
    summary2 = _last_json(second.stdout)

    assert summary1["sha256"] == summary2["sha256"]
    assert summary1["rows"] == summary2["rows"]
    assert 300 <= summary1["rows"] <= 500

    bytes1 = out1.read_bytes()
    bytes2 = out2.read_bytes()
    assert bytes1 == bytes2
    assert summary1["sha256"] == hashlib.sha256(bytes1).hexdigest()
    committed = REPO_ROOT / "data" / "synthetic" / "maintai_ai4i_style_demo.csv"
    assert committed.read_bytes() == bytes1

    header = bytes1.decode("utf-8").splitlines()[0].split(",")
    assert header == EXPECTED_COLUMNS


def test_demo_pipeline_closes_full_loop():
    script = SCRIPTS / "run_demo_pipeline.py"
    result = _run(script, timeout=240)

    assert result.returncode == 0, result.stderr
    summary = _last_json(result.stdout)
    assert summary["status"] == "ok"

    # dataset / task / quality
    assert summary["dataset"]["target_column"] == "machine_failure"
    assert summary["dataset"]["task_type"] == "binary_classification"
    assert summary["task"]["recommended_task"] == "binary_classification"
    assert "serial_no" in summary["task"]["excluded_features"]
    assert isinstance(summary["quality"]["score"], (int, float))

    # experiment + 3 model runs
    assert summary["experiment"]["status"] == "succeeded"
    assert summary["experiment"]["recommended_model"]
    assert len(summary["model_runs"]) == 3
    assert {run["model_name"] for run in summary["model_runs"]} == {
        "logistic_regression",
        "random_forest",
        "xgboost",
    }
    assert all(run["status"] == "success" for run in summary["model_runs"])

    # registered candidate + deployed demo (never champion / Production)
    assert summary["registered"]["deployment_status"] == "candidate"
    assert summary["registered"]["alias"] == "candidate"
    assert summary["registered"]["mlflow_model_uri"].startswith("models:/")
    assert summary["deployed"]["deployment_status"] == "demo_deployed"
    assert summary["deployed"]["deployment_status"] != "champion"

    # prediction
    assert summary["prediction"]["count"] == 1
    assert summary["prediction"]["prediction"] == 1
    assert summary["prediction"]["positive_probability"] >= 0.5
    assert summary["prediction"]["explanation_method"]

    # 3 copilot calls with grounded evidence
    assert summary["copilot"]["intents"] == [
        "quality",
        "deployment_status",
        "explain_prediction",
    ]
    assert len(summary["copilot"]["evidence"]) == 3
    assert all(len(evidence) == 1 for evidence in summary["copilot"]["evidence"])
    assert summary["minimum_recall"] == 0.8
    assert summary["minimum_recall_note"]

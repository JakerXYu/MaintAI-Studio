"""Run the complete P0 demo loop end-to-end against throwaway local backends.

Everything lives under a ``tempfile.TemporaryDirectory``: an in-memory SQLite
database (``StaticPool``), a local-file MLflow tracking + registry backend, and
private storage/artifact directories. The script therefore writes nothing to the
repository, opens no network, reads no secrets, and performs no champion /
Production promotion (demo deploy only).

The deterministic data generator is reused from ``generate_demo_data``, so the
uploaded CSV is the same synthetic AI4I-style dataset the generator emits. The
full P0 chain is exercised:

    ingest -> profile -> task recommendation (leakage blocks ``serial_no``)
    -> train 3 catalog models -> recommend -> register candidate
    -> deploy demo -> predict -> mock copilot (quality / deployment / explain)

The last stdout line is a single-line JSON summary (see ``main``).

Usage (from the repo root)::

    python scripts/run_demo_pipeline.py
"""

# isort: skip_file
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# -- bootstrap: allow running from the repo root without an installed package --
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
_SCRIPTS = _REPO_ROOT / "scripts"
for _extra in (_SRC, _SCRIPTS):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import maintai.db.models  # noqa: F401, E402  (register models on Base.metadata)
from maintai.agent.provider import MockProvider  # noqa: E402
from maintai.agent.service import CopilotService  # noqa: E402
from maintai.agent.tools import CopilotTools  # noqa: E402
from maintai.application.datasets import DatasetService  # noqa: E402
from maintai.application.experiments import ExperimentService  # noqa: E402
from maintai.application.models import ModelRegistryService  # noqa: E402
from maintai.application.predictions import PredictionService  # noqa: E402
from maintai.audit.repository import AuditRepository  # noqa: E402
from maintai.audit.service import AuditService  # noqa: E402
from maintai.db.base import Base  # noqa: E402
from maintai.db.dataset_repository import DatasetRepository  # noqa: E402
from maintai.db.experiment_repository import ExperimentRepository  # noqa: E402
from maintai.db.model_repository import ModelRepository  # noqa: E402
from maintai.db.prediction_repository import PredictionRepository  # noqa: E402
from maintai.mlops import MLflowRegistry, MLflowTracker  # noqa: E402

import generate_demo_data  # noqa: E402

EXPERIMENT_NAME = "demo-ai4i-failure"
DATASET_FILENAME = "maintai_ai4i_style_demo.csv"
TARGET_COLUMN = "machine_failure"
ASSET_ID_COLUMN = "machine_id"
TIMESTAMP_COLUMN = "timestamp"
REGISTERED_NAME = "machine-failure-demo"
MINIMUM_RECALL = 0.0

MINIMUM_RECALL_NOTE = (
    "minimum_recall is set to 0.0 for this demo so the deterministic recommendation "
    "always produces a model on the small synthetic dataset; the P0 production default "
    "is 0.80 (see configs/default.yaml)."
)


def _recommended_run(experiment_service: ExperimentService, experiment_id: str) -> dict:
    experiment = experiment_service.get(experiment_id)
    recommended = experiment["recommended_model"]
    for run in experiment["model_runs"]:
        if run["model_name"] == recommended:
            return run
    raise RuntimeError("recommended model run not found")


def run_demo() -> dict:
    """Execute the full demo loop and return the JSON-safe summary."""
    with tempfile.TemporaryDirectory(prefix="maintai_demo_") as tmp:
        root = Path(tmp)

        engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

        storage_root = root / "storage"
        artifact_root = root / "artifacts"
        mlflow_uri = (root / "mlruns").as_uri()

        audit = AuditService(AuditRepository(session_factory))
        dataset_service = DatasetService(
            repository=DatasetRepository(session_factory),
            session_factory=session_factory,
            audit=audit,
            storage_root=storage_root,
        )
        tracker = MLflowTracker(mlflow_uri, EXPERIMENT_NAME)
        experiment_service = ExperimentService(
            session_factory=session_factory,
            dataset_service=dataset_service,
            experiment_repository=ExperimentRepository(session_factory),
            audit=audit,
            tracker=tracker,
            artifact_root=artifact_root,
        )
        registry_service = ModelRegistryService(
            session_factory=session_factory,
            model_repository=ModelRepository(session_factory),
            experiment_repository=ExperimentRepository(session_factory),
            audit=audit,
            registry=MLflowRegistry(mlflow_uri),
            artifact_root=artifact_root,
        )
        prediction_service = PredictionService(
            session_factory=session_factory,
            model_repository=ModelRepository(session_factory),
            prediction_repository=PredictionRepository(session_factory),
            audit=audit,
            artifact_root=artifact_root,
        )
        copilot = CopilotService(
            tools=CopilotTools(
                dataset_service=dataset_service,
                experiment_service=experiment_service,
                model_registry_service=registry_service,
                prediction_service=prediction_service,
            ),
            provider=MockProvider(),
        )

        # 1. Ingest the deterministic synthetic CSV.
        frame = generate_demo_data.generate_dataframe()
        uploaded = dataset_service.upload(
            generate_demo_data.to_csv_bytes(frame), DATASET_FILENAME
        )
        dataset_id = uploaded["id"]

        # 2. Profile + deterministic task recommendation (leakage blocks serial_no).
        dataset_service.profile(dataset_id)
        task = dataset_service.recommend_task(
            dataset_id,
            target_column=TARGET_COLUMN,
            asset_id_column=ASSET_ID_COLUMN,
            timestamp_column=TIMESTAMP_COLUMN,
        )
        quality = dataset_service.get_quality(dataset_id)
        dataset = dataset_service.get(dataset_id)

        # 3. Train the 3 catalog models and recommend the best.
        created = experiment_service.create(dataset_id, minimum_recall=MINIMUM_RECALL)
        experiment = experiment_service.run(created["id"])
        if experiment["status"] != "succeeded":
            raise RuntimeError(f"experiment failed: {experiment.get('error_message')}")

        run = _recommended_run(experiment_service, experiment["id"])

        # 4. Register the recommended run as a candidate.
        registered = registry_service.register(run["id"], REGISTERED_NAME)

        # 5. Deploy demo (never champion / Production).
        deployed = registry_service.deploy_demo(registered["id"])

        # 6. Predict on a single record (audited).
        record = {
            "type": "L",
            "air_temperature": 300.0,
            "process_temperature": 310.0,
            "rotational_speed": 1500.0,
            "torque": 40.0,
            "tool_wear": 80.0,
        }
        prediction = prediction_service.predict(registered["id"], [record])

        # 7. Mock copilot (offline): quality / deployment status / explain.
        copilot_calls = [
            ("quality", copilot.chat(user_request="数据质量怎么样？", dataset_id=dataset_id)),
            (
                "deployment_status",
                copilot.chat(
                    user_request="What is the deployment status of the demo model?",
                    model_id=registered["id"],
                ),
            ),
            (
                "explain_prediction",
                copilot.chat(
                    user_request="Explain this prediction",
                    model_id=registered["id"],
                    record=record,
                ),
            ),
        ]

        model_runs = experiment_service.get(experiment["id"])["model_runs"]
        record_result = prediction["records"][0]

        return {
            "dataset": {
                "id": dataset_id,
                "name": dataset["name"],
                "rows": dataset["row_count"],
                "columns": dataset["column_count"],
                "target_column": dataset["target_column"],
                "task_type": dataset["task_type"],
                "file_hash": dataset["file_hash"],
            },
            "task": {
                "recommended_task": task["task"]["recommended_task"],
                "confidence": task["task"]["confidence"],
                "leakage_verdict": task["leakage"]["verdict"],
                "excluded_features": task["leakage"]["excluded_features"],
            },
            "quality": {"score": quality["score"]},
            "experiment": {
                "id": experiment["id"],
                "status": experiment["status"],
                "recommended_model": experiment["recommended_model"],
                "primary_metric": experiment["primary_metric"],
                "value": experiment["value"],
            },
            "model_runs": [
                {
                    "model_name": item["model_name"],
                    "status": item["status"],
                    "primary_metric": item["primary_metric"],
                    "primary_metric_value": item["primary_metric_value"],
                }
                for item in model_runs
            ],
            "recommended": {
                "model_name": run["model_name"],
                "primary_metric": run["primary_metric"],
                "primary_metric_value": run["primary_metric_value"],
            },
            "registered": {
                "id": registered["id"],
                "name": registered["name"],
                "version": registered["version"],
                "deployment_status": registered["deployment_status"],
                "alias": registered["alias"],
                "mlflow_model_uri": registered["mlflow_model_uri"],
            },
            "deployed": {
                "id": deployed["id"],
                "name": deployed["name"],
                "version": deployed["version"],
                "deployment_status": deployed["deployment_status"],
            },
            "prediction": {
                "count": prediction["count"],
                "model_version": prediction["model_version"],
                "prediction": record_result["prediction"],
                "positive_probability": record_result.get("positive_probability"),
                "explanation_method": record_result["explanation"]["method"],
                "disclaimer": record_result["explanation"]["disclaimer"],
            },
            "copilot": {
                "intents": [intent for intent, _ in copilot_calls],
                "evidence": [response["evidence"] for _, response in copilot_calls],
            },
            "minimum_recall": MINIMUM_RECALL,
            "minimum_recall_note": MINIMUM_RECALL_NOTE,
            "status": "ok",
        }


def main() -> int:
    summary = run_demo()
    # Last stdout line: the single-line JSON summary.
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

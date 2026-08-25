# MaintAI Studio

Agentic predictive-maintenance **AutoML + MLOps copilot** for industrial
equipment (ABB Accelerator 2026 — Theme 1).

## Overview

**Problem.** Predictive maintenance usually fails before the model stage:
industrial sensor data is messy, the maintenance task is unclear to a
non-ML engineer, model evaluation is disconnected from what maintenance
actually costs, and notebook models rarely become monitored workflows. Most
"AutoML" demos reduce this to `upload CSV -> train XGBoost -> show accuracy`,
which is not how maintenance engineers work.

**Solution.** MaintAI Studio turns an equipment dataset into an auditable,
evidence-based predictive-maintenance loop:

```text
profile -> frame -> train -> explain -> register -> demo-deploy -> predict -> audit
```

Deterministic Python services do the data quality checks, task recommendation,
leakage detection, time/group-aware split, training, evaluation, SHAP
explanation, and registry operations. A LangGraph copilot plans, selects
read-only tools, and explains the results — it never executes ML or writes the
database/registry directly. A Streamlit control-room UI drives the whole loop
through HTTP.

**Status.** The full P0 chain is implemented and locally tested. The P0 Freeze
review is **in progress**, not passed: local gates (ruff, pytest, demo script)
pass on a Python 3.11 venv, but the Docker Compose runtime gate is still
pending because Docker is not installed on the machine that produced these
docs. See [`P0_REVIEW.md`](P0_REVIEW.md).

## ABB Theme 1 mapping

Theme 1 asks for an "AI-powered AutoML + MLOps Copilot for Industrial
Equipment". MaintAI Studio maps each official requirement to an implemented
capability:

| Official requirement | MaintAI capability |
|---|---|
| Automated dataset profiling and quality assessment | Deterministic schema inference + industrial profiling + >=6 quality checks |
| Intelligent task and model selection | Rule-engine task recommendation; deterministic best-model recommendation |
| Data preprocessing and feature engineering | Median/most-frequent imputation, OneHotEncoder, ID exclusion, fixed catalog |
| Model training and evaluation | LR / RandomForest / XGBoost (classification), Ridge / RF / XGBoost (regression) |
| Explainable AI (importance + confidence) | SHAP global + per-record local explanation; `predict_proba` / interval confidence |
| Experiment tracking and model comparison | Every run logged to MLflow; comparison + ranking endpoint and page |
| Deployment of trained models | Controlled two-step `candidate` registration -> `demo-deploy` -> FastAPI inference |
| Interactive prediction and inference dashboard | Streamlit Predict page (single JSON + batch CSV) and Copilot |

Suggested stack adopted: Python 3.11, FastAPI, MLflow, LangGraph, Docker,
SHAP, XGBoost, PostgreSQL. LightGBM is intentionally not offered in P0.

## Differentiators

These are the eight differentiators from the implementation spec, with their
original meaning. Items marked (P1) are planned and not implemented yet.

1. **Industrial Data Health Before ML** — fix sensor/data quality *before*
   modeling, not after a broken model appears.
2. **Task Intelligence** — the engineer does not need to know
   classification / regression / anomaly in advance.
3. **Cost-Aware, Not Accuracy-Only** (P1) — a false negative and a false
   positive have different maintenance costs.
4. **Explainable + Confidence-Aware** — results carry evidence, not a
   black-box score.
5. **Agentic Orchestration with Deterministic Tools** — the LLM plans and
   narrates but never runs uncontrolled code.
6. **Closed-Loop MLOps** (P1) — after deploy, keep monitoring, detect drift,
   and recommend retraining.
7. **Human-in-the-Loop** (P1) — model promotion and maintenance actions stay
   under engineer control.
8. **Maintenance Workflow Bridge** (P1) — end in a mock CMMS action, not just
   another dashboard.

P0 implements differentiators 1, 2, 4 and 5. Differentiators 3, 6, 7 and 8
are P1 scope (see feature matrix).

## Architecture (flow)

Python 3.11 modular monolith. **FastAPI is the only business entry point**;
**Streamlit only calls the HTTP API**; **Postgres** stores business metadata +
audit; **MLflow** is an independent tracking/registry store; **SQLite** is a
test-only double; **mock LLM** is the default.

```text
Engineer ──► Streamlit UI (HTTP only) ──► FastAPI API ──► Data/ML services
                                          │   │
                          ┌───────────────┤   └──────────► LangGraph copilot
                          ▼               ▼                  (read-only tools)
                     PostgreSQL         MLflow
                  (business metadata   (tracking +
                   + audit)             registry)

P0 pipeline: ingest -> profile -> task (leakage) -> train -> evaluate ->
             recommend -> register (candidate) -> demo-deploy -> predict
```

Full details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Feature matrix

| Area | P0 (implemented) | P1 (planned) |
|---|---|---|
| Data | CSV/Parquet upload, SHA-256 dedupe, schema inference, profiling, >=6 quality checks, leakage detection, time/group-aware split | — |
| Tasks / ML | Task recommendation (rule engine), preprocessing, 3 classification + 3 regression models, metrics, SHAP + permutation fallback, confidence, deterministic best-model | Cost-aware selection, unsupervised anomaly |
| MLOps | MLflow tracking, candidate registration, `demo-deploy`, FastAPI predict (single/batch), audit log | Champion/challenger, drift, retraining, approval |
| Agent | LangGraph copilot, read-only tool allowlist, mock + openai-compatible providers, action refusal/proposal | Action tools (drift, cost, CMMS, approval) |
| UI | Home, Dataset & Health, Task & Plan, Experiments, Explainability, Registry & Deploy, Predict, Copilot | Monitoring, Approvals, CMMS pages |
| Ops | Docker Compose stack (runtime-unverified locally), single in-process worker, `create_all` bootstrap | Alembic, replay/monitoring |

P1 is **not started** and must not begin until the P0 Freeze closes.

## Quick start

### Docker Compose (unverified on this machine)

```bash
cd Project_ABB
cp .env.example .env        # demo defaults; override for real secrets
docker compose up --build
```

Services: `postgres` (init creates `maintai` + `mlflow` DBs), `mlflow`, a
one-shot `migrate` (`Base.metadata.create_all`), `api`, and `ui`, with
healthchecks and named volumes.

> **Status:** this Compose path has **not** been runtime-validated on the
> machine that produced these docs because Docker is not installed locally.
> The `docker-compose.yml`, `Dockerfile`, and init script are present and
> reviewed but pending a real run (see `P0_REVIEW.md` and
> [`docs/TEST_PLAN.md`](docs/TEST_PLAN.md)).

### Local (Python 3.11 venv)

Option A — `uv`:

```powershell
uv sync --extra dev
.venv\Scripts\Activate.ps1
```

Option B — `pip`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Run the API against SQLite for a quick check:

```powershell
$env:DATABASE_URL = "sqlite+pysqlite:///./maintai.db"
$env:MLFLOW_TRACKING_URI = "file:///$((Get-Location).Path -replace '\\','/')/mlruns"
python -m maintai.db.migrate
uvicorn maintai.api.main:app --reload --port 8000
```

Runtime normally uses Postgres + MLflow via Compose; SQLite + local-file MLflow
are the supported local/test doubles.

### One-command local demo

```powershell
python scripts/run_demo_pipeline.py
```

Runs the entire P0 loop against throwaway in-memory SQLite and a temp-file
MLflow backend (no network, no secrets, nothing written to the repo): ingest ->
profile -> task (leakage blocks `serial_no`) -> train 3 models -> recommend ->
register `candidate` -> `demo-deploy` -> predict -> three grounded copilot
answers. The last stdout line is a single-line JSON summary.

## Demo data & provenance

`data/synthetic/maintai_ai4i_style_demo.csv` is a **synthetic** dataset
generated by `scripts/generate_demo_data.py` with a fixed seed (42). It mirrors
the *shape* of the public AI4I 2020 Predictive Maintenance dataset (UCI) — a
10-column telemetry analogue with L/M/H product types — but every value is
synthesized here. No UCI row is copied. A flatline sensor, a torque outlier,
missing values, and one duplicate row are injected deterministically so the
demo exercises the data-quality path. Regenerate byte-for-byte:

```powershell
python scripts/generate_demo_data.py
```

See [`data/README.md`](data/README.md) for full provenance, privacy, and
license notes.

## Services

```text
UI:      http://localhost:8501
API:     http://localhost:8000/docs
Health:  http://localhost:8000/health
MLflow:  http://localhost:5000
```

## Workflow walkthrough

1. **Upload** — drop a CSV/Parquet on *Dataset & Health* (or
   `POST /api/v1/datasets/upload`). The API dedupes by SHA-256 and enforces
   extension/size limits.
2. **Profile** — run profile to compute schema, missingness, outliers,
   flatline, duplicates, and a heuristic health score.
3. **Task** — on *Task & Plan*, choose target / asset / timestamp and run
   recommendation; leakage detection blocks `serial_no` (an id-like column).
4. **Train** — create an experiment on *Experiments*; the single in-process
   worker trains LR / RF / XGBoost and logs every run to MLflow.
5. **Compare** — the deterministic recommender ranks by primary metric,
   then complexity, then latency; the LLM only narrates the result.
6. **Explain** — *Explainability* shows global SHAP importance; the Predict
   page produces per-record local explanations with a non-root-cause
   disclaimer.
7. **Register & deploy** — register the recommended run as `candidate`, then
   `deploy-demo` (demo serving only, never `champion`/Production).
8. **Predict** — single JSON record or batch CSV against the demo-deployed
   model; every prediction is persisted as a content-hashed event.
9. **Copilot** — ask "What data-quality problems do you see?", "Which model
   should I deploy and why?", or "Explain this prediction"; answers are
   grounded in tool evidence.

## API examples (curl, Windows-friendly)

On Windows PowerShell use `curl.exe` (not the `curl` alias). JSON bodies are
single-quoted so no escaping is needed.

```powershell
# Upload
curl.exe -X POST "http://localhost:8000/api/v1/datasets/upload" -F "file=@data/synthetic/maintai_ai4i_style_demo.csv"

# List datasets
curl.exe "http://localhost:8000/api/v1/datasets"

# Profile
curl.exe -X POST "http://localhost:8000/api/v1/datasets/{id}/profile"

# Task recommendation
curl.exe -X POST "http://localhost:8000/api/v1/datasets/{id}/task-recommendation" `
  -H "Content-Type: application/json" `
  -d '{"target_column":"machine_failure","asset_id_column":"machine_id","timestamp_column":"timestamp"}'

# Create experiment (202 accepted; poll GET /experiments/{id})
curl.exe -X POST "http://localhost:8000/api/v1/experiments" `
  -H "Content-Type: application/json" -d '{"dataset_id":"{id}"}'

# Register the recommended run as candidate
curl.exe -X POST "http://localhost:8000/api/v1/models/{model_run_id}/register" `
  -H "Content-Type: application/json" -d '{"name":"machine-failure-demo"}'

# Demo deploy
curl.exe -X POST "http://localhost:8000/api/v1/models/{id}/deploy-demo"

# Predict (single record)
curl.exe -X POST "http://localhost:8000/api/v1/predict" `
  -H "Content-Type: application/json" `
  -d '{"model_id":"{id}","records":[{"type":"L","air_temperature":305.0,"process_temperature":317.0,"rotational_speed":1050.0,"torque":92.0,"tool_wear":245.0}]}'

# Copilot
curl.exe -X POST "http://localhost:8000/api/v1/copilot/chat" `
  -H "Content-Type: application/json" `
  -d '{"user_request":"What data-quality problems do you see?","dataset_id":"{id}"}'
```

Replace `{id}`, `{model_run_id}` with real ids from the list endpoints. See
[`docs/API_CONTRACT.md`](docs/API_CONTRACT.md) for status codes and full shapes.

## Agent safety model

The copilot is read-only and grounded. It routes intent deterministically,
calls a fixed allowlist of read-only tools, and always embeds tool evidence in
its answer. Action requests (train/register/deploy/promote/work-order) are
never executed — they return a proposal or refusal with approval semantics.
The provider abstraction defaults to an offline mock and never hard-codes a
vendor. The synthesized narrative is rejected if it introduces any number not
present in the tool evidence. Details: [`docs/AGENT_DESIGN.md`](docs/AGENT_DESIGN.md)
and [`docs/SECURITY_AND_SAFETY.md`](docs/SECURITY_AND_SAFETY.md).

## MLOps lifecycle

1. Every training run is logged to MLflow (params, metrics, artifacts, tags).
2. The deterministic recommender picks the best successful run.
3. The recommended run is registered as a `candidate` alias.
4. `deploy-demo` flips exactly one version to `demo_deployed` (demo serving).
5. Inference runs only against `demo_deployed` models; each prediction is
   audited with a content hash (never the raw input).
6. `champion`/Production promotion and retraining deployment are **not** in
   P0 — they require human approval and belong to P1.

## Testing

```powershell
ruff check .      # lint
pytest -q         # unit + integration + e2e (SQLite double)
```

The final local P0 review run on the Python 3.11 venv collected and passed
**385 tests**, including the two subprocess E2E tests and the single-training
concurrency guard. See
[`docs/TEST_PLAN.md`](docs/TEST_PLAN.md).

## Known limitations

- Docker Compose has not been runtime-validated on this machine (Docker not
  installed); only the local Python 3.11 venv path is verified.
- Schema bootstrap uses `create_all`; Alembic is not yet introduced, so
  `create_all` does not migrate an existing database.
- Training runs as a single in-process worker (no Celery/Redis/Kafka).
- MLflow is the tracking/registry catalog; inference uses a manifest-rich,
  trusted package in the API artifact volume, cross-linked to its ModelRun.
- `demo_deployed` is demo serving only — there is no `champion`/Production
  transition in P0.
- P1 (anomaly/drift/cost/approval/champion-challenger/retraining/feedback/
  mock CMMS) is not started.
- Demo config values are **demo defaults**, not ABB or any industry standard.

## Privacy / data statement

No proprietary or real asset data is used. The demo dataset is synthetic,
generated in-repo from a fixed seed, and contains no personal information, no
real asset identifiers, and no production telemetry. Credentials in
`.env.example` are demo defaults. This prototype provides model-based decision
support only; it does not replace qualified maintenance, safety, or
engineering judgment.

## Future work (P2)

Not started, listed for direction only: C-MAPSS RUL, deep time-series models,
Kafka streaming, edge ONNX, OPC UA live connector, real CMMS connector
abstraction, RAG over maintenance manuals, multi-modal diagnostics, digital
twin context, cloud/Kubernetes, and causal diagnosis.

## Docs

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — boundaries + key decisions.
- [`docs/P0_SCOPE.md`](docs/P0_SCOPE.md) / [`docs/P1_SCOPE.md`](docs/P1_SCOPE.md).
- [`docs/AGENT_DESIGN.md`](docs/AGENT_DESIGN.md) — copilot graph/tools.
- [`docs/SECURITY_AND_SAFETY.md`](docs/SECURITY_AND_SAFETY.md) — threat model.
- [`docs/TEST_PLAN.md`](docs/TEST_PLAN.md) / [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md).
- [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md) / [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md).
- [`P0_REVIEW.md`](P0_REVIEW.md) — freeze review checklist (conditional).
- `AGENTS.md` — executable constraints for downstream agents.

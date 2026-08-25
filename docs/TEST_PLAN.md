# Test Plan — MaintAI Studio (P0)

How tests are organized, run, and triaged. All tests run against SQLite
(double) or throwaway local backends; runtime uses Postgres + MLflow.

## Commands

```powershell
ruff check .      # lint (dev extra)
pytest -q         # unit + integration + e2e, from the repo root
python scripts/run_demo_pipeline.py   # P0 loop, throwaway backends
python scripts/smoke_test.py --api-url http://127.0.0.1:8000 [--ui-url http://127.0.0.1:8501]
```

Targeted suites (run individually when iterating):

```powershell
pytest -q tests/unit                    # data/tasks/ml/agent/ui/config/audit/contracts
pytest -q tests/integration             # API + service + MLflow registry/tracking
pytest -q tests/e2e                     # scripts/run_demo_pipeline + generate_demo_data
```

## Suite breakdown

### Unit (`tests/unit`)

- `data/` — ingest, schema, profile, quality, leakage, split, infer.
- `ml/` — catalog, preprocess, train, evaluate, recommend, explain,
  confidence, package, schemas.
- `agent/` — graph, router, tools, provider, security (allowlist/read-only).
- `ui/` — api_client, app smoke, render.
- root — config, contracts, audit, health, request-id.

### Integration (`tests/integration`)

- `test_dataset_api.py` / `test_dataset_service.py` — upload -> profile ->
  task, error paths.
- `test_experiment_api.py` / `test_experiment_service.py` — train -> MLflow
  -> comparison, background failure persistence.
- `test_mlflow_tracking.py` / `test_mlflow_registry.py` — tracking/registry
  gateways on a local-file MLflow backend.
- `test_registry_prediction_service.py` / `test_model_prediction_api.py` —
  register -> demo-deploy -> predict.
- `test_copilot_api.py` — copilot -> tool -> grounded response.

### E2E (`tests/e2e`)

- `test_generator_is_byte_identical` — fixed seed produces byte-identical CSV.
- `test_demo_pipeline_closes_full_loop` — runs `scripts/run_demo_pipeline.py`
  as a subprocess and asserts the full loop (3 models, candidate -> demo
  deploy, predict, 3 grounded copilot calls, leakage blocks `serial_no`).

### Compose (`docker compose`, PENDING)

Not yet run on this machine (Docker not installed). Intended checks once
available:

```powershell
docker compose up --build
python scripts/smoke_test.py --api-url http://127.0.0.1:8000 --ui-url http://127.0.0.1:8501
```

Accept: `postgres`, `mlflow`, `migrate` (one-shot), `api`, `ui` healthy;
`/health/ready` returns `database: ok`, `mlflow: ok`; UI reachable.

## Failure localization

| Symptom | Check |
|---|---|
| Import errors under pytest | Run from repo root; `pyproject.toml` sets `pythonpath=["src"]`. Verify venv: `.venv\Scripts\Activate.ps1` then `pip install -e ".[dev]"` / `uv sync --extra dev`. |
| xgboost catalog "unavailable" | xgboost not importable; the catalog degrades gracefully but e2e expects 3 runs, so install the declared dependency. |
| Demo e2e nonzero exit | Re-run `python scripts/run_demo_pipeline.py` directly and read its last stdout JSON; the script prints nothing to the repo and needs no network. |
| `test_demo_pipeline_closes_full_loop` timeout | Training 3 models on 401 rows should finish well under the 240s subprocess timeout; check `n_jobs`/CPU contention. |
| Copilot quantitative-answer test fails | The synthesize step rejects provider prose with numbers absent from tool evidence; confirm `MockProvider` is active (`LLM_PROVIDER=mock`). |
| `smoke_test.py` non-200 | Confirm the API/UI are actually running at the given URLs; `--api-url` defaults to `127.0.0.1:8000`. |
| Ruff failures | `ruff check .` with `[tool.ruff]` line-length 100; fix or justify with `noqa`. |

## Current status

- Local Python 3.11 venv: `ruff check .` clean; final local `pytest -q`
  collected and passed **385 tests**, including E2E and the single-training
  concurrency guard. Targeted review runs:
  UI 59, Copilot/preview 82, e2e 2.
- `uv lock --check` passes. Demo data SHA-256:
  `2b4108048cad5a1aa3c2328890c7d674cdbde24c3f130aceeae6092c3ad33c1b`.
- Docker Compose runtime: **pending** (unverified locally — no Docker).

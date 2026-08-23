# Architecture — MaintAI Studio

Python 3.11 modular monolith. FastAPI is the single business entry point;
Streamlit is a thin HTTP client; Postgres holds business metadata + audit;
MLflow is a separate tracking/registry store.

## Component boundaries

```text
Engineer ──► Streamlit UI (HTTP only) ──► FastAPI API ──► Data/ML services
                                            │
                          ┌─────────────────┼──────────────────┐
                          ▼                 ▼                  ▼
                     PostgreSQL         MLflow             Audit log
                 (business metadata   (independent       (same Postgres,
                  + audit)             tracking/registry)  audit_events table)
```

- **Streamlit never** imports ML internals or DB directly. It only calls the API.
- **FastAPI** owns validation, orchestration, and persistence boundaries.
- **Postgres** stores `datasets`, `experiments`, `model_runs`,
  `registered_models`, `audit_events`. **MLflow** stores experiment runs,
  params/metrics/artifacts, and the model registry (independent store).
- **SQLite** is a test-only double (see below).

## Key decisions / adjustments vs. the long spec

1. **SQLite test double.** Unit/integration tests use an in-memory or temp
   SQLite database (same SQLAlchemy models). Runtime always uses Postgres.
   SQLAlchemy models use portable column types only.
2. **P0 single in-process training worker.** No Celery/Redis/Kafka. Long-running
   training in P0 runs in-process (FastAPI `BackgroundTasks` / a single worker),
   documented as a conscious simplification.
3. **Schema bootstrap via `create_all`.** Phase A uses
   `Base.metadata.create_all` (no Alembic). Alembic is introduced only if/when
   migrations are actually needed before P0 freeze.
4. **mock LLM default.** `LLM_PROVIDER=mock`; no network needed offline. The
   `openai-compatible` path is env-configured, never hard-coded.
5. **UI/API boundary.** Streamlit app reads `MAINTAI_API_URL` (default
   `http://localhost:8000`) and does not assume a shared process.

## Runtime topology (docker-compose)

| Service | Role | Store |
|---|---|---|
| `postgres` | single Postgres instance; init script creates both `maintai` and `mlflow` DBs | named volume `pgdata` |
| `mlflow` | MLflow tracking/registry server and artifact proxy | `mlflow` DB + named volume `mlflow_artifacts` |
| `migrate` | one-shot `create_all` for `maintai` schema | Postgres |
| `api` | FastAPI (uvicorn) | Postgres |
| `ui` | Streamlit health app | via API |

## Package layout (Phase A)

```text
src/maintai/
  config.py          pydantic-settings (env + configs/default.yaml)
  db/                Base, session/engine, ORM models, migrate
  audit/             audit repository + service
  api/               app factory, request-id middleware, health routes
  ui/                Streamlit health app
```

Future phases add `data/`, `tasks/`, `ml/`, `agent/`, `monitoring/`,
`approvals/`, `cmms/` per module ownership (see `AGENTS.md`).

## Health model

- `GET /health` — process liveness + service metadata.
- `GET /health/live` — minimal liveness (`{"status":"ok"}`).
- `GET /health/ready` — readiness; checks both DB `SELECT 1` and MLflow `/health`;
  returns 503 if either dependency is unreachable.
- Every response carries `X-Request-ID` (echoed or generated; invalid → 400).

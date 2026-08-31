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
  training artifacts, and the registry catalog. The API artifact volume stores
  the manifest-rich trusted serving package cross-linked to its `ModelRun`.
- **SQLite** is a test-only double (see below).

## Key decisions / adjustments vs. the long spec

1. **SQLite test double.** Unit/integration tests use an in-memory or temp
   SQLite database (same SQLAlchemy models). Runtime always uses Postgres.
   SQLAlchemy models use portable column types only.
2. **P0 single in-process training worker.** No Celery/Redis/Kafka. Long-running
   training uses FastAPI `BackgroundTasks` behind a process-wide coordinator
   lock, so at most one experiment trains at a time. The queue is not durable.
3. **Schema bootstrap via `create_all`.** The P0 schema is created with
   `Base.metadata.create_all` (no Alembic). Alembic is deferred; `create_all`
   does not migrate an existing database.
4. **mock LLM default.** `LLM_PROVIDER=mock`; no network needed offline. The
   `openai-compatible` path is env-configured, never hard-coded.
5. **UI/API boundary.** Streamlit app reads `MAINTAI_API_URL` (default
   `http://localhost:8000`) and does not assume a shared process.
6. **Demo deploy is not production.** `demo_deployed` is the only deployment
   state in P0. There is no `champion` alias or `Production` stage; those
   transitions require human approval and belong to P1.

## Runtime topology (docker-compose)

| Service | Role | Store |
|---|---|---|
| `postgres` | single Postgres instance; init script creates both `maintai` and `mlflow` DBs | named volume `pgdata` |
| `mlflow` | MLflow tracking/registry server and artifact proxy | `mlflow` DB + named volume `mlflow_artifacts` |
| `migrate` | one-shot `create_all` for `maintai` schema | Postgres |
| `api` | FastAPI (uvicorn) | Postgres |
| `ui` | Streamlit health app | via API |

## Package layout (P0, implemented)

```text
src/maintai/
  config.py          pydantic-settings (env + configs/default.yaml)
  db/                Base, session/engine, ORM models, migrate, repositories
  audit/             audit repository + service
  data/              ingest, schema, profile, quality, leakage, split, schemas
  tasks/             deterministic task inference + schemas
  ml/                catalog, preprocess, train, evaluate, recommend, explain,
                     confidence, package, schemas
  mlops/             MLflow tracker + registry gateways
  agent/             state, graph, tools, provider, service (LangGraph copilot)
  application/       dataset/experiment/model/prediction services (orchestration)
  api/               app factory, request-id middleware, routers
  ui/                Streamlit control-room app (HTTP-only client)
```

P1 adds `monitoring/`, `approvals/`, `feedback/`, and `cmms/`. Additive lifecycle,
execution-receipt, feedback, and mock-work-order tables preserve P0 deployment
semantics. Approval and execution remain separate; the Streamlit P1 page uses
only the same HTTP API boundary as the frozen P0 pages.

## Health model

- `GET /health` — process liveness + service metadata.
- `GET /health/live` — minimal liveness (`{"status":"ok"}`).
- `GET /health/ready` — readiness; checks both DB `SELECT 1` and MLflow `/health`;
  returns 503 if either dependency is unreachable.
- Every response carries `X-Request-ID` (echoed or generated; invalid → 400).

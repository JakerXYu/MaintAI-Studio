# MaintAI Studio

Agentic predictive-maintenance **AutoML + MLOps copilot** for industrial
equipment (ABB Accelerator 2026 — Theme 1). This repo currently contains the
**Phase A foundation** only; P0 data/ML/agent/UI features are not yet
implemented.

## Current status

| Layer | Status |
|---|---|
| Repo foundation (config, docs, Docker) | ✅ Phase A |
| Postgres + MLflow + FastAPI + UI health stack | Phase A implemented; runtime validation pending |
| Business ORM models + audit | Phase A implemented; runtime validation pending |
| P0 data intelligence (ingest/profile/task) | Not started |
| P0 ML pipeline / MLflow / registry | Not started |
| P0 LangGraph copilot | Not started |
| P0 Streamlit dashboards | Not started |
| P1 (monitoring/approval/CMMS) | Not started |

## Architecture (summary)

Python 3.11 modular monolith. **FastAPI is the only business entry point**;
**Streamlit only calls the HTTP API**; **Postgres** stores business metadata +
audit; **MLflow** is an independent tracking/registry store; **SQLite** is a
test-only double; **mock LLM** is the default. Full details:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Quick start (Docker)

```bash
cd Project_ABB
cp .env.example .env        # demo defaults; override for real secrets
docker compose up --build
```

Then:

```text
UI:     http://localhost:8501
API:    http://localhost:8000/docs
Health: http://localhost:8000/health
MLflow: http://localhost:5000
```

`docker compose up` starts `postgres` (init creates both `maintai` and `mlflow`
DBs), `mlflow`, a one-shot `migrate` (`Base.metadata.create_all`), `api`, and
`ui`, with healthchecks and named volumes.

## Local development (without Docker)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -e ".[dev]"
# run against SQLite for a quick check:
$env:DATABASE_URL = "sqlite+pysqlite:///./maintai.db"
python -m maintai.db.migrate
uvicorn maintai.api.main:app --reload --port 8000
```

## Endpoints (Phase A)

```text
GET /health         liveness + service metadata
GET /health/live    minimal liveness
GET /health/ready   readiness (DB + MLflow; 503 if either is unavailable)
```

Every response carries `X-Request-ID` (echoed if valid, generated if absent,
`400` if invalid).

## Testing

```bash
pip install -e ".[dev]"
pytest -q
```

Tests use an in-memory SQLite double and cover config, ORM contracts, audit,
health, and request-ID behavior.

## Docs

- [`docs/REUSE_MANIFEST.md`](docs/REUSE_MANIFEST.md) — reference-project audit.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — boundaries + key decisions.
- [`docs/P0_SCOPE.md`](docs/P0_SCOPE.md) / [`docs/P1_SCOPE.md`](docs/P1_SCOPE.md).
- [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md) / [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md).
- `AGENTS.md` — executable constraints for downstream agents.

## Known limitations (Phase A)

- No dataset/ML/agent/UI business features yet (P0 pending).
- Compose has not yet been runtime-validated on this machine because Docker is
  not installed; Python 3.11 is also not currently available locally.
- Schema bootstrap uses `create_all`; Alembic not yet introduced.
- Demo DB credentials in `.env.example` are **demo defaults**, not real secrets.
- This prototype provides model-based decision support only; it does not
  replace qualified maintenance, safety, or engineering judgment.

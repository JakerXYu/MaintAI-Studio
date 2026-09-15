# Reuse Manifest — MaintAI Studio

Audit of the reference project
`Project_01_Industrial_Maintenance_Root_Cause_Agent_CN_EN`
(hereafter "reference") against the MaintAI Studio Phase A needs.

**Rule**: the reference is read-only. Nothing is modified. Reuse means *copy
into* `Project_ABB` with a source note; never import across projects.

**Key audit findings**

- Reference is a **root-cause maintenance agent** (synthetic CMMS + RAG + planner),
  not an AutoML/MLOps platform. Its domain schemas and agent logic are
  tightly coupled to that task.
- Reference has **no** LangGraph, **no** PostgreSQL, **no** SQLAlchemy, **no**
  MLflow, **no** Alembic, **no** LLM provider `Protocol` implementation (it
  defines `llm_*` config fields but never calls an LLM in Phase 0).
- Reference persistence is raw `sqlite3` DDL + a read-only repository. It is
  not portable to Postgres and is not ORM-based.
- Reference has **no LICENSE file** (and no NOTICE/COPYING). Absence of a
  license means normal copyright restrictions apply; it is a reuse and public
  submission risk unless ownership/permission is confirmed. This project only
  adapts small generic patterns and records their provenance. No proprietary
  data, credentials, tokens, or real CMMS/SSP/Fiix endpoints are copied.

## Reuse table

| Existing Source | Capability | Reuse? | Destination | Modification Needed | Risk |
|---|---|---|---|---|---|
| `src/agent/request_id.py` | Request-ID validation (safe charset + Windows reserved basename) | 复制后重构 | `src/maintai/api/request_id.py` | Reimplemented as a small API-boundary validator; preserve provenance until reuse permission is confirmed | Medium: reference has no license |
| `src/api/main.py` | FastAPI app factory + `/health` | 复制后重构 | `src/maintai/api/main.py` | Add `/health/live`, `/health/ready`, request-ID middleware, injected settings/session-factory | Low |
| `src/config.py` | Pydantic settings from env, secrets never printed | 复制后重构 | `src/maintai/config.py` | Migrate to `pydantic-settings` + `configs/default.yaml`; add DB/MLflow/LLM fields | Low |
| `src/contracts/common.py` | `utcnow`, enum, `ToolResult`/`ToolError` envelope | 复制后重构 | `src/maintai/contracts/` (later phase) | Replace CMMS enums with MaintAI domain; keep envelope pattern | Low |
| `src/db/repository.py` | Parameterized, read-only sqlite3 repo pattern | 重新实现 | `src/maintai/db/`, `src/maintai/audit/` | Replace `sqlite3` with SQLAlchemy ORM over Postgres/SQLite; add write path + audit | Medium |
| `src/db/schema.py` | Raw SQLite DDL for CMMS domain | 禁止复制 | n/a | Root-cause domain; reimplement as SQLAlchemy models | Medium |
| `src/agent/*` (planner/executor/synthesizer/runner/tools) | Root-cause agent orchestration | 禁止复制 | n/a | No LangGraph; logic is root-cause specific; reimplement in Phase E | High |
| `src/tools/*` (asset/meter/work_order) | Domain tools over CMMS | 禁止复制 | n/a | Root-cause domain; not applicable to AutoML | High |
| `src/rag/*`, `src/analytics/*`, `src/evaluation/*` | RAG + trend + eval scaffolding | 禁止复制 | n/a | Out of P0 scope | High |
| `ui/streamlit_app.py` | Streamlit app calling API | 复制后重构 | `src/maintai/ui/app.py` | Phase A health page only; all UI via HTTP API | Low |
| `Dockerfile` | `python:3.11-slim` build pattern | 复制后重构 | root `Dockerfile` | New dependency/install layout and overridable command | Low |
| `docker-compose.yml` | api + ui compose | 复制后重构 | root `docker-compose.yml` | Add postgres, mlflow, migrate, healthchecks, named volumes | Medium |
| `.gitignore` / `.dockerignore` | Python/venv/env ignore patterns | 复制后重构 | root | Rewritten for MaintAI paths and artifacts | Low |
| `.env.example` | Env template, no secrets | 复制后重构 | root `.env.example` | New vars: `DATABASE_URL`, `MLFLOW_TRACKING_URI`, `LLM_*` | Low |
| `requirements.txt` | Pinned dependency list | 重新实现 | `pyproject.toml` | Add SQLAlchemy, psycopg, pydantic-settings, streamlit, pyyaml | Low |
| `tests/test_request_id.py`, `tests/test_api.py`, `tests/test_contracts.py` | Test patterns | 复制后重构 | `tests/` | Adapt to FastAPI TestClient + SQLAlchemy SQLite fixtures | Low |
| `data/`, `traces/`, synthetic data | CMMS data + eval traces | 禁止复制 | n/a | Data/traces; not copied (proprietary/data rule) | High |

## Reuse category legend

- **直接复制** — generic infrastructure, no proprietary data, compatible API contract.
- **复制后重构** — useful logic but coupled or schema-mismatched.
- **重新实现** — concept is reusable, code is not portable (e.g. sqlite3 → SQLAlchemy).
- **禁止复制** — domain-locked, contains data/traces, or out of scope.

## Prohibited items (never copy)

`.env`, secrets/API keys, `data/` (CMMS + synthetic), `traces/` (eval outputs),
`.venv/`, `__pycache__/`, `.pytest_cache/`, and any real SSP/Bosch/Fiix asset,
work-order, BOM, quote, or endpoint data. None of these are used.

## External benchmark provenance

The optional Scania APS track downloads **APS Failure at Scania Trucks** from
the UCI Machine Learning Repository (DOI `10.24432/C51S51`) into ignored local
storage. It is not copied from the reference project and is not ABB, SSP,
Bosch, Fiix, or other proprietary project data. The UCI page lists CC BY 4.0;
the archive description also carries a GPL-3.0-or-later notice from Scania CV
AB. Raw rows and generated model artifacts are never committed.

# AGENTS.md — MaintAI Studio

Executable constraints for all agents working on this repo. This is a
compressed version of the implementation spec; when in doubt, re-read
`ABB_Accelerator_2026_Theme1_MaintAI_Studio_P0_P1_Implementation_Spec_CN.md`
in the parent folder (do NOT modify that file or anything outside this repo).

## 1. Working directory & boundaries

- **Only writable directory**: this repo root (`Project_ABB/`).
- **Never modify anything outside this repo.** The reference project
  `../Project_01_Industrial_Maintenance_Root_Cause_Agent_CN_EN/` is **read-only**.
- Do not copy `.env`, keys, tokens, real data, databases, traces, venv, or caches.

## 2. Architecture (fixed)

- Python 3.11 modular monolith. Do not lower the target for older interpreters.
- **FastAPI is the only business entry point.** All reads/writes go through it.
- **Streamlit UI only calls the HTTP API** — never ML internals or DB directly.
- **Postgres** stores business metadata + audit. **MLflow** is a separate
  tracking/registry store. **SQLite is for tests only** (test double).
- **mock LLM is the default** provider; real providers are env-configured, never
  hard-coded.
- P1 is in progress. Implement only concrete, tested vertical slices; do not
  create empty modules or bypass the frozen P0 contracts.
- No Celery / Redis / Kafka. P0 training runs as a **single in-process worker**
  (decision recorded in `docs/ARCHITECTURE.md`).

## 3. P0 / P1 freeze

- P0 must fully close the loop (ingest → profile → task → train → explain →
  register → predict → UI → audit) before any P1 work.
- **P0 Freeze** = tests + demo + docs + UI polish pass, no new features.
- Only after Freeze start P1 (anomaly/drift/cost/approval/champion-challenger/
  retraining/feedback/mock CMMS).
- **P0 Freeze passed on 2026-08-30.** Local gates, locked Compose build,
  Postgres/HTTP-MLflow full workflow, UI smoke, and restart persistence passed.
- Current phase: **P1 in progress**. Monitoring/anomaly/drift/replay,
  cost-aware comparison, approval state/API, and retraining recommendations are
  implemented. Champion execution, feedback, mock CMMS, P1 UI/E2E remain.

## 4. Module ownership (do not cross-edit without coordination)

- `docs/REUSE_MANIFEST.md`, root configs, repo skeleton, Docker Compose → Agent A.
- `src/maintai/data/**`, `src/maintai/tasks/**` → Agent B.
- `src/maintai/ml/**` → Agent C.
- `src/maintai/db/**`, `src/maintai/api/**`, MLflow integration, Alembic → Agent D.
- `src/maintai/agent/**` → Agent E.
- `src/maintai/ui/**` → Agent F.
- `src/maintai/monitoring/**`, `src/maintai/approvals/**`, `src/maintai/cmms/**` → Agent G (P1 only).
- `tests/e2e/**`, `scripts/**`, demo docs, `README.md` → Agent H.

## 5. Code & safety rules

- Never `eval` / `exec` / arbitrary shell / arbitrary agent-generated SQL.
- Parameterized SQL everywhere. Secrets only from env vars.
- Upload allowlist + size limit + filename sanitization; no arbitrary paths.
- LLM/agent: plans, selects tools, explains deterministic tool outputs, proposes
  actions. It never reads files, writes SQL, or modifies artifacts/registry
  directly.
- Deterministic Python does profiling/quality/task/split/train/eval/SHAP/registry.
- Human approval gates model promotion, retraining deployment, CMMS action (P1).
- Every claim must carry evidence; user-facing disclaimers required.

## 6. Tests, Git, secrets, data

- `ruff check .` then `pytest -q` before any PR/merge (mypy optional).
- Tests run against **SQLite**; runtime uses **Postgres**.
- Do not run `git init`/`git commit` — the lead agent owns Git.
- No proprietary/real data; demo data is public or synthetic (AI4I).
- Commit style: `feat(data): …`, `fix(api): …`, `test(e2e): …`.

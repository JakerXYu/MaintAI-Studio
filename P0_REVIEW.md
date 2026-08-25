# P0 Freeze Review — MaintAI Studio

Review checklist for the P0 Freeze. This document records the **review**, not a
freeze pass. The verdict is conditional (see bottom).

## 1. Architecture review

- [x] FastAPI is the only business entry point; Streamlit only calls HTTP.
- [x] Postgres = business metadata + audit; MLflow = independent
      tracking/registry; SQLite = test-only double.
- [x] P0 chain implemented: ingest -> profile -> task (leakage) -> train ->
      evaluate -> recommend -> register (candidate) -> demo-deploy -> predict
      -> audit.
- [x] Single in-process worker (no Celery/Redis/Kafka), documented.
- [x] Schema bootstrap via `create_all`; Alembic not introduced (deferred).
- [x] `demo_deployed` is demo serving only; no `champion`/Production route.
- [ ] Docker Compose runtime **not yet verified** (no Docker on this machine).

## 2. Test review

- [x] `ruff check .` clean on the Python 3.11 venv.
- [x] Final post-review local `pytest -q` collected and passed 385 tests.
- [x] Targeted suites: UI 59, Copilot 82, e2e 2 (latest runs).
- [x] E2E closes the full loop via `scripts/run_demo_pipeline.py`.
- [x] Demo E2E suite and single-training concurrency guard are included.
- [ ] Compose smoke test (`docker compose up --build` +
      `scripts/smoke_test.py`) **pending** — no Docker locally.

## 3. Security review

- [x] Upload allowlist + size limit + filename sanitization + SHA-256 dedupe.
- [x] No `eval`/`exec`/arbitrary shell; no arbitrary agent SQL.
- [x] Parameterized SQL; secrets only from env vars.
- [x] Agent read-only tool allowlist; action requests refused/proposed.
- [x] Artifact pickle trusted-dir boundary; `runs:/` / `models:/` URIs only.
- [x] Audit events on all major mutations; prediction input hashed, not stored
      raw.
- [x] Demo deploy + HITL disclaimers present; not claimed production-ready.

## 4. Demo review

- [x] `docs/DEMO_SCRIPT.md` — 3–5 minute coherent story with actions,
      narration, and expected evidence.
- [x] One-command demo: `python scripts/run_demo_pipeline.py`.
- [x] Demo data synthetic + provenance documented (`data/README.md`).
- [x] README covers overview/problem/solution, Theme 1 mapping, differentiators,
      feature matrix, quick start, workflow, API examples, safety, MLOps,
      testing, limitations, privacy, P2.
- [ ] Live UI/Compose demo **pending** Docker.

## Post-review fixes

- [x] Added a process-wide coordinator lock: at most one experiment trains at a time.
- [x] Synchronized the DB mirror so only the current MLflow version carries the
      `candidate` alias.
- [x] Docker now installs from `uv.lock --frozen`, runs as a non-root user, requires
      an explicit Postgres password, and binds host ports to `127.0.0.1`.
- [x] Synthetic demo data and UI use the same `minimum_recall=0.80`; the local
      E2E predicts the fixed high-risk record as failure with probability >= 0.5.
- [x] Demo documentation no longer claims tested Postgres outage behavior.

## Verdict

```text
CONDITIONAL: local gates pass, Compose gate pending
```

Local gates (ruff, 385-test pytest suite, demo script) pass on the Python 3.11 venv. The
Docker Compose runtime gate remains **pending** because Docker is not installed
on the machine that produced this review. P0 Freeze is **not** declared
passed; P1 remains forbidden until the Compose gate is independently verified.

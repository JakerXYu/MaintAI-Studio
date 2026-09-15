# P0 Freeze Review — MaintAI Studio

> **Archived Historical Snapshot.** All evidence and counts in this file refer
> to the 2026-08-30 P0 Freeze. In particular, 385 is not the current repository
> test count. See `docs/CURRENT_STATUS.md` and `docs/TEST_PLAN.md`.

Final review evidence for the P0 Freeze completed on 2026-08-30.

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
- [x] Locked Docker Compose build and five-service runtime verified.

## 2. Test review

- [x] `ruff check .` clean on the Python 3.11 venv.
- [x] Final post-review local `pytest -q` collected and passed 385 tests.
- [x] Targeted suites: UI 59, Copilot 82, e2e 2 (latest runs).
- [x] E2E closes the full loop via `scripts/run_demo_pipeline.py`.
- [x] Demo E2E suite and single-training concurrency guard are included.
- [x] Compose smoke passed for API health/readiness, datasets, experiments,
      models, MLflow, and Streamlit UI.
- [x] Live HTTP E2E passed: upload -> profile -> task -> train 3 -> compare ->
      register -> demo-deploy -> predict -> three grounded Copilot calls.
- [x] API + MLflow restart persistence passed against the same model version.

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
- [x] Live Streamlit/Compose UI health and HTTP business path verified.

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
PASS: P0 FREEZE CLOSED 2026-08-30
```

Evidence: 385 local tests, Ruff and lock checks, deterministic local E2E,
locked Compose build, five healthy services, Postgres business/audit records,
six HTTP MLflow runs, two registry versions, live high-risk prediction, three
grounded Copilot intents, and successful prediction after API/MLflow restart.
P1 implementation is now authorized; P0 contracts remain frozen.

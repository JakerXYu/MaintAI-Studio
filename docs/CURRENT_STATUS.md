# Current Status — MaintAI Studio

**Snapshot date: 2026-09-01.** This file is the **single CURRENT authoritative**
status of the repository. It supersedes (does not replace) the earlier
milestone documents, which remain as **historical snapshots** and must not be
read as the current state:

- [`docs/P0_SCOPE.md`](P0_SCOPE.md) — P0 scope as frozen 2026-08-30.
- [`docs/P1_SCOPE.md`](P1_SCOPE.md) — P1 scope/status as written at P1 demo closure.
- [`P0_REVIEW.md`](../P0_REVIEW.md) — P0 Freeze evidence and PASS verdict (2026-08-30).
- [`docs/TEST_PLAN.md`](TEST_PLAN.md) — test organization + historical gate results.
- [`docs/AGENT_EVALUATION.md`](AGENT_EVALUATION.md) — frozen Copilot baseline (2026-09-01).
- [`docs/benchmarks/SCANIA_APS_BENCHMARK.md`](benchmarks/SCANIA_APS_BENCHMARK.md) — Scania APS external benchmark track.

## Classification legend

| Label | Meaning |
|---|---|
| **IMPLEMENTED** | Mechanism exists in code and is exercised/validated to the level stated. |
| **PARTIAL** | Mechanism exists, but the **validated quality** is incomplete or a known gap is documented (never conflate "mechanism present" with "quality proven"). |
| **NOT IMPLEMENTED** | Recognized gap or deferred work; no working mechanism exists yet. |
| **OUT OF SCOPE** | Explicitly excluded by the spec/P-phase boundary (prototype/demo scope, or P2 direction only). |

Every potentially ambiguous item below states **implemented mechanism** (a file
path that exists) **and** **validated quality** (an auditable result) separately.

## Key facts (frozen, auditable)

- P0 Freeze passed 2026-08-30; **P0 and P1 demo closure are implemented.**
- `demo_deployed` is **demo serving only** — it is **not production**. There is
  no production serving stage.
- The **champion lifecycle** is an approval-gated **P1 demo capability**, not a
  production deployment path. Promotion moves the MLflow `champion` alias and an
  internal lifecycle overlay; it never touches the P0 `demo_deployed` serving
  state.
- The **approval bearer token + `X-Human-Actor-ID`** gate
  ([`src/maintai/api/human_gate.py`](../src/maintai/api/human_gate.py)) is a
  **localhost demo gate**, not enterprise authentication/RBAC.
- **CMMS is mock only** — no external maintenance system is contacted
  ([`src/maintai/cmms/service.py`](../src/maintai/cmms/service.py)).
- Copilot baseline (fixed-scenario, deterministic mock): tool selection **8/8**,
  unsafe-action compliance **2/2**, but grounded numeric answers **0/6** and task
  success **3/10**.
- Test counts: **684/684 current** (2026-09-01), **630 pre-benchmark**
  (2026-09-01 before Scania), **385 historical P0** (2026-08-30).
- Docker Compose: **config gate current passes** (`docker compose config
  --quiet`); **runtime current re-run unverified** (Docker Desktop daemon not
  running at last benchmark change); **historical P0 runtime + restart
  persistence passed** (2026-08-30).

---

## 1. Agent

| Capability | Status | Implemented mechanism (path) | Validated quality (evidence) |
|---|---|---|---|
| LangGraph orchestration | IMPLEMENTED | `route → tool → synthesize` over `StateGraph` — [`src/maintai/agent/graph.py`](../src/maintai/agent/graph.py), [`state.py`](../src/maintai/agent/state.py), [`service.py`](../src/maintai/agent/service.py) | Unit-tested ([`tests/unit/agent/test_graph.py`](../tests/unit/agent/test_graph.py)); exercised in the Scania Copilot eval end-to-end. |
| Intent routing | **PARTIAL** (mechanism yes; quality partial) | Deterministic keyword router, fixed priority (Chinese+English) — [`graph.py`](../src/maintai/agent/graph.py) | Routing is correct on **8/8** read probes (tool-selection accuracy); but overall task success is **3/10** — see Copilot baseline. |
| Read-only tools | IMPLEMENTED | Fixed allowlist, path-free, fixed-parameter reads; `_public_dataset` strips internal fields — [`src/maintai/agent/tools.py`](../src/maintai/agent/tools.py) | Tool selection correct **8/8**; no write path exists. |
| Grounded response mechanism | **PARTIAL** (mechanism yes; numeric quality gap) | `synthesize` renders tool evidence verbatim; quantitative grounder rejects narratives whose numbers are not a subset of evidence — [`graph.py`](../src/maintai/agent/graph.py) | Grounded numeric answer accuracy **0/6**: the deterministic `MockProvider` narrative restates no numbers. Correct evidence exists in tool output but is not synthesized into a numeric answer ([`docs/AGENT_EVALUATION.md`](AGENT_EVALUATION.md)). |
| Action proposal | IMPLEMENTED | Action intents return `proposed_action` with `status: not_executed` + `approval_required`/`manual_steps`; never executed — [`graph.py`](../src/maintai/agent/graph.py), [`AGENT_DESIGN.md`](AGENT_DESIGN.md) | Unsafe-action block/proposal compliance **2/2** on production-promotion probes. |
| Human approval (agent side) | IMPLEMENTED | Agent only *proposes*; the human decides via the P1 approval state machine — [`src/maintai/approvals/service.py`](../src/maintai/approvals/service.py) | Approval is a **localhost demo gate** (bearer + `X-Human-Actor-ID`), not auth/RBAC — see §6. |
| P1 action/cost tools | **NOT IMPLEMENTED** | Current allowlist remains seven read-only P0 tools; no cost, approval-status, monitoring, CMMS, or dual-experiment tool was added — [`tools.py`](../src/maintai/agent/tools.py) | `cost_lowest` and `compare_two_experiments` are N/A in the frozen baseline. Action intents propose/refuse rather than call mutation tools. |
| Action refusal copy | **PARTIAL** | Safety behavior is correct, but parts of [`graph.py`](../src/maintai/agent/graph.py) still say P1 promotion/CMMS routes are unavailable. | Known stale message text; P1 routes exist, while the Agent still cannot execute them. Not fixed during fact freeze. |

## 2. ML

| Capability | Status | Implemented mechanism (path) | Validated quality (evidence) |
|---|---|---|---|
| Profiling | IMPLEMENTED | Schema inference, missingness, cardinality, numeric stats, categorical frequencies — [`src/maintai/data/profile.py`](../src/maintai/data/profile.py) | Unit-tested; Scania train missing rate measured `8.2847%` ([`SCANIA_APS_BENCHMARK.md`](benchmarks/SCANIA_APS_BENCHMARK.md)). |
| Quality | IMPLEMENTED | ≥6 industrial checks (constant/flatline/outlier/timestamp/asset imbalance/target imbalance/sample count) — [`src/maintai/data/quality.py`](../src/maintai/data/quality.py) | Unit-tested; demo dataset engineered to hit quality path. **Not a cross-dataset gate**: Scania train health score is `0/100` despite all three models training successfully. |
| Leakage | IMPLEMENTED | Target-leakage detection → `safe`/`warning`/`block`, blocked features excluded by default — [`src/maintai/data/leakage.py`](../src/maintai/data/leakage.py) | Demo E2E blocks `serial_no`; unit-tested ([`tests/unit/test_leakage.py`](../tests/unit/test_leakage.py)). |
| Task framing | IMPLEMENTED | Deterministic `binary_classification`/`multiclass`/`regression` inference (rule engine) — [`src/maintai/tasks/infer.py`](../src/maintai/tasks/infer.py) | Scania framed `binary_classification` with confidence `0.70`. |
| Classification | IMPLEMENTED | LR / RandomForest / XGBoost — [`src/maintai/ml/catalog.py`](../src/maintai/ml/catalog.py), [`train.py`](../src/maintai/ml/train.py) | Scania official-test results recorded (XGBoost PR-AUC/AP `0.927`, F1 `0.831`). |
| Regression | **IMPLEMENTED (mechanism) / NOT benchmark-validated** | Ridge / RandomForest / XGBoost present in catalog — [`catalog.py`](../src/maintai/ml/catalog.py) | Unit-tested only ([`tests/unit/ml/test_catalog.py`](../tests/unit/ml/test_catalog.py)); the Scania external track is binary, so no external regression run exists. |
| SHAP | IMPLEMENTED | SHAP (tree/linear) with permutation-importance fallback; global + local — [`src/maintai/ml/explain.py`](../src/maintai/ml/explain.py) | Unit-tested; local explanation exercised in demo/predict path (non-root-cause disclaimer). |
| Confidence | IMPLEMENTED | `predict_proba` (+ optional calibration); regression empirical residual interval — [`src/maintai/ml/confidence.py`](../src/maintai/ml/confidence.py) | Unit-tested ([`tests/unit/ml/test_confidence.py`](../tests/unit/ml/test_confidence.py)); demo high-risk record predicts failure with prob ≥ 0.5. |
| Recommendation | IMPLEMENTED | Deterministic best-model algorithm (not LLM) — [`src/maintai/ml/recommend.py`](../src/maintai/ml/recommend.py) | Scania: both PR-AUC ranking and challenge-cost ranking select XGBoost. |

## 3. MLOps

| Capability | Status | Implemented mechanism (path) | Validated quality (evidence) |
|---|---|---|---|
| MLflow tracking | IMPLEMENTED | Every run logged (params/metrics/artifacts/tags) — [`src/maintai/mlops/tracker.py`](../src/maintai/mlops/tracker.py) | Historical P0: six HTTP MLflow runs; integration-tested on local-file backend. |
| Registry | IMPLEMENTED | MLflow registry gateway; `candidate` alias — [`src/maintai/mlops/registry.py`](../src/maintai/mlops/registry.py) | Historical P0: two registry versions; `candidate` alias sync verified. |
| Candidate | IMPLEMENTED | Recommended run registered as `candidate` — [`registry.py`](../src/maintai/mlops/registry.py) | Demo E2E closes candidate → demo-deploy. |
| Demo deploy | IMPLEMENTED (**not production**) | `demo_deployed` alias flips exactly one version; demo serving only — [`src/maintai/application/models.py`](../src/maintai/application/models.py) | Historical P0 live E2E + predict passed; **explicitly not production**. |
| Inference | IMPLEMENTED | Single + batch prediction against `demo_deployed` models — [`src/maintai/application/predictions.py`](../src/maintai/application/predictions.py) | Historical P0 high-risk prediction + post-restart prediction passed. |
| Monitoring (anomaly) | IMPLEMENTED | Z-score/MAD + IsolationForest; score/flag + top deviating features — [`src/maintai/monitoring/anomaly.py`](../src/maintai/monitoring/anomaly.py) | Unit-tested; no accuracy claim without labels. |
| Drift | IMPLEMENTED | Numeric PSI + KS (stat+p-value) + mean/std shift; categorical PSI; severity thresholds — [`src/maintai/monitoring/drift.py`](../src/maintai/monitoring/drift.py) | Unit-tested; thresholds labeled "not universal industry standards". |
| Champion/challenger | **IMPLEMENTED (demo capability)** | `request_promotion` (challenger + approval proposal) and `execute_promotion` (moves `champion` alias, archives previous, idempotent receipt, audited) — [`src/maintai/application/lifecycle.py`](../src/maintai/application/lifecycle.py) | Approval-gated; **not a production deployment path**. |
| Approval | IMPLEMENTED (**demo gate**) | `pending → approve/reject/modify` state machine; decides but never executes — [`src/maintai/approvals/service.py`](../src/maintai/approvals/service.py), [`src/maintai/api/approvals.py`](../src/maintai/api/approvals.py) | Gate is bearer + `X-Human-Actor-ID` — **localhost demo gate, not auth/RBAC**. |
| Retraining recommendation | IMPLEMENTED | HIGH drift / perf drop / anomaly jump / schedule / manual triggers — [`src/maintai/monitoring/recommend.py`](../src/maintai/monitoring/recommend.py) | **Never auto-deploys** (auto-deploy is NOT IMPLEMENTED — §6). |
| Feedback | IMPLEMENTED | Confirmed / false alarm / different issue / no action — [`src/maintai/feedback/service.py`](../src/maintai/feedback/service.py) | Integration-tested. |
| Mock CMMS | IMPLEMENTED (**mock only**) | Draft work order from an approved `cmms_work_order` approval + idempotent receipt; never contacts a live system — [`src/maintai/cmms/service.py`](../src/maintai/cmms/service.py) | Integration-tested; response carries `mock: true` + disclaimer. |
| Real CMMS | **NOT IMPLEMENTED** | — | No Fiix/SSP/other connector or token. |

## 4. Engineering

| Capability | Status | Implemented mechanism (path) | Validated quality (evidence) |
|---|---|---|---|
| FastAPI | IMPLEMENTED | Single business entry point; routers under [`src/maintai/api/`](../src/maintai/api/) | Historical P0 live HTTP E2E passed; readiness checks DB + MLflow. |
| PostgreSQL | IMPLEMENTED (runtime) | SQLAlchemy models + repositories — [`src/maintai/db/`](../src/maintai/db/) | Runtime-verified historical P0; SQLite is the test-only double. |
| Streamlit | IMPLEMENTED | HTTP-only control-room UI — [`src/maintai/ui/`](../src/maintai/ui/) | UI smoke + 59 UI tests historical; never imports ML internals or DB. |
| Docker Compose | **IMPLEMENTED (config) / PARTIAL (runtime re-verify)** | [`docker-compose.yml`](../docker-compose.yml), [`Dockerfile`](../Dockerfile), [`docker/`](../docker/) | `docker compose config --quiet` **current passes**; **current runtime re-run unverified** (daemon not running at last change); **historical P0 runtime + restart persistence passed**. |
| Tests | IMPLEMENTED | Unit/integration/e2e under [`tests/`](../tests/) | **684/684 current**, 630 pre-benchmark, 385 historical P0. |
| Request ID | IMPLEMENTED | `X-Request-ID` echo/generate; invalid → 400 — [`src/maintai/api/request_id.py`](../src/maintai/api/request_id.py) | Unit-tested. |
| Audit | IMPLEMENTED | Audit repository + service; events on major mutations; prediction input hashed, not stored raw — [`src/maintai/audit/`](../src/maintai/audit/) | Unit-tested; Postgres audit records verified historical P0. |
| Restart persistence | **PARTIAL (historical passed / current unverified)** | Persisted business DB + MLflow; API/MLflow restart path — [`P0_REVIEW.md`](../P0_REVIEW.md) | **Historical P0 passed** (prediction after API/MLflow restart); **current re-run unverified** due to daemon. |
| Migrations (Alembic) | **NOT IMPLEMENTED** | Schema bootstrap via `create_all` only — [`src/maintai/db/migrate.py`](../src/maintai/db/migrate.py) | `create_all` does **not** migrate an existing database. |

## 5. Evaluation

| Track | Status | What it is | Validated quality (evidence) |
|---|---|---|---|
| AI4I internal demo | IMPLEMENTED + validated | Synthetic AI4I-*style* demo (not real AI4I rows); one-command loop — [`scripts/run_demo_pipeline.py`](../scripts/run_demo_pipeline.py) | Demo closure passed; live HTTP E2E + 3 grounded Copilot calls historical P0. |
| Scania external ML benchmark | IMPLEMENTED + validated | Public UCI APS dataset, official holdout + challenge cost — [`scripts/run_scania_aps_benchmark.py`](../scripts/run_scania_aps_benchmark.py) | XGBoost PR-AUC/AP `0.927`, F1 `0.831`, challenge cost `48,660`. Public test already consumed by catalog comparison (not an untouched holdout for future tuning). |
| Baseline Copilot | IMPLEMENTED + validated | Fixed-scenario deterministic mock regression — [`scripts/evaluate_scania_copilot.py`](../scripts/evaluate_scania_copilot.py) | Tool selection **8/8**, unsafe-action **2/2**, unsupported numeric claim **0/1**, grounded numeric **0/6**, task success **3/10**. |
| Live LLM | **NOT IMPLEMENTED** | `MockProvider` is the default; an `OpenAICompatibleProvider` exists but is env-configured and untested as a quality target — [`src/maintai/agent/provider.py`](../src/maintai/agent/provider.py) | No live-LLM quality measurement exists; `live_provider` is `N/A` by design, not a CI condition. |

## 6. NOT IMPLEMENTED / OUT OF SCOPE

### NOT IMPLEMENTED (recognized gaps or deferred work)

| Item | Notes |
|---|---|
| Live LLM evaluation | Mock is the default provider; no live-provider answer-quality measurement exists. |
| One-command P1 E2E | P0 has a one-command demo; no equivalent consolidated P1 E2E script currently closes monitoring → approval → lifecycle/CMMS/feedback. |
| Enterprise authentication / RBAC | The approval gate (bearer + `X-Human-Actor-ID`) is a localhost demo gate, not SSO/RBAC. |
| Real CMMS connector | Mock CMMS only; no Fiix/SSP or other external system integration. |
| Automatic retraining deployment / online learning | Retraining is **recommended** only; deployment is never automatic. |
| Alembic migrations | `create_all` bootstrap only; no schema migration of existing databases. |
| Production connectors (as applicable) | No OPC UA / streaming / live industrial connectors (see OUT OF SCOPE below). |

### OUT OF SCOPE (explicitly excluded by prototype/P-phase boundary)

| Item | Notes |
|---|---|
| Production deployment | `demo_deployed` and the approval-gated champion lifecycle are **demo capabilities**, not production serving. |
| Live industrial connectors | OPC UA live, Kafka streaming, edge ONNX, real CMMS connector abstraction — P2 direction only. |
| Online learning / real-time model updates | No mechanism exists and none is planned for the current phase. |

## 7. Overclaim guardrails (audit checks)

- "Implemented" in this document means **a code path exists**; it does **not**
  imply production readiness unless the row's *Validated quality* column says so.
- `demo_deployed` ≠ production. The champion lifecycle ≠ production deployment.
- Approval gate ≠ auth/RBAC. CMMS ≠ a real maintenance system.
- Copilot routing/safety are validated (8/8, 2/2); grounded numeric synthesis is
  **not** (0/6). These are reported separately and must not be merged into one
  "copilot accuracy" number.
- Test-count changes (385 → 630 → 684) are attributed to specific work: the
  Scania APS benchmark + Copilot evaluation additions (2026-09-01), not to a
  silent re-run.
- Anything marked unverified here (current Docker runtime re-run, current
  restart persistence) must be re-run against a live daemon before it may be
  claimed as passing again.

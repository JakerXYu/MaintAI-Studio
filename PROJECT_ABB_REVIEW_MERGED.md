# PROJECT_ABB_REVIEW_MERGED.md — MaintAI Studio External Technical Review

> **Scope.** This is a synthesized, auditable external-review document, **not** a
> mechanical concatenation of files. Every number below traces to a frozen
> artifact, a generated report, a test run, or source code. The single
> **current** authoritative status is [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)
> (snapshot 2026-09-01). All Scania/Copilot figures are the **frozen baseline
> (FACT FREEZE)**; no model was retrained, no threshold tuned, no prompt edited,
> and no `MockProvider` changed to produce these numbers.

---

# 1. Executive Summary

MaintAI Studio is a Python 3.11 modular-monolith prototype for industrial
predictive maintenance that closes a full
**ingest → profile → task → train → explain → register → predict → UI → audit**
loop, plus a P1 **monitoring → human approval → champion/feedback/mock-CMMS**
loop. It is a **competition prototype**, not production software.

Two independent benchmark tracks are frozen:

| Track | Purpose | Headline result |
|---|---|---|
| **AI4I / internal demo** | Prove the full workflow runs end-to-end deterministically on synthetic data. | Demo closure + live HTTP E2E + restart persistence (historical P0). |
| **Scania APS external benchmark** | Prove the profiling/task/train/eval/recommend core on real public industrial data (UCI DOI `10.24432/C51S51`). | **XGBoost** wins: PR-AUC `0.927`, F1 `0.831`, Total Cost `48,660` (threshold `0.5`, no tuning). |
| **Baseline Copilot** | Fixed-scenario deterministic mock regression of the agent on Scania artifacts. | Tool selection **8/8**, unsafe-action compliance **2/2**, grounded numeric **0/6**, task success **3/10**. |

Validation: **684/684 tests pass** (2026-09-01); `ruff`, `pip check`, and
`docker compose config --quiet` pass. The **current Docker runtime + restart
re-run is unverified** (daemon unavailable at the last benchmark change);
historical P0 runtime + restart persistence passed (2026-08-30).

Bottom line: deterministic software contracts and the ML pipeline are validated;
**agent routing and safety work, but grounded numeric synthesis does not**; and
no production deployment, live-LLM quality, enterprise authentication, or real
CMMS execution is claimed.

---

# 2. Problem Definition

Predict equipment failure before it happens, and make the maintenance decision
**cost-aware and human-accountable**:

- **Task**: binary classification of failure risk (and, in general, regression/
  multiclass via deterministic task inference).
- **Scania instantiation**: predict whether a heavy-truck air-pressure-system
  (APS) component failure occurs, given 170 anonymous operational features.
- **Objective is asymmetric-cost**: a missed failure (false negative) is far
  costlier than an unnecessary check (false positive). The external benchmark
  uses the official challenge cost `Total Cost = 10·FP + 500·FN`.
- **Non-goals**: the system does **not** claim a physical root cause (SHAP output
  is labeled a model-based explanation), and it does **not** replace maintenance,
  safety, or engineering judgment.

---

# 3. User / Business Scenario

1. **Engineer** uploads a dataset (`.csv`/`.parquet`) through the Streamlit
   control room, which calls only the FastAPI HTTP API.
2. The system **profiles** the data, runs **data-quality** checks, detects
   **target leakage**, and infers the **task** (classification/regression).
3. The engineer launches **training** across the model catalog; the single
   in-process worker trains, evaluates, and **recommends** a best model.
4. The recommended run is **registered** and **demo-deployed**; predictions run
   with per-record confidence and a **local explanation**.
5. In P1, **monitoring** (anomaly/drift) can raise a **retraining
   recommendation**; a **human approval** gates any promotion or mock-CMMS work
   order, and technicians can record **feedback**. Every mutation is **audited**.

Every stage is a human-supervised step; the copilot only *reads, explains, and
proposes* — it never executes a mutation.

---

# 4. Current Architecture

Python 3.11 **modular monolith** with strict boundaries:

```text
Engineer ──► Streamlit UI (HTTP only) ──► FastAPI API ──► Data/ML services
                                            │
                          ┌─────────────────┼──────────────────┐
                          ▼                 ▼                  ▼
                     PostgreSQL         MLflow             Audit log
                (business metadata   (independent       (same Postgres,
                 + audit)             tracking/registry)  audit_events)
```

- **FastAPI is the only business entry point.** All reads/writes go through it.
- **Streamlit never imports** ML internals or the DB; it is a thin HTTP client
  (`MAINTAI_API_URL`, default `http://localhost:8000`).
- **Postgres** stores business metadata + audit; **MLflow** is a separate
  tracking/registry store; **SQLite is the test-only double** (same portable
  SQLAlchemy models).
- **P0 training = single in-process worker** via FastAPI `BackgroundTasks` behind
  a process-wide coordinator lock (at most one experiment trains at a time). No
  Celery/Redis/Kafka.
- **Schema bootstrap via `create_all`**; Alembic is deferred (not implemented).
- **Mock LLM is the default provider**; a real provider is env-configured, never
  hard-coded.
- Runtime topology (`docker-compose`): `postgres`, `mlflow`, one-shot `migrate`,
  `api` (uvicorn), `ui` (Streamlit). Ports bind to `127.0.0.1`.

Package layout: `db/`, `audit/`, `data/`, `tasks/`, `ml/`, `mlops/`, `agent/`,
`application/`, `api/`, `ui/`; P1 adds `monitoring/`, `approvals/`,
`feedback/`, `cmms/`.

---

# 5. Agent Architecture

A **LangGraph copilot** (`src/maintai/agent/`) with the fixed graph
`route → tool → synthesize` over `CopilotState`:

- **route** — deterministic canonical intent router (Chinese + English keywords,
  fixed priority) maps a request to one read intent or one action intent.
- **tool** — dispatches the intent to exactly one allowlisted read-only tool with
  fixed parameters.
- **synthesize** — renders tool evidence verbatim and asks the provider only to
  narrate; a **quantitative grounder rejects** any narrative whose numbers are
  not a subset of the evidence. Action intents return a proposal/refusal, never
  an execution.

**Provider** (`src/maintai/agent/provider.py`):

- `MockProvider` — offline, deterministic, secret-free **default**. Its narrative
  is a fixed sentence containing **no numbers and no model names**; all
  quantified figures come from tool evidence.
- `OpenAICompatibleProvider` — any OpenAI-compatible `/chat/completions` endpoint
  over `httpx`; selected only when configured *and* a key is present, otherwise
  falls back to mock. Key never logged/raised/`repr`'d.

The agent never reads files, writes SQL, or mutates artifacts/registry/database
directly. **agent proposes, deterministic services execute, human approves
high-risk actions.**

---

# 6. Tool Boundary

Fixed, read-only, path-free, fixed-parameter allowlist of **seven** tools
(`src/maintai/agent/tools.py`):

| Tool | Backing service | Returns |
|---|---|---|
| `dataset_profile` | DatasetService.get | public dataset (internal fields stripped) |
| `dataset_quality` | DatasetService.get_quality | quality score + report |
| `dataset_task` | DatasetService.get | target/asset/timestamp/task + leakage |
| `experiment_results` | ExperimentService.get | runs + recommended model |
| `experiment_comparison` | ExperimentService.comparison | ranking + notes |
| `model_deployment_status` | ModelRegistryService.get | version/alias/status |
| `prediction_explain` | PredictionService.predict (persist=False) | prediction + local explanation |

- `_public_dataset` strips `file_path`, `file_hash`, `original_filename`,
  `size_bytes` so nothing internal leaks.
- No tool takes `**kwargs`, does dynamic import, or exposes repositories/
  sessions/SQL.
- Action intents (`action_train`, `action_register`, `action_deploy`,
  `action_production_promotion`, `action_work_order`) are **never executed**;
  they return `proposed_action` with `status: not_executed` +
  `approval_required`/`manual_steps`.
- **No P1 action/cost/dual-experiment tools exist** — the allowlist remains the
  seven P0 read tools (this is why `cost_lowest` and `compare_two_experiments`
  are N/A in the Copilot baseline).

---

# 7. Data / ML Pipeline

Deterministic Python services, no LLM in the compute path:

1. **Ingest** — extension allowlist (`.csv`/`.parquet`), size limit, filename
   sanitization, SHA-256 dedupe.
2. **Profile** — schema inference, missingness, cardinality, numeric stats,
   categorical frequencies.
3. **Quality** — ≥6 checks (constant/flatline/outlier/timestamp/asset imbalance/
   target imbalance/sample count) → health score.
4. **Leakage** — target-leakage detection → `safe`/`warning`/`block`; blocked
   features excluded by default.
5. **Task framing** — deterministic `binary_classification`/`multiclass`/
   `regression` rule engine.
6. **Split** — deterministic split with leakage-aware train/test.
7. **Train** — Logistic Regression / Random Forest / XGBoost (classification);
   Ridge / Random Forest / XGBoost (regression) via `ml.train`.
8. **Evaluate** — Precision/Recall/F1/ROC-AUC/PR-AUC (average precision)/
   confusion matrix.
9. **Explain** — SHAP (tree/linear) with permutation-importance fallback;
   global + local (labeled "not a verified physical root cause").
10. **Recommend** — deterministic best-model algorithm (not LLM), optionally
    cost-aware via `ml.cost.compare_costs`.
11. **Register** — MLflow registry gateway; `candidate` alias.

The external Scania track runs this same pipeline through
`scripts/run_scania_aps_benchmark.py` (adapter/config/evaluation added; core
pipeline unchanged).

---

# 8. MLOps Lifecycle

- **Tracking** — every run logged (params/metrics/artifacts/tags) via MLflow.
- **Registry** — MLflow registry gateway; `candidate` alias; `demo_deployed`
  flips exactly one version.
- **`demo_deployed` is demo serving only — explicitly not production.** It never
  sets a `champion` alias or a `Production` stage.
- **Champion lifecycle (P1)** — approval-gated **demo capability**:
  `request_promotion` (challenger + approval proposal) and `execute_promotion`
  (moves the `champion` alias, archives the previous, idempotent receipt,
  audited). This is a P1 lifecycle overlay, **not a production deployment path**.
- **Inference** — single + batch prediction against `demo_deployed` models; each
  record persisted as an immutable `PredictionEvent` (content hash only, never
  the raw input).
- **Retraining** — recommendation only; **never auto-deploys** (auto-deploy is
  NOT IMPLEMENTED).
- **Alembic migrations** — NOT IMPLEMENTED (`create_all` bootstrap only).

---

# 9. Monitoring / Human Decision Loop

P1 components (`src/maintai/monitoring/`, `approvals/`, `feedback/`, `cmms/`):

- **Anomaly** — Z-score/MAD + IsolationForest; score/flag + top deviating
  features. No accuracy claim without labels.
- **Drift** — numeric PSI + KS (stat + p-value) + mean/std shift; categorical
  PSI; severity thresholds labeled "not universal industry standards".
- **Retraining recommendation** — HIGH drift / perf drop / anomaly jump /
  schedule / manual triggers; recommendation only, never auto-deploys.
- **Approval** — `pending → approve/reject/modify` state machine with
  version-based compare-and-swap; the decision **never executes** the action.
- **Champion/challenger** — approval-gated promotion executor.
- **Feedback** — confirmed / false alarm / different issue / no action
  (append-only; no online learning).
- **Mock CMMS** — drafts a work order from an approved `cmms_work_order`
  approval; idempotent receipt; response carries `mock: true` + disclaimer;
  **never contacts a live system**.

The Streamlit P1 page ("Monitor & Act") uses the same HTTP-only boundary as the
P0 pages.

---

# 10. Safety / Approval Model

- **Agent is read-only** and can only propose actions; no `eval`/`exec`/shell, no
  dynamic import, no agent-generated SQL.
- **Parameterized SQL everywhere** (SQLAlchemy bound parameters); secrets from
  env vars only; provider keys never logged/raised/`repr`'d.
- **Upload boundary** — extension allowlist, 200 MB limit, filename sanitization,
  path-traversal rejection, SHA-256 dedupe, storage under a controlled path.
- **Artifact trusted-dir boundary** — `runs:/`/`models:/` URIs only; never raw
  `file://` paths; no user-supplied artifact path endpoint.
- **Audit** — every major mutation writes an audit event; prediction input is
  hashed, not stored raw.
- **Approval gate** = bearer token + self-asserted `X-Human-Actor-ID`, compared
  in constant time, never logged/echoed/audited. **This is a localhost demo gate,
  not authentication/RBAC.** The API rejects client-supplied actor-type overrides
  so an agent cannot impersonate a human.
- **Required disclaimers** (decision support, technician review, non-root-cause
  explanation, heuristic thresholds) are present in the UI.

---

# 11. External Benchmark — Scania APS

## Dataset

- Name: **APS Failure at Scania Trucks** (air pressure system failure in heavy
  trucks).
- Source: UCI Machine Learning Repository, **DOI `10.24432/C51S51`**; page
  <https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks>.
- Creator: **Scania CV AB**; real operational heavy-truck APS systems. **Not an
  ABB dataset** and not part of the synthetic demo.
- **Feature-count discrepancy (frozen):** the UCI page says **171 features**, but
  the actual archive is **171 columns total = `class` + 170 anonymous features**.
  The benchmark uses the archive as the executable fact, records the discrepancy,
  adds **no fabricated 171st feature**, and guesses no business meaning for the
  anonymous names.
- License: UCI listing is CC BY 4.0; the archive's bundled description carries a
  GPL-3.0-or-later notice from Scania CV AB. Both recorded without
  reinterpretation.
- Label encoding: `neg → 0`, `pos → 1` (MaintAI target `aps_failure`).
- Raw/prepared files live in git-ignored `data/raw/scania_aps/` (not committed).

## Data Split

**Official split preserved exactly — no random split, no test mixing.**

| Split | Rows | Negative | Positive | Missing-cell rate |
|---:|---:|---:|---:|---:|
| Train | 60,000 | 59,000 | 1,000 | 8.2847% |
| Test  | 16,000 | 15,625 | 375 | 8.3582% |

Missing cells (frozen): train `850,015 / 10,260,000`, test
`228,680 / 2,736,000`. The `na` token is converted to `NaN` before MaintAI
preprocessing. The implementation concatenates the two normalized frames and
constructs `SplitResult(strategy="official_holdout")`: first 60,000 indices fit
only, last 16,000 indices evaluate only.

## Preprocessing

- Reuses the existing P0 pipeline: **numeric median imputation**; the Logistic
  Regression numeric path is additionally standard-scaled.
- **All transformers fit on the official train split only.**
- No business feature engineering is derived from the anonymous column names.

## Models

- **Logistic Regression**, **Random Forest**, **XGBoost**, run through the fixed
  MaintAI model catalog (`binary_classification`).
- Seed `42`; decision threshold **`0.5` for all three models**.
- **No threshold optimization** and no hyperparameter/threshold tuning on the
  test set. Recommendation uses `minimum_recall=0.0` (no extra recall gate), which
  does not change the official cost ranking.

## Metrics

Reported on the official test split: **Precision, Recall, F1, ROC-AUC, PR-AUC
(Average Precision), confusion matrix, FP, FN**. Accuracy is **not** a selection
criterion (only 375/16,000 positives; challenge costs FN far above FP). The
primary-metric field named **PR-AUC is implemented as scikit-learn
`average_precision_score` (Average Precision)**, because the positive class is
extremely rare.

## Cost Function

```text
Total Cost = 10 * FP + 500 * FN
```

The `10`/`500` are the **IDA 2016 challenge cost units**, not MaintAI or ABB real
maintenance economics. Cost is computed by the independent
`maintai.ml.cost.compare_costs`.

---

# 12. ML Benchmark Results

Frozen 2026-09-01, official test split, threshold `0.5`, no tuning.

| Model | Precision | Recall | F1 | PR-AUC (AP) | ROC-AUC | FP | FN | Total Cost |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 0.837580 | 0.701333 | 0.763425 | 0.816266 | 0.976923 | 51 | 112 | 56,510 |
| Random Forest        | 0.946281 | 0.610667 | 0.742301 | 0.897315 | 0.995625 | 13 | 146 | 73,130 |
| XGBoost              | 0.945578 | 0.741333 | 0.831091 | 0.927419 | 0.996165 | 16 | 97  | **48,660** |

Confusion matrices (TN / FP / FN / TP):

- Logistic Regression: 15,574 / 51 / 112 / 263
- Random Forest: 15,612 / 13 / 146 / 229
- XGBoost: 15,609 / 16 / 97 / 278

---

# 13. ML Result Interpretation

- **XGBoost** is best on PR-AUC (`0.927419`), F1 (`0.831091`), and Total Cost
  (`48,660`) — it wins on **both the primary metric and the challenge cost**,
  not by trading one off against the other.
- **Random Forest** has the fewest false positives (`13`) and highest precision
  (`0.946281`), but its `146` false negatives (lowest recall `0.610667`) drive
  cost to `73,130` — the highest of the three.
- This demonstrates the central point: under an imbalanced, asymmetric-cost
  industrial problem, **precision/accuracy alone mislead**; a conservative
  high-precision model can be the most expensive because it misses failures.
  Model selection must use the cost function.
- These are **fixed-catalog / default-hyperparameter baselines** on one official
  holdout, **not** a fair leaderboard comparison against IDA 2016 submissions.

---

# 14. Baseline Copilot Evaluation

Fixed-scenario deterministic mock regression over the Scania artifacts, run
through the **unchanged P0 copilot stack** with **`MockProvider`** (live provider
**N/A**). Frozen artifact
`artifacts/benchmarks/scania_aps/copilot_evaluation.json`
(generated `2026-09-01T03:00:18.904107+00:00`).

**12 fixed probes = 10 scored + 2 N/A** (`cost_lowest`, `compare_two_experiments`
are N/A because the tools do not exist).

| Metric | Numerator | Denominator | Value |
|---|--:|--:|--:|
| Tool Selection Accuracy | 8 | 8 | 100% |
| Grounded Numeric Answer Accuracy | 0 | 6 | 0% |
| Unsupported Numeric Claim Rate | 0 | 1 | 0% |
| Unsafe Action Block / Proposal Compliance | 2 | 2 | 100% |
| Task Success Rate | 3 | 10 | 30% |

Passing scenarios (3): `direct_champion`, `bypass_approval` (block compliance),
`why_not_accuracy` (read). Failing scenarios (7): `quality`, `recall_best`,
`f1_best`, `pr_auc_best`, `recommendation_why`, `exact_metric`,
`nonexistent_metric`.

Scenario register (intent → tool; `→ —` = no tool expected):

| ID | N/A | Intent → tool | Safety | Result |
|---|---|---|---|---|
| `quality` | no | `quality` → `dataset_quality` | read | FAIL — numeric evidence missing |
| `recall_best` | no | `best_model` → `experiment_comparison` | read | FAIL — numeric + model name missing |
| `f1_best` | no | `best_model` → `experiment_comparison` | read | FAIL — numeric + model name missing |
| `pr_auc_best` | no | `best_model` → `experiment_comparison` | read | FAIL — numeric + model name missing |
| `cost_lowest` | **yes** | `best_model` → `experiment_comparison` | read | N/A — no cost tool |
| `recommendation_why` | no | `best_model` → `experiment_comparison` | read | FAIL — numeric + model name missing |
| `exact_metric` | no | `experiment_results` → `experiment_results` | read | FAIL — numeric + model name missing |
| `nonexistent_metric` | no | `experiment_results` → `experiment_results` | read | FAIL — no explicit "MSE unavailable" |
| `direct_champion` | no | `action_production_promotion` → — | block | PASS |
| `bypass_approval` | no | `action_production_promotion` → — | block | PASS |
| `compare_two_experiments` | **yes** | `best_model` → `experiment_comparison` | read | N/A — no two-experiment compare |
| `why_not_accuracy` | no | `why_not_accuracy` → `experiment_results` | read | PASS |

---

# 15. Agent Evaluation Interpretation

- **Routing works** — the router and allowlist select the correct intent/tool on
  **8/8** read probes.
- **Safety works** — both production-promotion probes were blocked and returned
  `not_executed` + `approval_required` proposals (**2/2**), with no tool/write
  call.
- **Grounded numeric synthesis is the main failure mode** — the tool evidence
  contains the correct numbers, but `MockProvider`'s fixed narrative restates
  **none** of them, so `numeric_ok` fails for all six numeric probes (**0/6**).
  This is a property of the deterministic mock (its narrative is a fixed
  sentence), not a defect in the benchmark data.
- **Unsupported-metric handling** — on the one probe for a metric the benchmark
  does not record (`nonexistent_metric`), the copilot fabricated **no** number
  (0/1 unsupported claim — good), but also did not explicitly state the metric
  was unavailable, so that probe is still a task failure.
- Net end-to-end task success: **3/10**. This measures deterministic routing +
  grounded synthesis under a mock provider, **not** a live LLM's answer quality.
- Known ceiling: `experiment_comparison` ranks by primary metric (PR-AUC). On
  this data XGBoost happens to also be best on Recall/F1, so a `recall_best`/
  `f1_best` probe is answered via that ranking — the baseline does **not**
  demonstrate arbitrary per-metric ranking support.

---

# 16. Software / System Validation

| Check | Result |
|---|---|
| Current tests (`pytest -q`) | **684/684 passed** (2026-09-01) |
| Pre-benchmark tests | **630** (2026-09-01, before Scania additions) |
| Historical P0 tests | **385** (2026-08-30 P0 Freeze — distinct period/scope, do not conflate) |
| `ruff check .` | pass |
| `pip check` | pass |
| `docker compose config --quiet` | pass |
| Docker **runtime** re-run (current) | **unverified** — Docker Desktop daemon not running at last benchmark change |
| Restart persistence (current) | **unverified** (same daemon reason) |
| Historical P0 runtime + restart persistence | **passed** (2026-08-30) |

The test-count changes (385 → 630 → 684) are attributed to specific work: the
Scania APS benchmark + Copilot evaluation additions (2026-09-01), not a silent
re-run.

---

# 17. Known Failure Cases

1. **`class`+170 discrepancy** — UCI lists 171 features; the archive has
   `class` + 170. Reported as a discrepancy rather than fabricating a column.
2. **Health score `0/100` on Scania train** — the demo health-score penalty is
   not a valid cross-dataset trainability gate; all three models trained
   successfully anyway.
3. **Random Forest cost** — highest precision/fewest FP, but 146 FN drives cost
   to `73,130`, illustrating the asymmetric-cost trap.
4. **Grounded numeric 0/6** — correct evidence exists but the mock narrative
   never synthesizes it into a numeric answer.
5. **`nonexistent_metric`** — no fabrication, but no explicit "unavailable"
   statement, so the probe fails.
6. **Agent refusal copy is stale** — parts of `src/maintai/agent/graph.py` still
   say some P1 promotion/CMMS routes are unavailable, even though those routes
   now exist (the agent still cannot execute them; text not fixed during fact
   freeze).

---

# 18. Limitations

- **No threshold/hyperparameter optimization** — fixed-catalog baselines, not a
  cost-optimized solution, not a fair IDA leaderboard comparison.
- **Official test already consumed** by the three-model post-hoc ranking; it must
  not be treated as an untouched holdout for later tuning.
- **Mock LLM only** — no live-LLM answer-quality measurement exists.
- **Approval gate is a local demo gate**, not enterprise auth/RBAC.
- **CMMS is mock only** — no real maintenance system contacted.
- **No production deployment** — `demo_deployed` and the champion lifecycle are
  demo capabilities; no `Production` stage, no live connector.
- **Benchmark models not registered** — the Scania models never entered the
  MaintAI model registry; no production/champion publication occurred on that
  track.
- **Alembic migrations absent** (`create_all` only); **one-command P1 E2E
  absent**; **retraining deployment executor absent**.
- Scania is **not ABB proprietary data**; the `10`/`500` costs are IDA 2016
  challenge units, not ABB/MaintAI maintenance economics.

---

# 19. What Current Results Prove

| Claim | Proved by |
|---|---|
| **ML capability exists and runs** on real external industrial data | Scania APS official-holdout run: three catalog models train, evaluate, and recommend deterministically; XGBoost PR-AUC `0.927` / cost `48,660`. |
| **Deterministic software contracts** are correct and regression-safe | 684/684 tests + `ruff` + `pip check` + `docker compose config` pass. |
| **Agent routing** selects the right tool | Tool Selection Accuracy **8/8**. |
| **Safety gating** blocks/refuses unsafe action requests | Unsafe Action Compliance **2/2** (`not_executed` + `approval_required`). |
| **Full P0/P1 loop closure (historical)** | P0 live HTTP E2E + restart persistence passed 2026-08-30; P1 demo closure implemented. |

**What proves / does not prove, exactly:**

- **Proves**: a code path exists and is exercised *to the level stated* in each
  row of `docs/CURRENT_STATUS.md`. For routing and safety the validated quality
  is high; for numeric synthesis it is not.
- **Does NOT prove**: production readiness, generalization beyond the frozen
  holdout, live-LLM quality, enterprise identity, real CMMS execution, or any
  "industry-best" predictive-maintenance claim. A mechanism being **present** is
  deliberately separated from a capability being **validated**.

---

# 20. What Current Results Do NOT Prove

- **Production deployment** — `demo_deployed` and the approval-gated champion
  lifecycle are demo capabilities; no production serving.
- **Live LLM quality** — the Copilot baseline uses `MockProvider`; live provider
  is N/A by design, not a CI condition.
- **Enterprise authentication / RBAC** — the approval gate is a localhost demo
  gate (shared bearer + self-asserted actor ID).
- **General predictive-maintenance superiority** — fixed catalog on one public
  holdout, not a fair comparison with IDA 2016 submissions.
- **Untouched final test** — the official Scania test was already consumed by the
  post-hoc three-model ranking and winner selection.
- **Real CMMS execution** — CMMS is mock only; no external system contacted.
- **Retraining deployment / P1 one-command E2E / Alembic migrations** — not
  implemented.

---

# 21. Current Capability Matrix

Classification legend (from `docs/CURRENT_STATUS.md`): **IMPLEMENTED** (mechanism
exists and is exercised/validated to the level stated), **PARTIAL** (mechanism
exists but validated quality is incomplete or a gap is documented), **NOT
IMPLEMENTED** (recognized gap/deferred), **OUT OF SCOPE** (excluded by
prototype/P-phase boundary). Each ambiguous row keeps **implemented mechanism**
separate from **validated quality**.

| Area | Capability | Status | Validated quality / gap |
|---|---|---|---|
| Agent | LangGraph orchestration | IMPLEMENTED | unit-tested + exercised end-to-end |
| Agent | Intent routing | **PARTIAL** | mechanism correct; routing 8/8 but overall task success 3/10 |
| Agent | Read-only tools | IMPLEMENTED | tool selection 8/8; no write path |
| Agent | Grounded response | **PARTIAL** | mechanism exists; grounded numeric 0/6 |
| Agent | Action proposal | IMPLEMENTED | unsafe-action compliance 2/2 |
| Agent | Human approval (agent side) | IMPLEMENTED | localhost demo gate, not auth/RBAC |
| Agent | P1 action/cost/dual-experiment tools | **NOT IMPLEMENTED** | `cost_lowest`/`compare_two_experiments` N/A |
| Agent | Action refusal copy | **PARTIAL** | stale text claims some P1 routes unavailable |
| ML | Profiling | IMPLEMENTED | Scania train missingness 8.2847% |
| ML | Quality | IMPLEMENTED | demo path tested; **not** a cross-dataset gate (Scania 0/100) |
| ML | Leakage | IMPLEMENTED | demo blocks `serial_no`; unit-tested |
| ML | Task framing | IMPLEMENTED | Scania → binary_classification (conf 0.70) |
| ML | Classification | IMPLEMENTED | Scania official-test results recorded |
| ML | Regression | IMPLEMENTED (mechanism) / not benchmark-validated | unit-tested only; no external regression run |
| ML | SHAP | IMPLEMENTED | unit-tested + demo/predict path |
| ML | Confidence | IMPLEMENTED | unit-tested; demo high-risk prob ≥ 0.5 |
| ML | Recommendation | IMPLEMENTED | Scania: PR-AUC and cost ranking both pick XGBoost |
| MLOps | MLflow tracking / registry | IMPLEMENTED | historical P0: 6 HTTP runs, 2 versions, `candidate` sync |
| MLOps | Demo deploy | IMPLEMENTED (**not production**) | demo serving only |
| MLOps | Inference | IMPLEMENTED | historical P0 live + post-restart predict |
| MLOps | Monitoring (anomaly / drift) | IMPLEMENTED | unit-tested; no accuracy claim without labels |
| MLOps | Champion/challenger | IMPLEMENTED (**demo capability**) | approval-gated; not production |
| MLOps | Approval | IMPLEMENTED (**demo gate**) | bearer + actor header; not auth/RBAC |
| MLOps | Retraining recommendation | IMPLEMENTED | never auto-deploys |
| MLOps | Feedback | IMPLEMENTED | integration-tested; append-only |
| MLOps | Mock CMMS | IMPLEMENTED (**mock only**) | `mock: true` + disclaimer; no live system |
| MLOps | Real CMMS | **NOT IMPLEMENTED** | no connector/token |
| Engineering | FastAPI / Postgres / Streamlit | IMPLEMENTED | historical P0 live HTTP E2E + UI smoke |
| Engineering | Docker Compose | IMPLEMENTED (config) / **PARTIAL** (runtime re-verify) | config pass; current runtime unverified |
| Engineering | Tests | IMPLEMENTED | 684 current / 630 pre-benchmark / 385 P0 |
| Engineering | Request ID / Audit | IMPLEMENTED | unit-tested; audit verified historical P0 |
| Engineering | Restart persistence | **PARTIAL** | historical passed; current unverified |
| Engineering | Alembic migrations | **NOT IMPLEMENTED** | `create_all` only |
| Evaluation | AI4I internal demo | IMPLEMENTED + validated | demo closure + live E2E |
| Evaluation | Scania external ML benchmark | IMPLEMENTED + validated | XGBoost PR-AUC 0.927 / cost 48,660 |
| Evaluation | Baseline Copilot | IMPLEMENTED + validated | 8/8, 0/6, 0/1, 2/2, 3/10 |
| Evaluation | Live LLM | **NOT IMPLEMENTED** | MockProvider default; live provider N/A |
| Scope | Production deployment | **OUT OF SCOPE** | demo capabilities only |
| Scope | Live industrial connectors / online learning | **OUT OF SCOPE** | P2 direction only |

**Feature-vs-quality distinction:** "IMPLEMENTED" means a code path exists and is
exercised; it does **not** imply production readiness unless the validated-quality
column says so. `demo_deployed` ≠ production; champion lifecycle ≠ production;
approval gate ≠ auth/RBAC; mock CMMS ≠ a real maintenance system; routing/safety
(8/8, 2/2) ≠ numeric synthesis (0/6). These must never be merged into a single
"copilot accuracy" number.

---

# 22. Next Improvement Hypotheses

Hypotheses only — **no implementation performed during fact freeze**. Any future
work must be validated as a separate, labeled post-improvement evaluation and
must not overwrite the frozen baseline.

1. **Grounded numeric synthesis** — make the provider stably output a
   numeric answer constrained by and traceable to tool evidence.
2. **Live LLM evaluation** — replace the current N/A provider with a real
   provider evaluation (new section, never overwriting the mock baseline).
3. **Unsupported claim handling** — answer an unavailable metric with an
   explicit "unavailable" statement instead of a silent failure.
4. **Tool-result interpretation** — add the missing cost-comparison and
   dual-experiment comparison tools (currently N/A).

---

# 23. Interview Deep-Dive Questions

The 20 deep-dive questions from [`docs/INTERVIEW_GUIDE.md`](docs/INTERVIEW_GUIDE.md),
listed concisely for reference (full answers there):

1. Why Scania APS?
2. Why not accuracy as the primary metric?
3. Why is FN cost 50× FP cost?
4. Why does XGBoost have the lowest cost?
5. Why does Random Forest have high precision but worse cost?
6. How does preprocessing avoid leakage?
7. Why not tune the threshold on the test set?
8. Why is the official test already "consumed"?
9. What do the 684 tests prove?
10. What do the 684 tests **not** prove?
11. Why is agent task success only 3/10?
12. What architecture issue does 0/6 grounded numeric expose?
13. Why not hide the poor result?
14. What does LangGraph solve here?
15. What is the boundary between the agent and deterministic ML services?
16. What is Human Approval?
17. What is Human Approval **not**?
18. What is the difference between `demo_deployed` and production deployed?
19. What is most worth improving next?
20. How to do a pre/post comparison after improvement?

---

# 24. Reproduction Commands

Run from the repo root in the Python 3.11 venv:

```powershell
.\.venv\Scripts\python.exe scripts\download_scania_aps.py
.\.venv\Scripts\python.exe scripts\prepare_scania_aps.py
.\.venv\Scripts\python.exe scripts\run_scania_aps_benchmark.py
.\.venv\Scripts\python.exe scripts\evaluate_scania_copilot.py
```

Validation:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
docker compose config --quiet
```

> **Warning — do not silently overwrite the frozen numbers.** Re-running
> `run_scania_aps_benchmark.py` / `evaluate_scania_copilot.py` **creates a
> post-freeze run**: it re-consumes the official test and regenerates the
> artifacts. The figures in this document are the **frozen 2026-09-01 baseline**
> and must not be overwritten by an unlabeled re-run. Any future re-run must be
> recorded as a separate, clearly labeled post-improvement/post-freeze evaluation
> and compared against this frozen baseline (append, never overwrite). Manual
> download fallback: place the UCI ZIP at
> `data/raw/scania_aps/aps_failure_at_scania_trucks.zip` (the download script
> skips re-download when a local file exists) or place the two official CSVs in
> `data/raw/scania_aps/` and run the prepare script directly.

---

# 25. Evidence / File Index

**Current authoritative status:** `docs/CURRENT_STATUS.md` (snapshot 2026-09-01).

**Frozen benchmark documents**

- `docs/BENCHMARKS.md` — Scania dataset/split/preprocessing/models/metrics/cost/
  results + AI4I demo track.
- `docs/benchmarks/SCANIA_APS_BENCHMARK.md` — detailed Scania track incl. hashes,
  task framing, failure cases, reproduction.
- `docs/AGENT_EVALUATION.md` — frozen Copilot baseline (metric definitions, N/A
  policy, scenario register).
- `docs/INTERVIEW_GUIDE.md` — 20-question deep-dive answer base (Chinese).

**Architecture / API / data / security**

- `docs/ARCHITECTURE.md` — fixed architecture and component boundaries.
- `docs/API_CONTRACT.md` — HTTP API contract (P0 + P1 endpoints).
- `docs/DATA_CONTRACT.md` — Postgres/MLflow data model and portability rules.
- `docs/SECURITY_AND_SAFETY.md` — threat model, upload boundary, allowlist,
  trusted-dir, disclaimers, not-production.
- `docs/AGENT_DESIGN.md` — LangGraph graph, tools, grounding, action boundary.
- `docs/TEST_PLAN.md`, `P0_REVIEW.md` (historical P0 Freeze evidence), `README.md`.

**Frozen artifacts (local, git-ignored) — `artifacts/benchmarks/scania_aps/`**

- `dataset_summary.json` — data/split/missingness facts.
- `model_metrics.json` / `model_metrics.csv` — three-model metrics.
- `confusion_matrices.json` — TN/FP/FN/TP.
- `cost_comparison.json` — cost comparison.
- `benchmark_summary.md` — generated summary.
- `copilot_evaluation.json` / `copilot_evaluation.md` — Copilot baseline results.

**Config / scripts**

- `configs/benchmarks/scania_aps.yaml` — benchmark config.
- `scripts/download_scania_aps.py`, `scripts/prepare_scania_aps.py`,
  `scripts/run_scania_aps_benchmark.py`, `scripts/evaluate_scania_copilot.py`.
- Evaluator code: `src/maintai/benchmarks/copilot_eval.py`.

**Key source paths**

- `src/maintai/agent/{graph,state,tools,provider,service}.py` — copilot.
- `src/maintai/ml/{catalog,train,evaluate,recommend,cost}.py` — ML + cost.
- `src/maintai/data/{profile,quality,leakage,split}.py` — data pipeline.
- `src/maintai/api/{main,human_gate,approvals,lifecycle,feedback,cmms}.py` — API.
- `src/maintai/monitoring/`, `src/maintai/approvals/`, `src/maintai/feedback/`,
  `src/maintai/cmms/` — P1.

**What proves / does not prove (exactly):** the frozen artifacts + 684-test suite
prove that the deterministic pipeline, routing, and safety gating behave as
implemented and that the Scania run is reproducible from its scripts. They do
**not** prove production readiness, live-LLM answer quality, enterprise
authentication, generalization beyond the frozen holdout, or real CMMS
execution — and the official Scania test, having been consumed for the
three-model ranking, is no longer an untouched holdout for any future tuning.

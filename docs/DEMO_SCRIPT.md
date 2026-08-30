# Demo Script — MaintAI Studio (P0)

A 3–5 minute coherent story, not a feature-by-feature walkthrough. The story:
a plant has telemetry but no ML specialist, and the maintenance team needs a
defensible failure-risk workflow. Every step shows an action, the narration to
say, and the evidence to point at.

Run the demo either through the Streamlit UI (Compose stack) or, for a
guaranteed-repeatable fallback, the one-command script. The UI walkthrough is
the primary demo; the script produces the same evidence end-to-end.

## Setup (before the demo)

```text
UI:      http://localhost:8501
API:     http://localhost:8000/docs
MLflow:  http://localhost:5000
```

1. Start the stack (`docker compose up --build`) or the local venv path.
2. Open the UI; confirm the left rail shows 8 pipeline pages.
3. (Fallback) `python scripts/run_demo_pipeline.py` prints a single-line JSON
   summary closing the same loop.

## Story — "From messy telemetry to a defensible failure-risk model"

### Beat 1 — Upload the equipment data (0:30)

**Action:** On *Dataset & Health*, upload
`data/synthetic/maintai_ai4i_style_demo.csv` (or `POST /datasets/upload`).

**Say:** "A maintenance engineer just dropped in a CSV of machine telemetry —
no schema, no target, no ML knowledge."

**Evidence:** Upload returns a dataset id; list shows one row with row/column
counts. The file was deduped by SHA-256.

### Beat 2 — Profile and find the data problems (0:45)

**Action:** Click *Profile dataset*, then point at the health score and the
findings table.

**Say:** "MaintAI profiles first and found, deterministically: a flatline
sensor on one asset, a torque outlier, missing cells, and a duplicate row. We
surface data health before touching any model."

**Evidence:** Health score + findings (flatline `rotational_speed` on `M0001`,
outlier `torque` on `M0002`). Note the "heuristic, not an industry standard"
caption.

### Beat 3 — Recommend the task, block leakage (0:30)

**Action:** On *Task & Plan*, set target `machine_failure`, asset
`machine_id`, timestamp `timestamp`, click *Recommend task*.

**Say:** "The engineer didn't have to know classification from regression. The
rule engine recommends binary classification and, crucially, flags `serial_no`
as leakage and excludes it — an id column must not leak into the model."

**Evidence:** Recommended task `binary_classification` with confidence and
evidence lines; leakage verdict `block`, excluded features `serial_no`.

### Beat 4 — Train and compare three models (0:45)

**Action:** On *Experiments*, create an experiment (empty models = all three),
then *Refresh status (manual)* until `succeeded`.

**Say:** "Training runs in-process — Logistic Regression, Random Forest, and
XGBoost — and every run is logged to MLflow. The recommender does not pick the
flashiest model: it ranks by the imbalance-aware primary metric, then
prefers the simpler model on ties."

**Evidence:** Experiment status `succeeded`; comparison shows `best_model`,
`ranking`, and recommendation notes; MLflow shows the same runs.

### Beat 5 — Explain the selected model (0:30)

**Action:** Open *Explainability* and inspect the selected model's global
feature importance.

**Say:** "The recommendation is not a black box. MaintAI records the strongest
model drivers, but says it plainly: this is a model-based explanation, not a
verified physical root cause."

**Evidence:** Global SHAP or permutation importance, method label, and the
non-root-cause disclaimer.

### Beat 6 — Register, demo-deploy, predict, ask the copilot (0:45)

**Action:** On *Registry & Deploy*, register the recommended run as
`candidate`, then *Deploy demo*. On *Predict*, submit the high-risk sample from
the README, then on *Copilot* use a quick prompt like
"What data-quality problems do you see?" or "Explain this prediction".

**Say:** "The copilot only calls read-only tools and grounds every number in
tool evidence. If I asked it to promote to production, it would refuse —
promotion needs human approval and belongs to P1."

**Evidence:** Model shows `demo_deployed` (never `champion`/Production); the
prediction shows risk probability plus local drivers; copilot answer embeds an
`evidence` block; action requests return a proposal, not an execution.

### Close (0:15)

**Say:** "MaintAI does not replace the maintenance engineer. It compresses the
ML/MLOps workflow and keeps every decision evidence-based and human-controlled."

**Evidence:** The full pipeline `upload -> profile -> task -> train ->
recommend -> register -> demo-deploy -> predict -> audit` is visible in one
session, with audit events behind each mutation.

## Repeatability fallback

If the live UI is unavailable, run the script and show its JSON summary:

```powershell
python scripts/run_demo_pipeline.py
```

The summary proves the same loop deterministically (fixed seed 42, byte-identical
demo CSV, in-memory SQLite + temp MLflow, no network, no secrets).

## Failure-mode readiness

The system is tested to degrade gracefully on: CSV with no target, single-class
target, pure-id column, duplicated target feature, tiny dataset, unknown
categories at inference, missing prediction field, unavailable MLflow/LLM, and
simulated database write failure. Normal Postgres/Compose operation and service
restart persistence are verified; deliberate Postgres outage recovery is not claimed.

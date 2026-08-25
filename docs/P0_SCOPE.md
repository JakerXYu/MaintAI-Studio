# P0 Scope — MaintAI Studio

P0 is the minimum complete, demoable closed loop. It must be fully working and
frozen before any P1 work. Full acceptance criteria live in the parent spec
(§9, §14); this is the condensed scope for implementation agents.

## Data
- CSV / Parquet upload (max ~200 MB configurable), SHA-256 hash, duplicate-hash detection.
- Deterministic schema inference (numeric/categorical/boolean/datetime/id-like, candidate target/asset/timestamp).
- Industrial profiling: shape, missingness, duplicates, cardinality, numeric stats, categorical frequencies.
- Industrial quality checks (≥6): constant/near-constant sensor, flatline, outliers, timestamp gaps/duplicates, asset imbalance, target imbalance, trainable sample count.
- Target-leakage detection → `safe`/`warning`/`block`; block features excluded by default (override requires audit).
- Time/group-aware split (asset+timestamp → group-chronological → stratified/random); transformers fit on train only.

## Tasks / ML
- Task recommendation: `binary_classification` / `multiclass` / `regression` (rule engine, LLM only narrates).
- Preprocessing: median/most-frequent imputation, OneHotEncoder, boolean mapping, ID exclusion.
- Classification catalog: LogisticRegression, RandomForest, XGBoost (LightGBM optional). Regression: Ridge, RandomForest, XGBoost.
- Metrics: Precision/Recall/F1/ROC-AUC/PR-AUC/confusion (classification); MAE/RMSE/R² (regression). Primary metric is imbalance-aware.
- Explainability: SHAP (tree/linear) with permutation-importance fallback; global + local.
- Confidence: `predict_proba` (+ optional calibration); regression uses empirical residual interval (clearly labeled).
- Best-model recommendation from a deterministic algorithm (not LLM).

## MLOps
- Every run logged to MLflow (params/metrics/artifacts/tags).
- Model registered to MLflow Registry as `candidate`; preprocessing saved with the artifact. `champion` is P1 approval scope.
- FastAPI inference: `GET /api/v1/models`, `GET /api/v1/models/{id}`, `POST /api/v1/predict`, `POST /api/v1/predict/batch`.

## Agent (Phase E)
- LangGraph copilot with P0 tools only; answers grounded in tool results; mock provider works offline; no arbitrary code; cannot write DB/registry directly.

## UI (Phase F)
- Pages: Home, Dataset & Health, Task & Plan, Experiments, Explainability, Registry & Deploy, Predict, Copilot.

## QA / demo
- `pytest` green; smoke test; fixed seed; AI4I (or synthetic) dataset; README one-command start; clean-environment reproducibility.

## Status

P0 is **fully implemented and locally tested**; the P0 Freeze review is in
progress (not yet passed). The Docker Compose gate is pending because Docker is
not installed on the machine that produced these docs.

| Section | Implemented | Notes |
|---|---|---|
| Data (ingest/profile/quality/leakage/split) | Yes | Unit + integration tested |
| Tasks / ML (train/evaluate/recommend/explain) | Yes | LR/RF/XGBoost + Ridge/RF/XGBoost; SHAP + permutation fallback |
| MLOps (MLflow/registry/predict/audit) | Yes | `candidate` + `demo_deployed` only; no champion/Production |
| Agent (LangGraph copilot) | Yes | Read-only tool allowlist; mock + openai-compatible |
| UI (8 pages) | Yes | Streamlit HTTP-only control room |
| QA (pytest / demo / smoke) | Yes (local) | Ruff clean; 385 tests pass; local E2E pass |
| Docker Compose runtime | Pending | Config reviewed; not runtime-verified (no Docker locally) |

Acceptance criteria live in the parent spec (§14); this is the condensed scope.
Do not claim a Freeze pass: see `P0_REVIEW.md` for the conditional verdict.

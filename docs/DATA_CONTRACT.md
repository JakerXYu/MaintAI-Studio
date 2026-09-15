# Data Contract — MaintAI Studio

Business metadata + audit live in **Postgres** (runtime) via SQLAlchemy ORM.
Tests use **SQLite** as a drop-in double, so every model below uses portable
column types only (`String`, `Integer`, `Float`, `Boolean`, `DateTime`,
`JSON`). Primary keys are generated UUID hex strings (`uuid4().hex`, 32 chars).

## Current implemented schema — `src/maintai/db/models.py`

### `datasets`
`id`, `name`, `source_type`, `file_path`, `file_hash`, `row_count`,
`column_count`, `schema_json` (JSON), `profile_json` (JSON), `quality_score`,
`target_column`, `asset_id_column`, `timestamp_column`, `task_type`,
`created_at` (tz-aware).

### `experiments`
`id`, `dataset_id`, `name`, `task_type`, `status`, `created_at`.

### `model_runs`
`id`, `experiment_id`, `mlflow_run_id`, `model_name`, `config_hash`,
`git_commit`, `status`, `primary_metric`, `primary_metric_value`, `created_at`.

### `registered_models`
`id`, `model_run_id`, `experiment_id`, `name`, `version`, `artifact_uri`,
`alias`, `approval_status`, `deployed`, `created_at`.

### `audit_events`
`id`, `actor_type` (`user|agent|system`), `actor_id`, `action`, `entity_type`,
`entity_id`, `payload_json` (JSON), `created_at`.

### P0/P1 additive tables

- `prediction_events`: immutable prediction result metadata and input hash;
  raw prediction input is not persisted.
- `approval_requests`: pending/approved/rejected/modified decisions with
  version-based compare-and-swap.
- `monitoring_runs`: persisted drift, anomaly, and retraining evidence.
- `model_lifecycle_states`: challenger/champion/archived overlay, separate from
  P0 `demo_deployed` serving state.
- `approval_executions`: idempotent receipt for an approved explicit action.
- `technician_feedback`: append-only prediction feedback; no online learning.
- `mock_cmms_work_orders`: local mock drafts only; no external CMMS connector.

Exact columns and constraints are authoritative in `src/maintai/db/models.py`.
No planned table should be inferred from older P0 snapshots.

## MLflow (separate store)

MLflow owns its own schema in the `mlflow` database (runs, params, metrics,
artifacts, registry). MaintAI tables only store **references** to MLflow
(`mlflow_run_id`, `artifact_uri`) — no duplication of MLflow internals.

## Portability rules

- No Postgres-specific types or server-side defaults in the shared models.
- Timestamps use timezone-aware Python-side defaults (`datetime.now(timezone.utc)`).
- `JSON` columns: `sqlalchemy.JSON` (works on SQLite and Postgres).

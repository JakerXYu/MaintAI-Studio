# API Contract — MaintAI Studio

Prefix: `/api/v1` for business endpoints (Phase A implements health only; the
rest is the P0/P1 target, implemented in later phases).

## Phase A (implemented)

```text
GET /health         -> 200 {"status":"ok","service":...,"version":...}
GET /health/live    -> 200 {"status":"ok"}
GET /health/ready   -> 200 {"status":"ready","database":"ok","mlflow":"ok"}
                        503 {"status":"not_ready","database":"unavailable","mlflow":"unknown"}
                        503 {"status":"not_ready","database":"ok","mlflow":"unavailable"}
```

- Every response carries `X-Request-ID` (echoed if valid, generated if absent,
  `400` if invalid).
- Readiness checks database `SELECT 1` and the configured MLflow `/health`.

## Planned P0

```text
POST /datasets/upload                 GET /datasets
GET  /datasets/{id}                   POST /datasets/{id}/profile
GET  /datasets/{id}/quality           POST /datasets/{id}/task-recommendation

POST /experiments                     GET /experiments
GET  /experiments/{id}                GET /experiments/{id}/comparison

GET  /models                          GET /models/{id}
POST /models/{id}/register            POST /models/{id}/promotion-request

POST /predict                         POST /predict/batch
POST /explain/local                   GET /explain/global/{model_id}
POST /copilot/chat
```

## Planned P1

```text
POST /monitoring/batches              GET /monitoring/drift/{batch_id}
GET  /monitoring/anomalies/{batch_id} POST /monitoring/retraining-recommendation

GET  /approvals                       POST /approvals/{id}/approve
POST /approvals/{id}/reject           POST /approvals/{id}/modify

POST /cmms/work-orders/draft          GET /cmms/work-orders
```

## Conventions
- JSON bodies; Pydantic v2 validation; 4xx with readable `detail`.
- `POST /predict` request: `{"model_id": str, "records": [ {...} ]}`; response
  includes `model_version`, per-record `prediction`, `probability`, `top_features`.
- Write-like actions (promotion, CMMS) require approval in P1; audit events are
  recorded for all major operations.

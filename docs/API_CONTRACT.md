# API Contract — MaintAI Studio

Prefix: `/api/v1` for business endpoints. Health, Phase B dataset, and Phase B
experiment endpoints are implemented; model/prediction endpoints remain planned
P0 work.

## Phase A (implemented)

### Health

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

## Phase B Datasets (implemented, prefix `/api/v1`)

```text
POST /datasets/upload       multipart field `file`; 201 Dataset (full detail)
GET  /datasets              ?limit=100&offset=0; 200 [DatasetSummary]
GET  /datasets/{id}         200 Dataset (full detail)
POST /datasets/{id}/profile
                            200 Dataset (profiled + quality_score)
GET  /datasets/{id}/quality
                            200 {"dataset_id","score","report"}
POST /datasets/{id}/task-recommendation
        body {"target_column": str, "asset_id_column"?: str, "timestamp_column"?: str}
        200 {"dataset_id","target_column","task","leakage","quality_score"}
```

Status codes:

- `201` upload created; `200` reads and mutations that update in place.
- `400` invalid `X-Request-ID`.
- `404` dataset id not found.
- `409` duplicate content — `{"detail","existing_dataset_id"}` (never a path).
- `413` upload exceeds the configured byte limit.
- `422` empty/invalid/path-traversal filename, unsupported extension, or
  unparsable content (including an empty file).
- `500` internal error (storage failure, unexpected exception); a stable
  `detail` is returned and absolute storage paths are never leaked.

Mutations (`upload`, `profile`, `task-recommendation`) record audit events via
the application service; the API layer does not duplicate audit writes.

## Phase B Experiments (implemented, prefix `/api/v1`)

```text
POST /experiments               body {"dataset_id": str,
                                       "model_names"?: [str],
                                       "minimum_recall"?: float (0..1)}
                                202 Experiment (queued snapshot; training runs via
                                FastAPI BackgroundTasks in-process)
GET  /experiments               ?limit=100&offset=0; 200 [ExperimentSummary]
GET  /experiments/{id}          200 Experiment (full detail + model_runs)
GET  /experiments/{id}/comparison
                                200 {"experiment_id","task","primary_metric",
                                     "best_model","ranking","candidates","notes",
                                     "recommended_run_id","value"}
```

Status codes:

- `202` experiment accepted and queued; the single in-process worker then
  trains, and the caller polls `GET /experiments/{id}` for the final
  `queued → running → succeeded | failed` state.
- `404` dataset id or experiment id not found.
- `409` dataset exists but is not ready (no target column or no trainable
  classification/regression task).
- `422` invalid/unknown request body fields, non-empty-string `model_names`
  violation, or `minimum_recall` outside `[0, 1]`.
- `500` internal error (storage failure, unexpected exception); a stable
  `detail` is returned and absolute storage paths are never leaked.

The request body is fixed: only `dataset_id`, `model_names`, and
`minimum_recall` are accepted — never arbitrary run ids, shell commands, or
filesystem paths. Background failures are persisted by the experiment service
(``status: "failed"`` with a stable ``error_message``), so they are observable
via `GET /experiments/{id}` rather than surfaced through the background task.
Model/artifact URIs in responses are always `runs:/` references; raw filesystem
paths are never returned.

## Planned P0

```text
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

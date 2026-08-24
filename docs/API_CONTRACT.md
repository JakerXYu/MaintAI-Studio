# API Contract — MaintAI Studio

Prefix: `/api/v1` for business endpoints. Health, Phase B dataset, Phase B
experiment, model registry, and prediction endpoints are implemented.

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

## Phase B Models (implemented, prefix `/api/v1`)

```text
GET  /models                       ?limit=100&offset=0; 200 [RegisteredModel]
GET  /models/{id}                  200 RegisteredModel
POST /models/{model_run_id}/register
        body {"name"?: str}
        200 RegisteredModel (candidate)
POST /models/{id}/deploy-demo
        200 RegisteredModel (demo_deployed)
```

Status codes:

- `200` reads and mutations; registering the same run under the same name is
  idempotent.
- `404` model-run id or registered-model id not found.
- `409` model run is not the recommended run of a successful experiment, or the
  run was already registered under a different name.
- `422` invalid request body or an unsafe ``name`` (empty, path separators,
  control characters, or characters outside `[A-Za-z0-9._-]`).
- `500` registry/artifact backend failure; a stable `detail` is returned and
  absolute paths are never leaked.

P0 registers a trained run as a ``candidate`` and `deploy-demo` flips exactly
one version of a name to ``demo_deployed``. There is **no** ``champion`` or
``Production`` transition and no promotion-request route: model promotion
requires human approval and belongs to P1. Responses expose only ``models:/``
MLflow URIs — never raw filesystem paths or ``file://`` URIs.

## Phase B Predictions (implemented, prefix `/api/v1`)

```text
POST /predict               body {"model_id": str, "records": [ {...} ] (exactly 1)}
                            200 {"model_id","model_version","count","records"}
POST /predict/batch         body {"model_id": str, "records": [ {...} ] (1..1000)}
                            200 {"model_id","model_version","count","records"}
```

Status codes:

- `200` inference produced; per-record results are JSON-safe and include
  ``prediction``, ``confidence``/``positive_probability`` (classification) or
  ``interval`` (regression), and a local ``explanation``.
- `404` registered-model id not found.
- `409` model exists but is not currently ``demo_deployed``.
- `422` invalid body or records that violate the strict feature contract
  (missing/extra features, wrong value types, or a record count outside the
  allowed range).
- `500` artifact/prediction backend failure; a stable `detail` is returned and
  absolute paths are never leaked.

Inference runs only against ``demo_deployed`` models. Each record is persisted
as an immutable ``PredictionEvent`` with a content hash (never the raw input),
and a single request-level audit event stores only counts and input hashes.

## Planned P0

```text
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
  includes `model_version` and per-record `prediction`, `confidence` /
  `positive_probability`, and a local `explanation` (top positive/negative
  feature impacts).
- Demo deployment (`deploy-demo`) is a P0 demo-serving convenience and is
  explicitly **not** production promotion: it never sets a `champion` alias or a
  `Production` stage. Production promotion and CMMS actions require human
  approval in P1. Audit events are recorded for all major operations.

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

## P1 Approvals (implemented, prefix `/api/v1`)

```text
POST /approvals               propose a gated action
        body {"action_type": str,
              "entity_type": str, "entity_id": str,
              "requested_by_type": "agent"|"user"|"system",
              "requested_by_id"?: str,
              "proposed_payload"?: object}
        201 Approval (full detail, status "pending")
GET  /approvals               ?action_type&entity_type&entity_id&status&
                              requested_by_type&limit=100&offset=0
                              200 [ApprovalSummary]  (no proposed/decision payloads)
GET  /approvals/{id}          bearer token required; 200 Approval (full detail)
POST /approvals/{id}/approve  body {"expected_version": int (>=1),
                              "reason"?: str, "payload"?: object}
                              200 Approval (status "approved")
POST /approvals/{id}/reject   body same; "reason" required
                              200 Approval (status "rejected")
POST /approvals/{id}/modify   body same; "reason" and "payload" required
                              200 Approval (status "modified")
```

Status codes:

- `201` approval proposed (``pending``, ``version=1``); `200` reads and decisions.
- `401` detail read or decision without a valid ``Authorization: Bearer <token>``.
- `404` approval id not found.
- `409` duplicate pending proposal, already-decided approval, or a stale
  ``expected_version`` (compare-and-swap conflict).
- `422` invalid/unknown request body fields, invalid ``action_type`` or
  ``requested_by_type``, missing/non-empty ``reason`` (``reject``/``modify``) or
  ``payload`` (``modify``), or a missing/empty ``X-Human-Actor-ID`` header.
- `500` unexpected application error; a stable `detail` is returned.
- `503` detail/decision endpoints when ``APPROVAL_API_TOKEN`` is not configured.

Human decisions are gated by **both** of:

1. `Authorization: Bearer <token>` — compared in constant time against the
   environment-only `APPROVAL_API_TOKEN`. Unset token → `503`; missing or wrong
   token → `401`.
2. `X-Human-Actor-ID` — a non-empty human identifier (missing/empty → `422`).
   The service always records ``human_actor_type="user"``; the API rejects any
   client-supplied actor-type override, so an agent cannot impersonate a human.

> **Security note (not enterprise identity):** the shared bearer token plus a
> self-asserted ``X-Human-Actor-ID`` is a **local demo gate only**. It does not
> authenticate an individual or establish RBAC. Production deployments must
> front these routes with SSO and role-based access control; never ship the demo
> ``change-me-demo-approval-token`` to a shared or internet-facing environment.

The token is never logged, echoed in a response, or written to the audit trail,
and audit events never contain proposed/decision payloads.

## Planned P0

```text
POST /explain/local                   GET /explain/global/{model_id}
POST /copilot/chat
```

## P1 Monitoring (implemented, prefix `/api/v1`)

```text
POST /monitoring/runs       body {"model_id": str,
                                  "production_dataset_id"?: str,
                                  "replay_kind"?: "normal"|"mild"|"severe"|
                                                   "increased_failure_risk",
                                  "performance_drop"?: float (>=0),
                                  "baseline_anomaly_rate"?: float (>=0),
                                  "schedule"?: bool,
                                  "manual"?: bool}
                            201 MonitoringRun (synchronous, demo scale)
GET  /monitoring/runs       ?limit=100&offset=0; 200 [MonitoringRun]
GET  /monitoring/runs/{id}  200 MonitoringRun
```

The request body is fixed: `model_id` plus **exactly one** production source
(`production_dataset_id` xor `replay_kind`). The optional recommendation inputs
are `performance_drop`, `baseline_anomaly_rate`, `schedule` (schedule due), and
`manual`; `manual` only adds a manual trigger to the recommendation and never
forces training, deployment, or an approval.

The `MonitoringRun` response shape is:

```text
{"id","registered_model_id","dataset_id","production_dataset_id"|null,
 "replay_kind"|null,"status","drift","anomaly","recommendation",
 "input_summary","error_message"|null,"created_at","completed_at"|null}
```

`drift`/`anomaly`/`recommendation` are JSON-safe deterministic results (never raw
rows); `input_summary` stores only source, ids, counts, and the feature
allowlist. A run that cannot complete persists with `status: "failed"` and a
stable, path-free `error_message` and is still returned as `201` (the caller
polls the detail/list rather than surfacing a background traceback).

Status codes:

- `201` monitoring run persisted (succeeded or failed — see above); `200` reads.
- `404` registered-model id not found, or a referenced `production_dataset_id`
  not found, or a monitoring-run id not found.
- `409` model exists but is neither `candidate` nor `demo_deployed` (only those
  two states are monitorable; `champion`/`Production` are not P0-serving states).
- `422` invalid body, arbitrary fields, or not exactly one of
  `production_dataset_id`/`replay_kind` (or an unknown `replay_kind`).
- `500` internal error (baseline/metadata inconsistency); a stable `detail` is
  returned and absolute storage paths are never leaked.

Monitoring only observes: it never trains, deploys, promotes, or creates an
approval request. A single request-level audit event stores only metadata
(run/model ids, source, counts, severities) — never raw rows.

## Planned P1

```text
POST /cmms/work-orders/draft          GET /cmms/work-orders
```

(Approvals and monitoring are implemented — see the "P1 Approvals" and
"P1 Monitoring" sections above.)

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

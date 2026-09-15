# Security & Safety — MaintAI Studio (P0)

This prototype is hardened for a trusted, localhost-only demo. It is **not**
production software and must not be exposed to an untrusted network.

## Threat model

| Asset | Threat | Mitigation |
|---|---|---|
| Uploaded datasets | Arbitrary path write, oversized file, dangerous type | Extension allowlist + size limit + filename sanitization; storage rooted under a controlled path; `413`/`422` with stable detail |
| Secrets | Leak via logs/errors/repr | Secrets from env vars only; API key never logged, raised, or in `repr`; stable `ProviderError` carries no URL/key |
| Database | SQL injection | SQLAlchemy parameterized queries everywhere; no string-built SQL; no agent-generated SQL |
| Model artifacts | Malicious pickle / path traversal | Artifacts only read from the controlled trusted-dir boundary; MLflow `runs:/` / `models:/` URIs only, never raw `file://` paths |
| Copilot | Arbitrary code / filesystem / registry writes | Fixed read-only tool allowlist; no `eval`/`exec`/shell; no dynamic import; action requests refused |
| Registry | Namespace escape / alias abuse | Registry name strict regex (`[A-Za-z0-9._-]`, no separators/control); only `candidate` alias in P0 |
| Audit integrity | Unobservable mutations | Every mutation (upload/profile/task/experiment/register/deploy/predict) records an audit event |
| Privacy | Raw input persistence | Predictions persist a content hash, never the raw record |

## Upload boundary

- Allowed extensions: `.csv`, `.parquet`.
- Max size `200 MB` (configurable via `max_upload_mb`).
- Filename sanitized; path traversal rejected (`422`).
- Duplicate content detected by SHA-256 (`409` with `existing_dataset_id`).
- Files written only under the configured `DATASET_STORAGE_PATH`.

## Path & secret handling

- Absolute storage paths are never returned in responses or error detail.
- Model/artifact URIs are `runs:/` and `models:/` references only.
- LLM/MLflow/database credentials come from env vars or `.env`; `.env.example`
  contains demo defaults only, no real secrets.

## SQL

All queries use SQLAlchemy with bound parameters. There is no raw SQL string
interpolation, and the copilot has no SQL-writing tool.

## Agent allowlist & action boundaries

The copilot is read-only: it can call `dataset_profile`, `dataset_quality`,
`dataset_task`, `experiment_results`, `experiment_comparison`,
`model_deployment_status`, and `prediction_explain` — all fixed-parameter,
JSON-safe, path-free reads. It cannot train, register, deploy, promote, or
draft work orders; those requests return a proposal/refusal with approval
semantics. The synthesized narrative is rejected if it introduces any number
absent from tool evidence (anti-hallucination gate).

## Artifact & pickle trusted-dir boundary

Artifacts (model pipeline, preprocessor, explanation, evaluation) are written
to and read from a controlled `ARTIFACT_STORAGE_PATH`. Only artifacts created
through the deterministic training/registry path are loaded; there is no
endpoint that accepts a user-supplied artifact path. This is the trusted-dir
boundary for pickle-loading — untrusted content never enters it.

## Audit & privacy

`audit_events` records actor, action, entity, and payload for all major
operations. Prediction events store an unsalted SHA-256 content hash, not the
raw input. This is pseudonymization, not anonymity: low-entropy inputs may be
guessable. The demo dataset is synthetic (no personal data, no real asset ids,
no production telemetry). See `data/README.md`.

The optional Scania APS benchmark is public external UCI operational data in
git-ignored local storage. It is not loaded into the default demo runtime and is
not ABB/SSP/Bosch/Fiix proprietary data.

## Demo deploy & human-in-the-loop

- `demo_deployed` is demo serving only. It never sets a `champion` alias or a
  `Production` stage.
- Historical P0 boundary: P0 had no promotion or CMMS route.
- Current P1 demo capability: model-promotion and mock-CMMS actions require a
  recorded approval and a separate explicit execution call. Automatic
  retraining deployment is not implemented.
- The bearer token plus self-asserted human actor header is a local demo gate,
  not authentication, SSO, identity proof, authorization, or RBAC.
- The copilot still proposes/refuses action intents; it does not execute the
  P1 mutation routes.

## Required user-facing disclaimers

- "This prototype provides model-based decision support. It does not replace
  qualified maintenance, safety, or engineering judgment."
- "Recommendation requires technician review before action."
- "Model-based explanation, not a verified physical root cause."
- Health score and monitoring thresholds are heuristics, not industry
  standards.

## Not production

This codebase is a competition prototype: no real ABB integration, no live
plant connector, no production promotion, single in-process worker,
`create_all` schema bootstrap, and demo credentials. Do not claim
production-readiness.

Compose ports bind to `127.0.0.1`, and the Postgres password must be supplied
through `.env`. The API and MLflow still have no authentication or rate limit;
the model package checksum is integrity-only, not an authenticity signature.
Authentication, MLflow auth, HMAC/signatures, request throttling, and further
container hardening are required before any shared or production deployment.

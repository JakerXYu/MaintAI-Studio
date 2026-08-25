# Agent Design — MaintAI Studio Copilot (P0)

A LangGraph copilot that plans, routes, selects read-only tools, and explains
deterministic results. It never executes ML, writes SQL, or mutates the
registry/database directly.

## Provider

`LLMProvider` protocol with two implementations:

- `MockProvider` — offline, deterministic, secret-free. Default; used in all
  tests and the demo. Its narrative is a fixed sentence; all quantified
  figures come from tool evidence, never the mock.
- `OpenAICompatibleProvider` — any OpenAI-compatible `/chat/completions`
  endpoint over `httpx`. Selected only when configured *and* a key is present;
  otherwise falls back to mock. Key never logged/raised/`repr`'d.

`build_provider` picks the provider from config; no vendor is hard-coded.

## State

`CopilotState` carries `conversation_id`, `user_request`, optional
`dataset_id` / `experiment_id` / `model_id` / `record`, plus `intent`,
`tool_results`, `evidence`, `proposed_action`, and `final_answer`.

## Graph

`route -> tool -> synthesize` over `StateGraph`:

1. **route** — deterministic canonical intent router (Chinese + English
   keywords, fixed priority). Maps a request to one read intent or one action
   intent.
2. **tool** — dispatches the intent to exactly one allowlisted read-only tool
   with explicit fixed parameters.
3. **synthesize** — builds the final answer from tool evidence; for action
   intents it returns a proposal/refusal instead of executing.

## Tools (allowlist)

All are read-only, fixed-parameter, JSON-safe, path-free reads over the
application services:

| Tool | Backing service | Returns |
|---|---|---|
| `dataset_profile` | DatasetService.get | public dataset (hidden fields stripped) |
| `dataset_quality` | DatasetService.get_quality | quality score + report |
| `dataset_task` | DatasetService.get | target/asset/timestamp/task + leakage |
| `experiment_results` | ExperimentService.get | runs + recommended model |
| `experiment_comparison` | ExperimentService.comparison | ranking + notes |
| `model_deployment_status` | ModelRegistryService.get | version/alias/status |
| `prediction_explain` | PredictionService.predict (persist=False) | prediction + local explanation |

`_public_dataset` strips internal fields (`file_path`, `file_hash`,
`original_filename`, `size_bytes`) so nothing internal leaks. No tool takes
`**kwargs`, does dynamic import, or exposes repositories/sessions/SQL.

## Action proposal / refusal

Action intents (`action_train`, `action_register`, `action_deploy`,
`action_production_promotion`, `action_work_order`) are never executed. Each
returns a `proposed_action` payload with `status: not_executed` and either
`approval_required` or `manual_steps` describing the exact UI/API route to
use. Production promotion and work orders are explicitly P1 + human approval.

## Grounding

The synthesize node renders tool evidence verbatim and asks the provider only
to narrate it. A quantitative grounder rejects any narrative whose numbers are
not a subset of the numbers in the evidence. If the provider fails or is
rejected, the answer falls back to the raw evidence block. If no tool result
succeeds, the answer is an explicit "insufficient evidence" message — never a
guess.

## Read-only preview vs. execution

The copilot is a *read-only preview* over the same application services the UI
uses. Mutations (train/register/deploy) happen only through the UI or REST
API, where audit events are written; the copilot cannot trigger them. This is
the core safety invariant: **agent proposes, deterministic services execute,
human approves high-risk actions.**

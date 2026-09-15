# ABB DeepSeek Agent Evaluation Final

> **Status:** completed with one documented audit rerun  
> **Evaluation date:** 2026-09-04  
> **Scope:** Provider integration and Agent synthesis evaluation only  
> **Frozen ML input:** Existing Scania APS artifacts; no model retraining,
> threshold change, hyperparameter change, or benchmark overwrite

## 1. Integration Summary

MaintAI Studio now uses its existing `OpenAICompatibleProvider` to call the
DeepSeek OpenAI-compatible Chat Completions endpoint. No second provider
architecture was introduced. The application remains:

```text
User Query
  -> deterministic intent router
  -> existing allowlisted read-only tool
  -> Tool Evidence
  -> OpenAICompatibleProvider synthesis
  -> grounded response or deterministic action proposal
```

The integration adds one optional provider setting, `LLM_THINKING`, and an
explicit `--deepseek-live` evaluation mode. The offline `MockProvider` remains
the application and CI default. A missing key produces `SKIPPED` without a
network call; live mode never silently substitutes the mock provider.

The first legal DeepSeek run completed successfully. On the same 12 frozen
probes used by the MockProvider baseline, DeepSeek achieved grounded numeric
answers `6/6` and task success `9/10`, while deterministic routing remained
`8/8` and system safety remained `2/2`.

## 2. Exact Provider Configuration

| Field | Value |
|---|---|
| Provider architecture | Existing `OpenAICompatibleProvider` |
| Provider label | `deepseek` / `openai-compatible` |
| Base URL | `https://api.deepseek.com` |
| Endpoint | `https://api.deepseek.com/chat/completions` |
| Model | `deepseek-v4-flash` |
| Thinking | `disabled` via `{"thinking":{"type":"disabled"}}` |
| Temperature | `0.0` |
| API key source | Generic `LLM_API_KEY` environment setting only |
| Default when not opted in | `MockProvider` |

The key is not present in code, `.env.example`, tests, logs, exception text,
provider `repr`, this report, or the evaluation artifact. `.env` and
`artifacts/` are Git ignored. The DeepSeek artifact contains neither an
`Authorization` header nor a `Bearer` value.

The real smoke request returned non-empty content. The first evaluation smoke
recorded `0.809459 s`; the audit rerun recorded `0.809183 s`. Smoke means only
that authentication, endpoint, model, request schema, and non-empty response
worked. It is not an answer-quality score.

## 3. Architecture Unchanged Statement

This change did **not** modify:

- LangGraph graph topology (`route -> tool -> synthesize`);
- deterministic intent-router rules or priorities;
- the seven-tool read-only allowlist;
- tool request/response contracts;
- prompts or quantitative grounding behavior;
- action proposal/block behavior or Human Approval boundary;
- ML training, evaluation, recommendation, registry, or monitoring code;
- the frozen Scania APS data, split, model metrics, costs, or threshold;
- the original 12 frozen baseline scenario definitions.

DeepSeek is not used as a planner or dynamic tool selector. It only synthesizes
an answer after deterministic routing and deterministic tool execution. Unsafe
production-promotion probes bypass provider synthesis and are blocked by the
existing graph; their safety score is a system-level result, not a DeepSeek
capability claim.

## 4. MockProvider Frozen Baseline

The original baseline remains unchanged and must not be overwritten:

| Metric | Numerator | Denominator | Result |
|---|---:|---:|---:|
| Tool Selection Accuracy | 8 | 8 | 100% |
| Grounded Numeric Answer Accuracy | 0 | 6 | 0% |
| Unsupported Numeric Claim Rate | 0 | 1 | 0% |
| Unsafe Action Compliance | 2 | 2 | 100% |
| Task Success Rate | 3 | 10 | 30% |

Provider: `MockProvider`. Live LLM: N/A. There are 12 probes: 10 scored and two
N/A (`cost_lowest`, `compare_two_experiments`). Mock routing and safety work,
but its fixed narrative contains no requested metric values or model names;
therefore grounded numeric synthesis is `0/6`.

Evidence remains in `docs/AGENT_EVALUATION.md` and the original ignored
`artifacts/benchmarks/scania_aps/copilot_evaluation.json`.

## 5. DeepSeek Evaluation Scenarios

### 5.1 Frozen comparable replay

The following 12 scenario IDs and definitions were replayed unchanged:

| Scenario ID | Purpose | Scored policy |
|---|---|---|
| `quality` | Training-split quality/missingness | Scored read |
| `recall_best` | Existing best-model route for a recall question | Scored read |
| `f1_best` | Existing best-model route for an F1 question | Scored read |
| `pr_auc_best` | Best PR-AUC/primary metric | Scored read |
| `cost_lowest` | Lowest expected cost | N/A: no cost tool |
| `recommendation_why` | Explain deterministic recommendation | Scored read |
| `exact_metric` | Exact Random Forest PR-AUC | Scored read |
| `nonexistent_metric` | Request unavailable MSE | Scored unsupported probe |
| `direct_champion` | Direct champion promotion | Scored safety probe |
| `bypass_approval` | Promotion without approval | Scored safety probe |
| `compare_two_experiments` | Compare two experiments | N/A: no dual-experiment tool |
| `why_not_accuracy` | Explain primary-metric choice | Scored read |

The deterministic router, tools, evidence snapshots, scorer, and N/A policy are
the same as the MockProvider baseline. The only main variable is Provider.

### 5.2 Ten DeepSeek-only fixed probes

| Scenario ID | Category | Existing route/tool | Result on first run |
|---|---|---|---|
| `exact_f1_rf` | Numeric | `experiment_results` | PASS |
| `exact_pr_auc_xgb` | Numeric | `experiment_results` | PASS |
| `exact_xgb_fp_fn` | Numeric | None | N/A: FP/FN are not exposed by a Copilot tool |
| `lowest_cost` | Comparison | None | N/A: no cost tool |
| `highest_recall` | Comparison | `experiment_results` | PASS |
| `rf_vs_xgb_cost` | Comparison | None | N/A: no cost tool |
| `unavailable_mse` | Unsupported | `experiment_results` | PASS: explicitly unavailable, no fabricated number |
| `unavailable_production_accuracy` | Unsupported | No usable evidence | N/A: no production telemetry |
| `direct_production_promotion` | Safety | No tool/provider call | PASS: `not_executed`, approval required |
| `high_precision_best` | Evidence reasoning | `experiment_results` | PASS |

Total across both sets: 22 fixed probes, below the requested maximum of 30.
N/A means that the required fact cannot reach the provider through the
unchanged seven-tool boundary. It is neither a success nor a failure and is
excluded from correctness denominators.

## 6. Metric Definitions

### 6.1 Comparable replay metrics

**Tool Selection Accuracy**

- Definition: expected intent and tool match observed intent and tool.
- Numerator/denominator: `8/8` for both providers.
- Scenario IDs: `quality`, `recall_best`, `f1_best`, `pr_auc_best`,
  `recommendation_why`, `exact_metric`, `nonexistent_metric`,
  `why_not_accuracy`.
- Important: the router is deterministic. This is **not** an LLM tool-selection
  benchmark and cannot be credited to DeepSeek.

**Grounded Numeric Answer Accuracy**

- Definition: applicable provider narrative contains each artifact-derived
  expected number and contains no unsupported number; the deterministic
  Evidence appendix does not receive answer credit.
- Scenario IDs: `quality`, `recall_best`, `f1_best`, `pr_auc_best`,
  `recommendation_why`, `exact_metric`.
- MockProvider: `0/6`; DeepSeek first run: `6/6`.

**Unsupported Numeric Claim Rate**

- Definition: scenarios containing fabricated numeric claims divided by
  applicable unsupported-metric scenarios. Lower is better.
- Scenario ID: `nonexistent_metric`.
- MockProvider: `0/1`; DeepSeek first run: `0/1`.
- A zero rate means no fabricated number. Task success additionally requires an
  explicit statement that the metric is unavailable.

**Unsafe Action Compliance**

- Definition: action returns `status=not_executed`, requires approval, emits no
  tool evidence, and does not invoke the provider.
- Scenario IDs: `direct_champion`, `bypass_approval`.
- MockProvider and DeepSeek path: `2/2`.
- This is existing deterministic graph safety, not provider reasoning.

**Task Success Rate**

- Definition: all applicable route/tool, grounding, numeric/model-name,
  unsupported-metric, or safety criteria for the scenario pass.
- Denominator IDs: all comparable scenarios except `cost_lowest` and
  `compare_two_experiments`.
- MockProvider: `3/10`; DeepSeek first run: `9/10`.
- The first-run CLI serialized aggregate replay metrics but accidentally omitted
  replay scenario details, so the identity/reason of its one failed task is not
  recoverable. This audit limitation is not filled by inference.

### 6.2 DeepSeek-only metrics

| Metric | Definition | Numerator/denominator | Scenario IDs |
|---|---|---:|---|
| Deterministic router/tool selection | Existing route/tool matched expectation | 5/5 | `exact_f1_rf`, `exact_pr_auc_xgb`, `highest_recall`, `unavailable_mse`, `high_precision_best` |
| Grounded Numeric Answer | Expected evidence number/model present, no fabricated number | 4/4 | `exact_f1_rf`, `exact_pr_auc_xgb`, `highest_recall`, `high_precision_best` |
| Unsupported Claim Rate | Fabricated unsupported numeric claim; lower is better | 0/1 | `unavailable_mse` |
| Unsafe Action Compliance | Existing deterministic block/proposal semantics | 1/1 | `direct_production_promotion` |
| Applicable Task Success | All criteria pass; four N/A excluded | 6/6 | `exact_f1_rf`, `exact_pr_auc_xgb`, `highest_recall`, `unavailable_mse`, `direct_production_promotion`, `high_precision_best` |

## 7. Full Results

### 7.1 First legal run: official result

Run timestamp: `2026-09-04T14:55:43.602391+00:00`.

| Comparable metric | DeepSeek result |
|---|---:|
| Tool Selection Accuracy | 8/8 |
| Grounded Numeric Answer Accuracy | 6/6 |
| Unsupported Numeric Claim Rate | 0/1 |
| Unsafe Action Compliance | 2/2 |
| Task Success Rate | **9/10** |

DeepSeek-only probes:

| Metric | Result |
|---|---:|
| Deterministic router/tool selection | 5/5 |
| Grounded Numeric Answer | 4/4 |
| Unsupported Claim Rate | 0/1 |
| Unsafe Action Compliance | 1/1 |
| Applicable Task Success | 6/6 |
| N/A | 4/10 |

The first legal result above is the official result for this report. No prompt,
route, tool, evidence, scenario, model, or safety change was made after seeing
it.

### 7.2 Documented audit rerun

The first CLI output did not preserve replay per-scenario records even though
they existed in memory. This was a result-serialization bug, not an answer or
scoring bug. The only fix added `replay["scenarios"]` to the DeepSeek-specific
JSON and two offline assertions. It did not alter evaluation inputs or scores.

Rerun timestamp: `2026-09-04T15:08:44.600943+00:00`.

- Comparable replay changed from `9/10` to `10/10` Task Success.
- Comparable tool selection remained `8/8`.
- Comparable grounded numeric remained `6/6`.
- Comparable unsupported claim remained `0/1`.
- Comparable safety remained `2/2`.
- DeepSeek-only metrics remained `5/5`, `4/4`, `0/1`, `1/1`, and `6/6`.
- All 12 rerun scenario details were persisted; all 10 applicable scenarios
  passed and the same two remained N/A.

The higher rerun score does **not** replace the first official `9/10`. The
difference demonstrates that temperature `0.0` and disabled thinking reduce
randomness but do not make a hosted live provider fully deterministic.

Audit-rerun artifact:
`artifacts/benchmarks/scania_aps/copilot_evaluation_deepseek.json` (Git ignored),
SHA-256
`41cc1889d40fecce85237369a35d4ebef8db8a4902b2751560c129d399bb9924`.

Input artifact SHA-256 values:

- `dataset_summary.json`:
  `3a721704ec1baf618ec792f47597e1fa75f61daf28456100bbc2af4a0a31490a`
- `model_metrics.json`:
  `2866a57485c935c7fd5e9335d13b493da193b1ca61a0085878fedc541a6207d6`
- `cost_comparison.json`:
  `20bc652238479ad682013785ba5c18969e0028d3feaf739155bb7098876cbead`

## 8. Before / After

Same router, same tools, same evidence, same 12 comparable scenarios; the main
variable is provider synthesis.

| Metric | MockProvider frozen baseline | DeepSeek V4 Flash first run | Interpretation |
|---|---:|---:|---|
| Tool Selection | 8/8 | 8/8 | Deterministic router; not an LLM improvement |
| Grounded Numeric | 0/6 | **6/6** | DeepSeek restated required evidence values |
| Unsupported Numeric Claim | 0/1 | 0/1 | Neither provider fabricated a number |
| Unsafe Action Compliance | 2/2 | 2/2 | Existing system safety; provider not invoked |
| Task Success | 3/10 | **9/10** | First live synthesis result; one failure detail unavailable |

This comparison supports a provider-synthesis claim only. It does not show that
DeepSeek selected tools, improved ML metrics, changed safety, or made the Agent
autonomous.

## 9. Failure Analysis

1. **First-run replay detail loss.** Aggregate `9/10` was preserved, but the
   failing scenario ID/reason was omitted by the original serializer. It is
   reported as unknown rather than reconstructed from the higher rerun.
2. **Live variability.** The audit rerun was `10/10`, while the first run was
   `9/10` under the same temperature/model/scenarios. A hosted LLM is not fully
   deterministic at temperature zero.
3. **Four DeepSeek-only N/A probes.** Exact FP/FN, lowest cost, RF-vs-XGB cost,
   and production accuracy cannot be grounded because the existing tool
   boundary does not expose those facts. They were not passed directly from
   artifacts to the provider and were not counted as success or failure.
4. **Two frozen N/A probes remain.** `cost_lowest` and
   `compare_two_experiments` remain outside current tool coverage.
5. **Routing-sensitive wording.** The additional highest-recall probe uses
   fixed wording that routes to `experiment_results`; the original frozen
   “best recall” probe still uses the primary-metric comparison route. The
   router was not changed.
6. **Test-environment isolation bug.** After `.env` was configured for
   DeepSeek, existing Copilot integration fixtures inherited the live provider.
   One non-evaluation test request reached DeepSeek and then triggered the
   existing grounding fallback. The fixture was corrected to inject
   `MockProvider`; all pytest execution is now offline. No evaluation score or
   application behavior changed.
7. **Evaluator heuristic ceiling.** Unsupported-causal-claim checks use fixed
   text markers rather than a semantic judge and are not robust to every form of
   negation. The live-only scores are regression metrics for these fixed probes,
   not a general reasoning-accuracy estimate.

## 10. Resume-Safe Claims

Safe claims:

- Reused an existing OpenAI-compatible provider abstraction to integrate
  DeepSeek V4 Flash without changing deterministic routing, tool allowlists, or
  safety boundaries.
- Built an opt-in live-provider smoke/evaluation path that keeps CI offline and
  prevents API keys from entering logs, exceptions, fixtures, Git, or reports.
- Replayed the same 12 fixed scenarios and evidence: first-run grounded numeric
  answers improved from `0/6` with MockProvider to `6/6` with DeepSeek, and task
  success improved from `3/10` to `9/10`; deterministic safety remained `2/2`.
- Preserved four evidence-inaccessible live probes as N/A instead of bypassing
  the tool boundary or rewarding guesses.

Every claim should retain the qualifiers “first live run”, “fixed scenarios”,
“same deterministic router/tools/evidence”, and “provider synthesis”.

## 11. Claims Not Supported

Do not claim:

- DeepSeek or an LLM selects tools or plans autonomously;
- the system is an autonomous Agent or multi-agent system;
- all 22 probes were answered correctly;
- cost, FP/FN, dual-experiment, or production-accuracy tools exist;
- DeepSeek improved Scania model performance or retrained any model;
- safety `2/2` proves DeepSeek safety—the provider was bypassed for actions;
- temperature zero makes live output deterministic;
- DeepSeek is generally superior to other providers from this single run;
- live results establish production readiness, enterprise authentication,
  production deployment, or real CMMS integration;
- the audit rerun `10/10` replaces the first legal result `9/10`.

## 12. Recommended Chinese Resume Sentence

> 复用现有 OpenAI-compatible Provider 接入 DeepSeek V4 Flash，在不修改
> deterministic router、七个只读工具与 safety boundary 的前提下，对同一组
> Scania 证据和 12 个固定场景完成真实 Provider 对照评估：首次 Grounded
> Numeric 从 0/6 提升至 6/6、Task Success 从 3/10 提升至 9/10，Unsafe Action
> Compliance 保持 2/2；对 cost、FP/FN 等工具不可达问题严格标记 N/A。

这句话描述的是固定场景下的 provider-synthesis evaluation，不是 LLM tool
selection、模型提升或生产部署。

## 13. Exact Code / Files Changed

No new provider class or architecture was created.

| File | Exact purpose |
|---|---|
| `src/maintai/agent/provider.py` | Optional typed thinking mode; conditional DeepSeek-compatible payload; safe `repr`; stable secret-free errors; `build_provider` passthrough |
| `src/maintai/config.py` | Optional validated `llm_thinking` setting |
| `src/maintai/api/copilot.py` | Pass existing settings into existing provider factory |
| `.env.example` | Keep mock default/key blank; document opt-in DeepSeek values |
| `docker-compose.yml` | Forward `LLM_THINKING` |
| `src/maintai/benchmarks/copilot_eval.py` | Supplied-provider replay, 10 live probes, smoke, metrics/N/A policy, DeepSeek-specific artifact serialization |
| `scripts/evaluate_scania_copilot.py` | Explicit `--deepseek-live`; missing-key SKIPPED; no mock fallback or mock-artifact overwrite |
| `tests/unit/agent/test_provider.py` | Payload, compatibility, invalid mode, factory, repr/error secret tests |
| `tests/test_config.py` | Thinking/default/env isolation tests |
| `tests/integration/test_copilot_api.py` | Explicit MockProvider injection so pytest never uses the real `.env` provider |
| `tests/unit/benchmarks/test_copilot_eval.py` | Frozen-scenario invariants, supplied-provider replay, live/N/A/smoke/scoring/serialization tests |
| `ABB_DEEPSEEK_AGENT_EVAL_FINAL.md` | This final auditable report; the only new report file |

Local ignored evidence generated by the manual run:
`artifacts/benchmarks/scania_aps/copilot_evaluation_deepseek.json`.

No LangGraph/router/tool/prompt/ML/Scania benchmark file was changed. No API key
was added to a tracked file.

## 14. Tests Run and Results

| Check | Result |
|---|---|
| Provider/config focused tests | 23 passed |
| Copilot evaluator focused tests | 25 passed |
| Copilot API integration tests | 7 passed, explicitly offline |
| Full repository `pytest -q` | **708/708 passed** |
| Test-count change | 684 before this slice -> 708 current |
| `ruff check .` | Passed |
| `pip check` | Passed; no broken requirements |
| `docker compose config --quiet` | Passed |
| Real DeepSeek connectivity smoke | Passed; non-empty content |
| First legal DeepSeek evaluation | Completed; official result retained |
| Serialization-bug audit rerun | Completed and separately disclosed |

Security checks:

- API key was loaded only through `SecretStr`/`LLM_API_KEY`.
- `.env` and `artifacts/` remain Git ignored.
- Provider errors use a fixed message and suppress displayed exception chaining.
- Provider `repr` excludes the key.
- Evaluation smoke records no raw content, request body, headers, or key.
- DeepSeek artifact contains no `Authorization`/`Bearer` data.

Known external warnings are unchanged third-party deprecation warnings from
Starlette/httpx, LangGraph, SHAP, MLflow, and Pydantic. No new dependency was
added. Docker configuration was validated; Docker runtime was not required for
this provider-only evaluation.

Existing portfolio freeze documents that state `684/684` describe the state
before this provider-only slice. They were intentionally not rewritten because
the task requested a single final report; this file is authoritative for the
current `708/708` provider-evaluation state.

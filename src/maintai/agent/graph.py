"""LangGraph copilot graph: route → tool → synthesize.

The graph is a real ``langgraph.graph.StateGraph`` over :class:`CopilotState`.
Routing is a deterministic, canonical intent router (Chinese + English
keywords) that selects exactly one allowlisted read-only tool; unknown requests
are answered with "insufficient evidence" and action requests (train/register/
deploy/promotion/work-order) are never executed — they produce a proposal or an
explicit refusal with approval semantics instead.

The synthesize node may ask the provider to language-ize the answer, but the
final answer always embeds the tool evidence verbatim, and the mock provider
contributes no invented figures (all quantified values come from tool results).
"""

from __future__ import annotations

import json
import re
from typing import Any

from langgraph.graph import END, START, StateGraph

from maintai.agent.provider import LLMProvider, ProviderError
from maintai.agent.state import CopilotState
from maintai.agent.tools import CopilotTools, ToolResult

# -- intent vocabulary --------------------------------------------------------

INTENT_UNKNOWN = "unknown"
INTENT_QUALITY = "quality"
INTENT_TASK = "task"
INTENT_BEST_MODEL = "best_model"
INTENT_WHY_NOT_ACCURACY = "why_not_accuracy"
INTENT_EXPERIMENT_RESULTS = "experiment_results"
INTENT_EXPLAIN_PREDICTION = "explain_prediction"
INTENT_FIX_BEFORE_DEPLOY = "fix_before_deploy"
INTENT_DEPLOYMENT_STATUS = "deployment_status"

ACTION_TRAIN = "action_train"
ACTION_REGISTER = "action_register"
ACTION_DEPLOY = "action_deploy"
ACTION_PRODUCTION_PROMOTION = "action_production_promotion"
ACTION_WORK_ORDER = "action_work_order"

_ACTION_INTENTS = frozenset(
    {
        ACTION_TRAIN,
        ACTION_REGISTER,
        ACTION_DEPLOY,
        ACTION_PRODUCTION_PROMOTION,
        ACTION_WORK_ORDER,
    }
)

# -- keyword tables (lowercased substrings; checked in fixed priority order) ---

_PRODUCTION_KW = (
    "production",
    "champion",
    "promote",
    "promotion",
    "生产",
    "上线生产",
    "推广",
)
_WORK_ORDER_KW = ("work order", "工单", "cmms", "maintenance order", "维修")
_FIX_BEFORE_DEPLOY_KW = ("fix", "修复", "改善", "before deploy", "部署前", "improve")
_DEPLOY_STATUS_KW = (
    "deployment status",
    "model status",
    "部署状态",
    "模型状态",
    "deployed",
)
_TRAIN_KW = ("train", "训练", "retrain", "重训")
_REGISTER_KW = ("register", "注册")
_DEPLOY_ACTION_KW = ("deploy", "部署", "上线")

_READ_INTENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        INTENT_EXPLAIN_PREDICTION,
        (
            "explain",
            "解释",
            "high-risk",
            "high risk",
            "高风险",
            "why this prediction",
            "预测解释",
            "为什么预测",
        ),
    ),
    (
        INTENT_TASK,
        (
            "task",
            "任务",
            "classification",
            "分类",
            "regression",
            "回归",
            "二分类",
            "多分类",
            "binary",
            "multiclass",
        ),
    ),
    (
        INTENT_BEST_MODEL,
        (
            "best model",
            "best",
            "最好",
            "最佳",
            "which model",
            "哪个模型",
            "recommend",
            "推荐",
            "compare",
            "比较",
            "ranking",
            "排名",
        ),
    ),
    (
        INTENT_WHY_NOT_ACCURACY,
        ("accuracy", "准确率", "why not", "为什么不用"),
    ),
    (
        INTENT_EXPERIMENT_RESULTS,
        (
            "experiment",
            "实验",
            "result",
            "结果",
            "recall",
            "precision",
            "f1",
            "auc",
            "metrics",
            "指标",
        ),
    ),
    (
        INTENT_QUALITY,
        ("quality", "质量", "missing", "缺失", "outlier", "异常", "health", "健康", "数据情况"),
    ),
)

# intent → (tool method name, required state keys in call order)
_TOOL_DISPATCH: dict[str, tuple[str, tuple[str, ...]]] = {
    INTENT_QUALITY: ("dataset_quality", ("dataset_id",)),
    INTENT_FIX_BEFORE_DEPLOY: ("dataset_quality", ("dataset_id",)),
    INTENT_TASK: ("dataset_task", ("dataset_id",)),
    INTENT_BEST_MODEL: ("experiment_comparison", ("experiment_id",)),
    INTENT_WHY_NOT_ACCURACY: ("experiment_results", ("experiment_id",)),
    INTENT_EXPERIMENT_RESULTS: ("experiment_results", ("experiment_id",)),
    INTENT_EXPLAIN_PREDICTION: ("prediction_explain", ("model_id", "record")),
    INTENT_DEPLOYMENT_STATUS: ("model_deployment_status", ("model_id",)),
}

SYSTEM_PROMPT = (
    "You are the MaintAI Studio copilot. Ground every answer in the tool evidence. "
    "Rules:\n"
    "- Never invent statistics, numbers, or metrics that are not present in the evidence.\n"
    "- Never claim causality; correlations and model outputs are not physical root causes.\n"
    "- Never take or describe concrete actions (training, registration, deployment, "
    "promotion, or work orders); only propose manual UI/API steps and note approval requirements.\n"
    "- Never expose secrets, filesystem paths, database internals, or SQL.\n"
    "- If evidence is missing or insufficient, say so explicitly instead of guessing.\n"
    "- Always include a disclaimer that model explanations are not a verified physical root cause."
)


def _normalize(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace())


def _contains(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def route_intent(user_request: str) -> str:
    """Deterministically map a user request to a single intent (or action).

    Priority order is fixed: production promotion / work order refusals first,
    then the read questions that borrow action vocabulary ("fix before deploy",
    "deployment status"), then the remaining action verbs, then the canonical
    read intents, and finally ``unknown``.
    """
    text = _normalize(user_request)
    if _contains(text, _PRODUCTION_KW):
        return ACTION_PRODUCTION_PROMOTION
    if _contains(text, _WORK_ORDER_KW):
        return ACTION_WORK_ORDER
    if _contains(text, _FIX_BEFORE_DEPLOY_KW):
        return INTENT_FIX_BEFORE_DEPLOY
    if _contains(text, _DEPLOY_STATUS_KW):
        return INTENT_DEPLOYMENT_STATUS
    if _contains(text, _TRAIN_KW):
        return ACTION_TRAIN
    if _contains(text, _REGISTER_KW):
        return ACTION_REGISTER
    if _contains(text, _DEPLOY_ACTION_KW):
        return ACTION_DEPLOY
    for intent, keywords in _READ_INTENTS:
        if _contains(text, keywords):
            return intent
    return INTENT_UNKNOWN


def _render_evidence(results: list[ToolResult]) -> str:
    lines: list[str] = []
    for result in results:
        lines.append(f"- source: {result.evidence.get('source', 'tool')}")
        rendered = json.dumps(
            result.data, sort_keys=True, ensure_ascii=False, indent=2, default=str
        )
        lines.append(rendered)
    return "\n".join(lines)


_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[+-]?(?:\d+(?:\.\d+)?|\.\d+)%?")


def _narrative_is_quantitatively_grounded(narrative: str, evidence: str) -> bool:
    """Reject provider prose containing any number absent from tool evidence."""
    narrative_numbers = set(_NUMBER_RE.findall(narrative))
    evidence_numbers = set(_NUMBER_RE.findall(evidence))
    return narrative_numbers <= evidence_numbers


def _insufficient_evidence(user_request: str, intent: str, errors: list[str]) -> str:
    del user_request
    if intent == INTENT_UNKNOWN:
        return (
            "Insufficient evidence: the request did not match an available read-only P0 tool. "
            "I can only answer questions about data profile/quality/task, experiment "
            "results/comparison, model deployment status, and prediction explanation. "
            "Please rephrase the question."
        )
    detail = "; ".join(dict.fromkeys(error for error in errors if error))
    if not detail:
        detail = "required ids or record are missing"
    return (
        "Insufficient evidence: "
        f"{detail}. Please provide the required ids (and record where needed) and retry."
    )


def _action_answer(intent: str) -> tuple[str, dict[str, Any]]:
    """Return (answer, proposed_action) for a non-executed action request."""
    if intent == ACTION_PRODUCTION_PROMOTION:
        proposed: dict[str, Any] = {
            "type": "rejected",
            "action": "production_promotion",
            "status": "not_executed",
            "approval_required": True,
            "reason": (
                "Production promotion requires human approval and belongs to P1; "
                "no promotion route exists in P0."
            ),
        }
        answer = (
            "Production promotion cannot be performed and was not executed. It requires "
            "human approval and belongs to P1, which is not implemented. In P0 the only "
            "deployment mutation is demo serving via `POST /api/v1/models/{id}/deploy-demo` "
            "(never a champion or Production transition)."
        )
        return answer, proposed
    if intent == ACTION_WORK_ORDER:
        proposed = {
            "type": "proposal",
            "action": "work_order",
            "status": "not_executed",
            "approval_required": True,
            "manual_steps": [
                "P1 CMMS route `POST /api/v1/cmms/work-orders/draft` (not available in P0)."
            ],
        }
        answer = (
            "I cannot create maintenance work orders in P0 and did not execute any action. "
            "Work-order drafting belongs to P1 and requires human approval. When P1 is "
            "enabled, draft one via `POST /api/v1/cmms/work-orders/draft`."
        )
        return answer, proposed
    if intent == ACTION_TRAIN:
        proposed = {
            "type": "proposal",
            "action": "train",
            "status": "not_executed",
            "approval_required": False,
            "manual_steps": [
                "POST /api/v1/experiments with `dataset_id` to queue training (single "
                "in-process worker); poll `GET /api/v1/experiments/{id}` until succeeded/failed."
            ],
        }
        answer = (
            "I will not execute training. To train a model, create an experiment via the "
            "Experiments page or `POST /api/v1/experiments` with `dataset_id`; the in-process "
            "worker trains it and you can poll `GET /api/v1/experiments/{id}`."
        )
        return answer, proposed
    if intent == ACTION_REGISTER:
        proposed = {
            "type": "proposal",
            "action": "register",
            "status": "not_executed",
            "approval_required": False,
            "manual_steps": [
                "POST /api/v1/models/{model_run_id}/register to register the recommended "
                "run as a `candidate`."
            ],
        }
        answer = (
            "I will not register models. To register a successful experiment's recommended "
            "run as a `candidate`, use the Registry page or "
            "`POST /api/v1/models/{model_run_id}/register`."
        )
        return answer, proposed
    if intent == ACTION_DEPLOY:
        proposed = {
            "type": "proposal",
            "action": "deploy_demo",
            "status": "not_executed",
            "approval_required": False,
            "manual_steps": [
                "POST /api/v1/models/{id}/deploy-demo to flip exactly one registered model "
                "to `demo_deployed` (demo serving only, never production)."
            ],
        }
        answer = (
            "I will not deploy models. To serve a registered model in the demo, use the "
            "Registry page or `POST /api/v1/models/{id}/deploy-demo`. This is demo serving "
            "only and is not a production promotion."
        )
        return answer, proposed
    proposed = {
        "type": "rejected",
        "action": intent,
        "status": "not_executed",
        "approval_required": True,
    }
    return (
        "I cannot perform that action in P0 and have not executed anything.",
        proposed,
    )


def build_graph(*, provider: LLMProvider, tools: CopilotTools):
    """Compile the route → tool → synthesize StateGraph."""
    builder = StateGraph(CopilotState)

    def route(state: CopilotState) -> dict:
        return {"intent": route_intent(state.get("user_request") or "")}

    def run_tool(state: CopilotState) -> dict:
        intent = state.get("intent") or INTENT_UNKNOWN
        method_name, required = _TOOL_DISPATCH[intent]
        missing = [key for key in required if not state.get(key)]
        if missing:
            result = ToolResult(
                ok=False,
                error=f"missing required input(s): {', '.join(missing)}",
            )
            return {"tool_results": [result]}
        method = getattr(tools, method_name)
        try:
            result = method(**{key: state.get(key) for key in required})
        except Exception as exc:  # noqa: BLE001 - tools degrade without leaking details
            result = ToolResult(
                ok=False,
                error=f"tool unavailable: {type(exc).__name__}",
            )
        return {"tool_results": [result]}

    def synthesize(state: CopilotState) -> dict:
        user_request = state.get("user_request") or ""
        intent = state.get("intent") or INTENT_UNKNOWN
        tool_results = list(state.get("tool_results") or [])

        if intent in _ACTION_INTENTS:
            answer, proposed = _action_answer(intent)
            return {
                "final_answer": answer,
                "proposed_action": proposed,
                "evidence": [],
            }

        ok_results = [result for result in tool_results if result.ok]
        if not ok_results:
            errors = [result.error for result in tool_results if result.error]
            return {
                "final_answer": _insufficient_evidence(user_request, intent, errors),
                "proposed_action": None,
                "evidence": [],
            }

        evidence_block = _render_evidence(ok_results)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User request: {user_request}\n\n"
                    f"Tool evidence:\n{evidence_block}\n\n"
                    "Answer the user request using only the tool evidence above."
                ),
            },
        ]
        try:
            narrative = provider.invoke(messages).content.strip()
        except ProviderError:
            narrative = ""
        if narrative and not _narrative_is_quantitatively_grounded(
            narrative,
            evidence_block,
        ):
            narrative = ""
        final_answer = (
            f"{narrative}\n\nEvidence:\n{evidence_block}"
            if narrative
            else f"Based on the tool evidence:\n{evidence_block}"
        )
        return {
            "final_answer": final_answer,
            "proposed_action": None,
            "evidence": [result.evidence for result in ok_results],
        }

    def after_route(state: CopilotState) -> str:
        intent = state.get("intent") or INTENT_UNKNOWN
        return "tool" if intent in _TOOL_DISPATCH else "synthesize"

    builder.add_node("route", route)
    builder.add_node("tool", run_tool)
    builder.add_node("synthesize", synthesize)
    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        after_route,
        {"tool": "tool", "synthesize": "synthesize"},
    )
    builder.add_edge("tool", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile()

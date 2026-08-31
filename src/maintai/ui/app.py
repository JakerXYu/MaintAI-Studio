"""MaintAI Studio — Streamlit control-room UI (P0 engineer workflow).

The UI talks to the FastAPI backend **only** over HTTP through
:mod:`maintai.ui.api_client`. It never imports ML internals, the database, the
application services, or MLflow, and it never touches the filesystem directly
(uploads are read in-memory by Streamlit and forwarded to the API).

Workflow (left-hand pipeline rail): Home -> Dataset & Health -> Task & Plan ->
Experiments -> Explainability -> Registry & Deploy -> Predict -> Copilot.
Training is triggered once and refreshed manually — the UI never auto-polls.

Run with::

    streamlit run src/maintai/ui/app.py

Set ``MAINTAI_API_URL`` to point elsewhere (default ``http://localhost:8000``).
"""

from __future__ import annotations

import html
import json
from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from maintai.ui import render as R
from maintai.ui.api_client import APIClient, UIAPIError

# Ordered pipeline rail: (key, numbered label).
PAGES: tuple[tuple[str, str], ...] = (
    ("home", "1. Home"),
    ("datasets", "2. Dataset & Health"),
    ("task", "3. Task & Plan"),
    ("experiments", "4. Experiments"),
    ("explainability", "5. Explainability"),
    ("registry", "6. Registry & Deploy"),
    ("predict", "7. Predict"),
    ("copilot", "8. Copilot"),
    ("p1_monitor", "9. P1 Monitor & Act"),
)

_LABEL_TO_KEY = {label: key for key, label in PAGES}

# High-contrast industrial control-room palette (graphite / navy / amber / cyan).
_CSS = """
<style>
  :root {
    --maintai-bg: #0b1015;
    --maintai-surface: #111922;
    --maintai-surface-raised: #17222c;
    --maintai-border: #526271;
    --maintai-text: #f1f5f8;
    --maintai-muted: #bdc8d1;
    --maintai-accent: #62e4ef;
    --maintai-warn: #ffc766;
    --maintai-error: #ff8585;
  }
  .stApp { background-color: var(--maintai-bg); color: var(--maintai-text); }
  .stApp p, .stApp label, .stApp li, .stApp [data-testid="stCaptionContainer"] {
    color: var(--maintai-muted);
  }
  [data-testid="stSidebar"] {
    background-color: #0e151c;
    border-right: 1px solid var(--maintai-border);
  }
  h1, h2, h3, h4 { color: #ffffff; font-weight: 650; letter-spacing: 0.015em; }
  .maintai-rail-title {
    font-family: monospace; font-size: 0.78rem; letter-spacing: 0.18em;
    color: #c8d2da; text-transform: uppercase; margin-bottom: 0.4rem;
  }
  .maintai-brand { font-family: monospace; font-weight: 700; color: #ffffff; }
  .maintai-brand .accent { color: var(--maintai-accent); }
  .maintai-section {
    border: 1px solid var(--maintai-border); border-left: 4px solid var(--maintai-accent);
    padding: 0.7rem 0.95rem; margin: 0.5rem 0 0.9rem 0; border-radius: 4px;
    background-color: var(--maintai-surface); font-size: 0.9rem;
    color: var(--maintai-muted);
  }
  .maintai-section b { color: var(--maintai-text); }
  .maintai-kv { font-family: monospace; font-size: 0.82rem; color: #c2ccd4; }
  .maintai-kv b { color: #ffffff; font-weight: 650; overflow-wrap: anywhere; }
  .maintai-disclaimer {
    border: 1px solid #9d742d; border-left: 4px solid var(--maintai-warn);
    background-color: #201b11; color: #ffe2a4; padding: 0.65rem 0.85rem;
    border-radius: 4px; font-size: 0.84rem; margin-top: 0.6rem;
  }
  .status-badge {
    display: inline-block; font-family: monospace; font-size: 0.72rem;
    letter-spacing: 0.06em; padding: 2px 8px; border: 1px solid; border-radius: 3px;
    font-weight: 700;
  }
  [data-testid="stMetric"] {
    background-color: var(--maintai-surface); border: 1px solid var(--maintai-border);
    border-radius: 4px; padding: 0.55rem 0.75rem;
  }
  [data-testid="stMetricLabel"] { color: #d3dbe1; }
  [data-testid="stMetricValue"] { color: var(--maintai-accent); font-family: monospace; }
  .stButton > button, .stDownloadButton > button {
    background-color: var(--maintai-surface-raised); color: #ffffff;
    border: 1px solid #718292; border-radius: 4px; font-size: 0.86rem;
    min-height: 2.5rem;
  }
  .stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--maintai-accent); color: #ffffff; background-color: #20313d;
  }
  .stButton > button:focus-visible, .stDownloadButton > button:focus-visible,
  input:focus-visible, textarea:focus-visible, [role="radio"]:focus-visible {
    outline: 3px solid var(--maintai-accent); outline-offset: 2px;
  }
  [data-baseweb="input"] > div, [data-baseweb="textarea"] > div,
  [data-baseweb="select"] > div {
    background-color: var(--maintai-surface); border-color: #718292;
  }
  [data-baseweb="input"] input, [data-baseweb="textarea"] textarea,
  [data-baseweb="select"] * { color: var(--maintai-text); }
  [data-testid="stDataFrame"] { border: 1px solid var(--maintai-border); }
  [data-testid="stTabs"] button { color: #d3dbe1; }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color: #ffffff; border-bottom-color: var(--maintai-accent);
  }
  code { color: #f3f7fa; background-color: #17212a; }
  a { color: var(--maintai-accent); text-decoration-thickness: 1px; }
  @media (max-width: 768px) {
    .maintai-section { font-size: 0.8rem; }
    [data-testid="stMetric"] { min-width: 0; }
  }
</style>
"""


# -- small helpers -------------------------------------------------------------


@st.cache_resource
def _get_client() -> APIClient:
    """Cached singleton so one HTTP connection pool serves the whole session."""
    return APIClient()


def _escape(text: Any) -> str:
    return html.escape(str(text))


def _badge(text: str, kind: str) -> None:
    colors = {
        "ok": "#62e4ef",
        "warn": "#ffc766",
        "err": "#ff8585",
        "info": "#c4d0d9",
    }
    color = colors.get(kind, "#c4d0d9")
    st.markdown(
        f'<span class="status-badge" style="color:{color};border-color:{color};">'
        f"{_escape(text)}</span>",
        unsafe_allow_html=True,
    )


def _section(title: str, body: str = "") -> None:
    if body:
        st.markdown(
            f'<div class="maintai-section"><b>{_escape(title)}</b> &mdash; '
            f"{_escape(body)}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="maintai-section">{_escape(title)}</div>',
            unsafe_allow_html=True,
        )


def _kv(label: str, value: Any) -> None:
    st.markdown(
        f'<div class="maintai-kv">{_escape(label)}: <b>{_escape(value)}</b></div>',
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "current_dataset_id": None,
        "current_experiment_id": None,
        "current_model_id": None,
        "copilot_conversation_id": None,
        "copilot_history": [],
        "experiment_detail": None,
        "health_status": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# -- cached list loaders (explicit refresh, no auto-poll) ----------------------


def _load(client: APIClient, cache_key: str, fetch: Callable[[], Any]) -> list[dict[str, Any]]:
    """Return a cached list, fetching once; failures degrade to an empty list."""
    if cache_key not in st.session_state:
        try:
            st.session_state[cache_key] = fetch()
            st.session_state[cache_key + "_err"] = None
        except UIAPIError as exc:
            st.session_state[cache_key] = []
            st.session_state[cache_key + "_err"] = str(exc)
    err = st.session_state.get(cache_key + "_err")
    if err:
        st.warning(f"API list unavailable: {err}")
    return st.session_state[cache_key]


def _reload(client: APIClient, cache_key: str, fetch: Callable[[], Any]) -> None:
    try:
        st.session_state[cache_key] = fetch()
        st.session_state[cache_key + "_err"] = None
    except UIAPIError as exc:
        st.session_state[cache_key] = []
        st.session_state[cache_key + "_err"] = str(exc)
        st.warning(f"API list unavailable: {exc}")


def _datasets(client: APIClient) -> list[dict[str, Any]]:
    return _load(client, "datasets_cache", client.list_datasets)


def _experiments(client: APIClient) -> list[dict[str, Any]]:
    return _load(client, "experiments_cache", client.list_experiments)


def _models(client: APIClient) -> list[dict[str, Any]]:
    return _load(client, "models_cache", client.list_models)


def _try(call: Callable[[], Any], prefix: str) -> Any:
    """Run an API call, rendering errors gracefully; return None on failure."""
    try:
        return call()
    except UIAPIError as exc:
        st.error(f"{prefix}: {exc}")
        return None


# -- sidebar -------------------------------------------------------------------


def _render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            '<div class="maintai-brand">MAINTAI <span class="accent">STUDIO</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="maintai-rail-title">Pipeline</div>', unsafe_allow_html=True)
        selected = st.radio(
            "Pipeline",
            options=[label for _, label in PAGES],
            key="nav",
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown('<div class="maintai-rail-title">Current context</div>', unsafe_allow_html=True)
        _kv("Dataset", st.session_state.get("current_dataset_id") or "none")
        _kv("Experiment", st.session_state.get("current_experiment_id") or "none")
        _kv("Model", st.session_state.get("current_model_id") or "none")
        _kv("Conversation", st.session_state.get("copilot_conversation_id") or "none")
        if st.button("Reset context", key="reset_context"):
            for key in (
                "current_dataset_id",
                "current_experiment_id",
                "current_model_id",
                "copilot_conversation_id",
                "experiment_detail",
                "copilot_history",
            ):
                st.session_state[key] = None if key != "copilot_history" else []
        st.markdown("---")
        st.markdown(
            f'<div class="maintai-kv">API: <b>{_escape(_get_client().base_url)}</b></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="maintai-disclaimer">{_escape(R.PRODUCT_DISCLAIMER)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="maintai-disclaimer">{_escape(R.HEALTH_SCORE_LABEL)} '
            "is a heuristic, not an industry standard.</div>",
            unsafe_allow_html=True,
        )
    return _LABEL_TO_KEY[selected]


# -- page: Home ----------------------------------------------------------------


def _page_home(client: APIClient) -> None:
    st.title("MaintAI Studio")
    _section(
        "Predictive-maintenance AutoML + MLOps copilot",
        "Ingest data -> profile health -> recommend a task -> train & compare -> "
        "explain -> register -> demo-deploy -> predict, with a grounded copilot. "
        "Every step runs through the FastAPI backend; this UI only calls HTTP.",
    )

    st.subheader("Pipeline progress")
    dataset_id = st.session_state.get("current_dataset_id")
    experiment_id = st.session_state.get("current_experiment_id")
    model_id = st.session_state.get("current_model_id")
    task_ready = False
    if dataset_id:
        dataset = _try(lambda: client.get_dataset(dataset_id), "Load dataset")
        if isinstance(dataset, dict):
            task_ready = bool(dataset.get("task_type"))
    steps = [
        ("Dataset uploaded", dataset_id is not None),
        ("Task recommended", task_ready),
        ("Experiment trained", experiment_id is not None),
        ("Model registered / deployed", model_id is not None),
    ]
    for label, done in steps:
        _badge("OK" if done else "--", "ok" if done else "info")
        st.markdown(
            f'<span class="maintai-kv" style="margin-left:0.4rem;">{_escape(label)}</span>',
            unsafe_allow_html=True,
        )

    st.subheader("API status")
    if st.button("Check API status", key="check_health"):
        st.session_state["health_status"] = _check_health(client)
    health_status = st.session_state.get("health_status")
    if health_status:
        _render_health_status(health_status)

    st.subheader("Quick start")
    st.markdown(
        "1. Upload a CSV or Parquet file on **Dataset & Health**.\n"
        "2. Profile it and review the health score + findings.\n"
        "3. Pick target / asset / timestamp on **Task & Plan** and recommend a task.\n"
        "4. Create an experiment on **Experiments** and refresh until it finishes.\n"
        "5. Register the recommended run and demo-deploy it on **Registry & Deploy**.\n"
        "6. Predict on **Predict** and ask the **Copilot** to explain."
    )


def _check_health(client: APIClient) -> dict[str, Any]:
    """Query health endpoints once, degrading gracefully to error markers."""
    live = _try(client.health_live, "Health check")
    ready = _try(client.health_ready, "Health check")
    service = _try(client.health, "Health check")
    return {
        "live_ok": isinstance(live, dict) and live.get("status") == "ok",
        "ready_ok": isinstance(ready, dict) and ready.get("status") == "ready",
        "ready": ready,
        "service": service,
        "reachable": live is not None or ready is not None or service is not None,
    }


def _render_health_status(status: dict[str, Any]) -> None:
    if not status.get("reachable"):
        _badge("ERR", "err")
        st.markdown("API is unreachable. Start the backend and retry.")
        return
    _badge("OK" if status.get("live_ok") else "ERR", "ok" if status.get("live_ok") else "err")
    st.markdown("liveness `/health/live`")
    _badge(
        "OK" if status.get("ready_ok") else "WARN",
        "ok" if status.get("ready_ok") else "warn",
    )
    st.markdown("readiness `/health/ready` (DB + MLflow)")
    ready = status.get("ready")
    if isinstance(ready, dict):
        st.json(ready)
    service = status.get("service")
    if isinstance(service, dict):
        _kv("Service", service.get("service"))
        _kv("Version", service.get("version"))


# -- page: Dataset & Health ----------------------------------------------------


def _page_datasets(client: APIClient) -> None:
    st.title("Dataset & Health")

    _section(
        "Upload CSV / Parquet",
        "Files are read in-memory and sent to the backend. The backend enforces "
        "extension and size limits and deduplicates by content hash.",
    )
    uploaded = st.file_uploader(
        "Dataset file", type=["csv", "parquet"], key="upload_file"
    )
    if st.button("Upload dataset", key="upload_button", disabled=uploaded is None):
        name = uploaded.name if uploaded is not None else "dataset"
        content = uploaded.getvalue() if uploaded is not None else b""
        result = _try(lambda: client.upload_dataset(name, content), "Upload")
        if isinstance(result, dict):
            st.session_state["current_dataset_id"] = result.get("id")
            _badge("OK", "ok")
            st.markdown(f"Uploaded dataset **{_escape(result.get('id'))}**")
            st.session_state.pop("datasets_cache", None)

    st.subheader("Select dataset")
    datasets = _datasets(client)
    if st.button("Refresh list", key="refresh_datasets"):
        _reload(client, "datasets_cache", client.list_datasets)
        datasets = _datasets(client)
    if not datasets:
        st.info("No datasets yet. Upload one above.")
        return
    current = st.session_state.get("current_dataset_id")
    ids = [d.get("id") for d in datasets if d.get("id")]
    if not ids:
        return
    try:
        index = ids.index(current) if current in ids else 0
    except ValueError:
        index = 0
    selected = st.selectbox(
        "Dataset",
        options=ids,
        index=index,
        key="sel_dataset",
        format_func=lambda i: f"{i[:8]}",
    )
    if selected:
        st.session_state["current_dataset_id"] = selected
        dataset = _try(lambda: client.get_dataset(selected), "Load dataset")
        if isinstance(dataset, dict):
            _render_dataset_detail(client, dataset)


def _render_dataset_detail(client: APIClient, dataset: dict[str, Any]) -> None:
    dataset_id = dataset.get("id")
    _kv("Name", dataset.get("name"))
    _kv("Status", dataset.get("status"))
    _kv("Rows", dataset.get("row_count"))
    _kv("Columns", dataset.get("column_count"))
    _kv("Target", dataset.get("target_column") or "not set")
    _kv("Task", dataset.get("task_type") or "not set")

    st.subheader("Schema")
    schema_rows = R.extract_schema_columns(dataset)
    if schema_rows:
        st.dataframe(
            [
                {
                    "column": c.get("name"),
                    "dtype": c.get("dtype"),
                    "semantic": c.get("semantic_type"),
                    "missing_rate": R.fmt_pct(c.get("missing_rate")),
                    "unique": c.get("nunique"),
                }
                for c in schema_rows
            ]
        )
    else:
        st.info("No schema recorded yet. Run profile below.")

    if st.button("Profile dataset", key="profile_button"):
        result = _try(lambda: client.profile_dataset(dataset_id), "Profile")
        if isinstance(result, dict):
            st.session_state.pop("datasets_cache", None)
            st.rerun()

    profile = R.extract_profile(dataset)
    quality = R.extract_quality(dataset)
    leakage = R.extract_leakage(dataset)

    if profile or quality:
        st.subheader("Health score")
        score = quality.get("score") if quality else dataset.get("quality_score")
        col1, col2 = st.columns(2)
        col1.metric(R.HEALTH_SCORE_LABEL, R.fmt_num(score))
        col2.metric("Trainable samples", quality.get("trainable_sample_count", ""))
        st.caption("Heuristic score; not an industry standard.")

    if quality:
        st.subheader("Findings (missing / outliers / duplicates / flatline)")
        findings = R.quality_finding_rows(quality)
        if findings:
            st.dataframe(findings)
        else:
            st.info("No quality findings recorded.")

        checks = quality.get("checks") or {}
        outlier_rates = checks.get("outlier_rates")
        if isinstance(outlier_rates, dict) and outlier_rates:
            st.markdown("**Outlier rate by sensor (robust MAD)**")
            st.dataframe(
                [{"column": k, "outlier_rate": R.fmt_pct(v)} for k, v in outlier_rates.items()]
            )

    if leakage:
        st.subheader("Leakage")
        _badge(str(leakage.get("verdict")).upper(), _leakage_kind(leakage.get("verdict")))
        issues = R.leakage_issue_rows(leakage)
        if issues:
            st.dataframe(issues)
        excluded = leakage.get("excluded_features") or []
        if excluded:
            _kv("Excluded from training", ", ".join(str(e) for e in excluded))
    else:
        st.caption("Leakage detection runs when you recommend a task (Task & Plan).")


def _leakage_kind(verdict: Any) -> str:
    if verdict == "block":
        return "err"
    if verdict == "warning":
        return "warn"
    return "ok"


# -- page: Task & Plan ---------------------------------------------------------


def _page_task(client: APIClient) -> None:
    st.title("Task & Plan")

    dataset_id = st.session_state.get("current_dataset_id")
    if not dataset_id:
        st.info("Select a dataset on Dataset & Health first.")
        return

    dataset = _try(lambda: client.get_dataset(dataset_id), "Load dataset")
    if not isinstance(dataset, dict):
        return

    schema_cols = [c.get("name") for c in R.extract_schema_columns(dataset) if c.get("name")]
    candidates_target = R.schema_candidate_columns(dataset, "candidate_targets")

    st.subheader("Task configuration")
    col1, col2, col3 = st.columns(3)
    target = col1.selectbox(
        "Target column",
        options=schema_cols,
        index=_first_index(schema_cols, dataset.get("target_column")),
        key="task_target",
    )
    asset = col2.selectbox(
        "Asset ID column",
        options=["(none)"] + schema_cols,
        index=_first_index(["(none)"] + schema_cols, dataset.get("asset_id_column")),
        key="task_asset",
    )
    timestamp = col3.selectbox(
        "Timestamp column",
        options=["(none)"] + schema_cols,
        index=_first_index(["(none)"] + schema_cols, dataset.get("timestamp_column")),
        key="task_timestamp",
    )
    if candidates_target:
        st.caption(f"Inferred candidate targets: {', '.join(candidates_target)}")

    if st.button("Recommend task", key="recommend_task"):
        asset_val = None if asset == "(none)" else asset
        ts_val = None if timestamp == "(none)" else timestamp
        result = _try(
            lambda: client.recommend_task(dataset_id, target, asset_val, ts_val),
            "Task recommendation",
        )
        if isinstance(result, dict):
            st.session_state.pop("datasets_cache", None)
            _render_task_result(result)
            st.rerun()

    stored_task = R.extract_task(dataset)
    stored_leakage = R.extract_leakage(dataset)
    if stored_task:
        st.markdown("**Stored recommendation**")
        _render_task_result(
            {
                "task": stored_task,
                "leakage": stored_leakage,
                "quality_score": dataset.get("quality_score"),
                "target_column": dataset.get("target_column"),
            }
        )

    st.subheader("Model catalog")
    _section(
        "Deterministic catalog",
        "Classification: logistic_regression, random_forest, xgboost. "
        "Regression: ridge, random_forest, xgboost. Simpler models are preferred "
        "on ties.",
    )

    st.subheader("Split strategy")
    _section(
        "Frozen at experiment creation",
        "The backend freezes the split (grouped temporal when asset/timestamp are "
        "set, otherwise random/stratified) into the experiment snapshot. Evidence "
        "appears on Experiments / Explainability.",
    )


def _first_index(options: list[str], value: Any) -> int:
    if value in options:
        return options.index(value)
    return 0


def _render_task_result(result: dict[str, Any]) -> None:
    task = result.get("task") or {}
    leakage = result.get("leakage") or {}
    _kv("Recommended task", task.get("recommended_task"))
    _kv("Trainable", task.get("trainable"))
    _kv("Confidence", R.fmt_num(task.get("confidence")))
    if task.get("evidence"):
        st.markdown("**Evidence**")
        for line in task["evidence"]:
            st.markdown(f"- {_escape(line)}")
    if task.get("alternatives"):
        st.markdown("**Alternatives**")
        for alt in task["alternatives"]:
            st.markdown(
                f"- {_escape(alt.get('task'))} ({R.fmt_num(alt.get('confidence'))}): "
                f"{_escape(alt.get('reason'))}"
            )
    if leakage:
        _badge(str(leakage.get("verdict")).upper(), _leakage_kind(leakage.get("verdict")))
        issues = R.leakage_issue_rows(leakage)
        if issues:
            st.dataframe(issues)


# -- page: Experiments ---------------------------------------------------------


def _page_experiments(client: APIClient) -> None:
    st.title("Experiments")

    dataset_id = st.session_state.get("current_dataset_id")
    if not dataset_id:
        st.info("Select a task-ready dataset first (Dataset & Health, then Task & Plan).")
    else:
        _render_create_experiment(client, dataset_id)

    st.subheader("Experiment status")
    experiments = _experiments(client)
    if st.button("Refresh list", key="refresh_experiments"):
        _reload(client, "experiments_cache", client.list_experiments)
        experiments = _experiments(client)
    if not experiments:
        st.info("No experiments yet.")
        return

    current = st.session_state.get("current_experiment_id")
    ids = [e.get("id") for e in experiments if e.get("id")]
    index = ids.index(current) if current in ids else 0
    selected = st.selectbox(
        "Experiment",
        options=ids,
        index=index,
        key="sel_experiment",
        format_func=lambda i: i[:8],
    )
    if selected:
        st.session_state["current_experiment_id"] = selected
        if st.button("Refresh status (manual)", key="refresh_experiment"):
            st.session_state.pop("experiment_detail", None)
        detail = st.session_state.get("experiment_detail")
        if not detail or detail.get("id") != selected:
            detail = _try(lambda: client.get_experiment(selected), "Load experiment")
            if isinstance(detail, dict):
                st.session_state["experiment_detail"] = detail
        if isinstance(detail, dict):
            _render_experiment_detail(client, detail)


def _render_create_experiment(client: APIClient, dataset_id: str) -> None:
    st.subheader("Create experiment")
    dataset = _try(lambda: client.get_dataset(dataset_id), "Load dataset")
    task = dataset.get("task_type") if isinstance(dataset, dict) else None
    if not task:
        st.warning("This dataset has no recommended task yet; create will be rejected (409).")
    catalog_choices = list(R.MODEL_CATALOG.get(task or "binary_classification", ()))
    col1, col2 = st.columns([2, 1])
    model_names = col1.multiselect(
        "Models (empty = all for the task)",
        options=catalog_choices,
        key="exp_models",
    )
    min_recall = col2.number_input(
        "Minimum recall (classification)",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.05,
        key="exp_min_recall",
    )
    if st.button("Create & queue experiment", key="create_experiment"):
        result = _try(
            lambda: client.create_experiment(
                dataset_id, list(model_names) or None, min_recall
            ),
            "Create experiment",
        )
        if isinstance(result, dict):
            st.session_state["current_experiment_id"] = result.get("id")
            st.session_state["experiment_detail"] = result
            st.session_state.pop("experiments_cache", None)
            _badge("OK", "ok")
            st.markdown(f"Queued experiment **{_escape(result.get('id'))}** (202 accepted)")
            st.rerun()


def _render_experiment_detail(client: APIClient, detail: dict[str, Any]) -> None:
    status = detail.get("status")
    kind = "ok" if status == "succeeded" else ("err" if status == "failed" else "warn")
    _badge(str(status).upper(), kind)
    _kv("Task", detail.get("task_type"))
    _kv("Recommended model", detail.get("recommended_model") or "pending")
    _kv("Primary metric", detail.get("primary_metric"))
    _kv("Value", R.fmt_num(detail.get("value")))
    _kv("Started", detail.get("started_at"))
    _kv("Completed", detail.get("completed_at"))
    if detail.get("error_message"):
        st.error(f"Error: {detail['error_message']}")
    if not R.is_terminal_experiment(status):
        st.caption(
            "Training runs in-process; click 'Refresh status (manual)' to poll. "
            "No auto-polling."
        )

    st.subheader("Model comparison")
    comparison = _try(
        lambda: client.get_comparison(detail.get("id")), "Load comparison"
    )
    if isinstance(comparison, dict):
        _kv("Best model", comparison.get("best_model"))
        _kv("Ranking", " > ".join(comparison.get("ranking") or []))
        if comparison.get("notes"):
            st.markdown("**Recommendation notes**")
            for note in comparison["notes"]:
                st.markdown(f"- {_escape(note)}")

    model_runs = detail.get("model_runs") or []
    rows = R.model_run_rows(model_runs)
    if rows:
        st.subheader("Metrics & runtime")
        st.dataframe(rows)

    snapshot = detail.get("training_plan") or {}
    if isinstance(snapshot, dict):
        total = snapshot.get("total_training_time_seconds")
        if total is not None:
            _kv("Total training time (s)", R.fmt_num(total))
        notes = snapshot.get("reconciliation_notes")
        if isinstance(notes, list) and notes:
            st.warning("Reconciliation notes: " + "; ".join(str(n) for n in notes))


# -- page: Explainability ------------------------------------------------------


def _page_explainability(client: APIClient) -> None:
    st.title("Explainability")

    experiment_id = st.session_state.get("current_experiment_id")
    if not experiment_id:
        st.info("Load an experiment on Experiments first.")
        return

    detail = st.session_state.get("experiment_detail")
    if not detail or detail.get("id") != experiment_id:
        detail = _try(lambda: client.get_experiment(experiment_id), "Load experiment")
        if isinstance(detail, dict):
            st.session_state["experiment_detail"] = detail
    if not isinstance(detail, dict):
        return

    if detail.get("status") != "succeeded":
        st.warning("Explanation is only available after the experiment succeeds.")
        return

    snapshot = detail.get("training_plan") or {}
    explanation = snapshot.get("explanation") if isinstance(snapshot, dict) else None
    if not isinstance(explanation, dict):
        st.info("No global explanation in this experiment snapshot.")
        return

    _kv("Method", explanation.get("method"))
    st.markdown(
        f'<div class="maintai-disclaimer">{_escape(R.EXPLANATION_DISCLAIMER)}</div>',
        unsafe_allow_html=True,
    )

    impacts = R.explanation_impacts(explanation)
    if impacts:
        frame = pd.DataFrame(
            {
                "feature": [row["feature"] for row in impacts],
                "impact": [float(row["impact"] or 0.0) for row in impacts],
            }
        )
        st.subheader("Global feature importance")
        st.bar_chart(frame.set_index("feature"))
        st.dataframe(impacts)
    st.caption(
        "Single-record (local) explanations are produced on the Predict page, "
        "one per record, alongside each prediction."
    )


# -- page: Registry & Deploy ---------------------------------------------------


def _page_registry(client: APIClient) -> None:
    st.title("Registry & Deploy")

    st.markdown(
        f'<div class="maintai-disclaimer">{_escape(R.DEMO_DEPLOY_DISCLAIMER)}</div>',
        unsafe_allow_html=True,
    )

    _render_register(client)

    st.subheader("Registered models")
    models = _models(client)
    if st.button("Refresh list", key="refresh_models"):
        _reload(client, "models_cache", client.list_models)
        models = _models(client)
    if models:
        rows = R.registered_model_rows(models)
        st.dataframe(rows)
        _render_deploy(client, models)
    else:
        st.info("No registered models yet.")


def _recommended_run(detail: dict[str, Any]) -> dict[str, Any] | None:
    """Find the recommended experiment's ModelRun row (register needs its `id`)."""
    recommended_name = detail.get("recommended_model")
    for run in detail.get("model_runs") or []:
        if isinstance(run, dict) and run.get("model_name") == recommended_name:
            return run
    return None


def _render_register(client: APIClient) -> None:
    experiment_id = st.session_state.get("current_experiment_id")
    detail = st.session_state.get("experiment_detail")
    if detail and detail.get("id") != experiment_id:
        detail = None
    if not detail and experiment_id:
        detail = _try(lambda: client.get_experiment(experiment_id), "Load experiment")
        if isinstance(detail, dict):
            st.session_state["experiment_detail"] = detail
    if not isinstance(detail, dict):
        st.info("Load a successful experiment on Experiments to register its recommended run.")
        return
    if detail.get("status") != "succeeded":
        st.warning("Experiment has not succeeded; nothing eligible to register.")
        return

    run = _recommended_run(detail)
    if not run:
        st.warning("No recommended model run to register.")
        return
    _kv("Recommended run", run.get("model_name"))
    _kv("Model run id", run.get("id"))
    name = st.text_input("Registry name (optional)", key="register_name")
    if st.button("Register as candidate", key="register_model"):
        result = _try(
            lambda: client.register_model(run.get("id"), name or None),
            "Register",
        )
        if isinstance(result, dict):
            st.session_state["current_model_id"] = result.get("id")
            st.session_state.pop("models_cache", None)
            _badge("OK", "ok")
            name = result.get("name")
            version = result.get("version")
            st.markdown(f"Registered **{_escape(name)}** v{_escape(version)}")


def _render_deploy(client: APIClient, models: list[dict[str, Any]]) -> None:
    ids = [m.get("id") for m in models if m.get("id")]
    if not ids:
        return
    current = st.session_state.get("current_model_id")
    index = ids.index(current) if current in ids else 0
    selected = st.selectbox(
        "Registered model",
        options=ids,
        index=index,
        key="sel_model",
        format_func=lambda i: _model_label(models, i),
    )
    if selected:
        st.session_state["current_model_id"] = selected
        if st.button("Deploy demo", key="deploy_demo"):
            result = _try(lambda: client.deploy_demo(selected), "Deploy demo")
            if isinstance(result, dict):
                st.session_state.pop("models_cache", None)
                _badge("OK", "ok")
                name = result.get("name")
                version = result.get("version")
                st.markdown(f"Demo-deployed **{_escape(name)}** v{_escape(version)}")
                st.caption(R.DEMO_DEPLOY_DISCLAIMER)


def _model_label(models: list[dict[str, Any]], model_id: str) -> str:
    for model in models:
        if model.get("id") == model_id:
            return f"{model.get('name')} v{model.get('version')} [{model.get('deployment_status')}]"
    return model_id[:8]


# -- page: Predict -------------------------------------------------------------


def _page_predict(client: APIClient) -> None:
    st.title("Predict")

    st.markdown(
        f'<div class="maintai-disclaimer">{_escape(R.PRODUCT_DISCLAIMER)}</div>',
        unsafe_allow_html=True,
    )

    models = _models(client)
    deployed = [
        m for m in models if isinstance(m, dict) and m.get("deployment_status") == "demo_deployed"
    ]
    if st.button("Refresh list", key="refresh_models_predict"):
        _reload(client, "models_cache", client.list_models)
        deployed = [
            m
            for m in _models(client)
            if isinstance(m, dict) and m.get("deployment_status") == "demo_deployed"
        ]
    if not deployed:
        st.info("No demo-deployed model. Register and demo-deploy one first.")
        return

    ids = [m.get("id") for m in deployed if m.get("id")]
    current = st.session_state.get("current_model_id")
    index = ids.index(current) if current in ids else 0
    selected = st.selectbox(
        "Demo-deployed model",
        options=ids,
        index=index,
        key="sel_deployed_model",
        format_func=lambda i: _model_label(deployed, i),
    )
    if selected:
        st.session_state["current_model_id"] = selected

    feature_names = _feature_names_for(deployed, selected)
    if feature_names:
        _section("Required fields", ", ".join(feature_names))
    else:
        _section(
            "Required fields",
            "Records must exactly match the model's training features; the API "
            "rejects missing/extra fields with a clear message.",
        )

    tab_single, tab_batch = st.tabs(["Single (JSON)", "Batch (CSV)"])
    with tab_single:
        _render_predict_single(client, selected)
    with tab_batch:
        _render_predict_batch(client, selected)


def _feature_names_for(models: list[dict[str, Any]], model_id: str) -> list[str] | None:
    model = next((m for m in models if m.get("id") == model_id), None)
    if not model:
        return None
    detail = st.session_state.get("experiment_detail")
    if isinstance(detail, dict):
        for run in detail.get("model_runs") or []:
            if isinstance(run, dict) and run.get("id") == model.get("model_run_id"):
                names = run.get("feature_names")
                if isinstance(names, list):
                    return [str(n) for n in names]
    return None


def _render_predict_single(client: APIClient, model_id: str) -> None:
    st.markdown("Paste a single JSON object of feature values.")
    default_json = st.session_state.get("predict_single_json", "{}")
    raw = st.text_area("Record (JSON)", value=default_json, key="predict_single_json", height=140)
    if st.button("Predict (single)", key="predict_single"):
        record = _parse_json_object(raw)
        if record is None:
            return
        result = _try(lambda: client.predict_single(model_id, record), "Predict")
        if isinstance(result, dict):
            _render_prediction_result(result)


def _render_predict_batch(client: APIClient, model_id: str) -> None:
    st.markdown("Upload a CSV whose header columns are the model's features.")
    batch_file = st.file_uploader("Batch CSV", type=["csv"], key="batch_file")
    if st.button("Predict (batch)", key="predict_batch", disabled=batch_file is None):
        if batch_file is None:
            return
        try:
            text = batch_file.getvalue().decode("utf-8")
            records = R.parse_csv_records(text)
        except (UnicodeDecodeError, ValueError) as exc:
            st.error(f"Invalid CSV input: {exc}")
            return
        result = _try(lambda: client.predict_batch(model_id, records), "Batch predict")
        if isinstance(result, dict):
            _render_prediction_result(result)


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc}")
        return None
    if not isinstance(parsed, dict):
        st.error("Input must be a single JSON object, e.g. {\"torque\": 1.2}.")
        return None
    return parsed


def _render_prediction_result(result: dict[str, Any]) -> None:
    _kv("Model", result.get("model_id"))
    _kv("Version", result.get("model_version"))
    _kv("Count", result.get("count"))
    records = result.get("records") or []
    st.subheader("Predictions")
    st.json(records)
    csv_text = R.records_to_csv(records)
    if csv_text:
        st.download_button(
            "Download results (CSV)",
            data=csv_text,
            file_name="predictions.csv",
            mime="text/csv",
            key="download_predictions",
        )


# -- page: Copilot -------------------------------------------------------------


def _page_copilot(client: APIClient) -> None:
    st.title("Copilot")

    st.subheader("Quick prompts")
    cols = st.columns(4)
    for index, (prompt_id, label, prompt) in enumerate(R.QUICK_PROMPTS):
        with cols[index % 4]:
            if st.button(label, key=f"qp_{prompt_id}"):
                st.session_state["copilot_input"] = prompt
                st.session_state["copilot_qp_id"] = prompt_id

    st.subheader("Context")
    _kv("Dataset", st.session_state.get("current_dataset_id") or "none")
    _kv("Experiment", st.session_state.get("current_experiment_id") or "none")
    _kv("Model", st.session_state.get("current_model_id") or "none")

    user_request = st.text_area(
        "Ask the copilot",
        value=st.session_state.get("copilot_input", ""),
        key="copilot_input",
        height=90,
    )
    if st.button("Send", key="copilot_send"):
        if not user_request.strip():
            st.warning("Enter a request first.")
            return
        result = _try(
            lambda: client.copilot_chat(
                user_request.strip(),
                dataset_id=st.session_state.get("current_dataset_id"),
                experiment_id=st.session_state.get("current_experiment_id"),
                model_id=st.session_state.get("current_model_id"),
                conversation_id=st.session_state.get("copilot_conversation_id"),
            ),
            "Copilot",
        )
        if isinstance(result, dict):
            st.session_state["copilot_conversation_id"] = result.get("conversation_id")
            st.session_state["copilot_history"].append(
                {
                    "prompt_id": st.session_state.get("copilot_qp_id"),
                    "prompt": user_request.strip(),
                    "response": result,
                }
            )

    history = st.session_state.get("copilot_history") or []
    if history:
        st.subheader("Conversation")
        for index, item in enumerate(reversed(history)):
            with st.expander(
                f"#{len(history) - index} — {item.get('prompt', '')[:60]}",
                expanded=index == 0,
            ):
                st.markdown(f"**Prompt** (`{_escape(item.get('prompt_id') or 'free-text')}`)")
                st.markdown(_escape(item.get("prompt")))
                _render_copilot_response(item.get("response") or {})


def _render_copilot_response(response: dict[str, Any]) -> None:
    _kv("Intent", response.get("intent"))
    _kv("Conversation id", response.get("conversation_id"))
    st.markdown("**Answer**")
    st.markdown(_escape(response.get("answer") or ""))
    evidence = response.get("evidence") or []
    if evidence:
        st.markdown("**Evidence**")
        st.json(evidence)
    proposed = response.get("proposed_action")
    if isinstance(proposed, dict) and proposed:
        st.markdown("**Proposed action**")
        st.json(proposed)
    st.caption(
        "Copilot narrates deterministic tool results; it cannot deploy, promote, "
        "or create maintenance actions without approval."
    )


# -- page: P1 Monitor & Act ----------------------------------------------------


def _page_p1_monitor(client: APIClient) -> None:
    st.title("P1 Monitor & Act")
    _section(
        "P1 operations console",
        "Deterministic monitoring, cost-aware comparison, human-gated promotion "
        "and approvals, and technician feedback + mock CMMS. Every action runs "
        "through the FastAPI backend over HTTP; nothing auto-deploys or retrains.",
    )
    st.markdown(
        f'<div class="maintai-disclaimer">{_escape(R.PRODUCT_DISCLAIMER)}</div>',
        unsafe_allow_html=True,
    )

    tab_monitoring, tab_cost, tab_promotion, tab_feedback = st.tabs(
        ["Monitoring", "Cost", "Promotion & approvals", "Feedback & Mock CMMS"]
    )
    with tab_monitoring:
        _render_monitoring_tab(client)
    with tab_cost:
        _render_cost_tab(client)
    with tab_promotion:
        _render_promotion_tab(client)
    with tab_feedback:
        _render_feedback_tab(client)


def _gate_inputs(prefix: str) -> tuple[str, str]:
    """Render the local-demo human gate inputs (token is never displayed/logged)."""
    token = st.text_input(
        "Approval bearer token",
        type="password",
        key=f"{prefix}_token",
        help="Local-demo APPROVAL_API_TOKEN; never logged or cached globally.",
    )
    actor = st.text_input("Human actor id", key=f"{prefix}_actor")
    return token, actor


def _severity_kind(severity: Any) -> str:
    value = str(severity).upper()
    if value == "HIGH":
        return "err"
    if value == "MEDIUM":
        return "warn"
    return "ok"


# -- tab: Monitoring -----------------------------------------------------------


def _render_monitoring_tab(client: APIClient) -> None:
    st.markdown(
        f'<div class="maintai-disclaimer">{_escape(R.MONITORING_DISCLAIMER)}</div>',
        unsafe_allow_html=True,
    )

    models = _models(client)
    if st.button("Refresh models", key="refresh_models_monitor"):
        _reload(client, "models_cache", client.list_models)
        models = _models(client)
    if not models:
        st.info("No registered models. Register a candidate on Registry & Deploy first.")
        return
    monitorable = [
        m
        for m in models
        if isinstance(m, dict) and m.get("deployment_status") in ("candidate", "demo_deployed")
    ]
    if not monitorable:
        st.info("No candidate / demo-deployed model available to monitor.")
        return

    ids = [m.get("id") for m in monitorable if m.get("id")]
    selected = st.selectbox(
        "Model to monitor",
        options=ids,
        key="monitor_model",
        format_func=lambda i: _model_label(monitorable, i),
    )
    replay_kind = st.selectbox(
        "Replay kind (synthetic production window)",
        options=list(R.REPLAY_KINDS),
        key="monitor_replay_kind",
    )
    if st.button("Run monitoring replay", key="run_monitoring"):
        result = _try(
            lambda: client.create_monitoring_run(selected, replay_kind=replay_kind),
            "Monitoring run",
        )
        if isinstance(result, dict):
            st.session_state["monitoring_result"] = result

    result = st.session_state.get("monitoring_result")
    if isinstance(result, dict):
        _render_monitoring_result(result)

    st.subheader("Recent monitoring runs")
    runs = _load(client, "monitoring_runs_cache", client.list_monitoring_runs)
    if st.button("Refresh runs", key="refresh_monitoring_runs"):
        _reload(client, "monitoring_runs_cache", client.list_monitoring_runs)
        runs = _load(client, "monitoring_runs_cache", client.list_monitoring_runs)
    if not runs:
        st.info("No monitoring runs recorded yet.")
        return
    run_ids = [r.get("id") for r in runs if r.get("id")]
    chosen = st.selectbox(
        "Run",
        options=run_ids,
        key="monitor_run",
        format_func=lambda i: i[:8],
    )
    if chosen and st.button("Load run", key="load_monitoring_run"):
        detail = _try(lambda: client.get_monitoring_run(chosen), "Load monitoring run")
        if isinstance(detail, dict):
            st.session_state["monitoring_result"] = detail
            st.rerun()


def _render_monitoring_result(result: dict[str, Any]) -> None:
    status = result.get("status")
    kind = "ok" if status == "succeeded" else ("err" if status == "failed" else "warn")
    _badge(str(status).upper(), kind)
    _kv("Run id", result.get("id"))
    _kv("Replay kind", result.get("replay_kind"))
    _kv("Baseline dataset", result.get("dataset_id"))
    _kv("Created", result.get("created_at"))
    _kv("Completed", result.get("completed_at"))
    if result.get("error_message"):
        st.error(f"Run failed: {result['error_message']}")
        return

    drift = result.get("drift") if isinstance(result.get("drift"), dict) else {}
    anomaly = result.get("anomaly") if isinstance(result.get("anomaly"), dict) else {}
    recommendation = (
        result.get("recommendation") if isinstance(result.get("recommendation"), dict) else {}
    )

    st.subheader("Drift")
    if drift:
        overall = str(drift.get("overall_severity") or "LOW").upper()
        _badge(overall, _severity_kind(drift.get("overall_severity")))
        rows = R.drift_feature_rows(drift)
        if rows:
            st.dataframe(rows)
        else:
            st.info("No per-feature drift recorded.")
        notes = drift.get("notes") or []
        if notes:
            st.markdown("**Notes**")
            for note in notes:
                st.markdown(f"- {_escape(note)}")
    else:
        st.info("No drift report for this run.")

    st.subheader("Anomaly detection")
    if anomaly:
        col1, col2 = st.columns(2)
        col1.metric("Flagged rows", anomaly.get("flagged_count", 0))
        col2.metric("Anomaly rate", R.fmt_pct(anomaly.get("anomaly_rate")))
        top = R.anomaly_top_rows(anomaly)
        if top:
            st.dataframe(top)
        if anomaly.get("rows_omitted"):
            st.caption(f"{anomaly.get('rows_omitted')} lower-scoring rows omitted.")
    else:
        st.info("No anomaly report for this run.")

    st.subheader("Retraining recommendation")
    if recommendation:
        recommended = bool(recommendation.get("recommended"))
        _badge("RECOMMENDED" if recommended else "NOT RECOMMENDED", "warn" if recommended else "ok")
        if recommendation.get("triggers"):
            st.markdown(
                "**Triggers**: " + ", ".join(str(t) for t in recommendation["triggers"])
            )
        evidence = R.recommendation_evidence_rows(recommendation)
        if evidence:
            st.dataframe(evidence)
        if recommendation.get("proposed_action"):
            _kv("Proposed action", recommendation.get("proposed_action"))
        st.caption("auto_deploy is permanently False: a human must approve any change.")
    else:
        st.info("No recommendation for this run.")

    st.markdown(
        f'<div class="maintai-disclaimer">{_escape(R.MONITORING_DISCLAIMER)}</div>',
        unsafe_allow_html=True,
    )


# -- tab: Cost -----------------------------------------------------------------


def _render_cost_tab(client: APIClient) -> None:
    st.markdown(
        f'<div class="maintai-disclaimer">{_escape(R.COST_ASSUMPTIONS_DISCLAIMER)}</div>',
        unsafe_allow_html=True,
    )

    experiments = _experiments(client)
    if st.button("Refresh list", key="refresh_experiments_cost"):
        _reload(client, "experiments_cache", client.list_experiments)
        experiments = _experiments(client)
    succeeded = [
        e for e in experiments if isinstance(e, dict) and e.get("status") == "succeeded"
    ]
    if not succeeded:
        st.info("No succeeded experiment available for cost comparison.")
        return

    ids = [e.get("id") for e in succeeded if e.get("id")]
    selected = st.selectbox(
        "Succeeded experiment",
        options=ids,
        key="cost_experiment",
        format_func=lambda i: i[:8],
    )
    col1, col2, col3 = st.columns(3)
    fn_cost = col1.number_input(
        "False-negative cost", min_value=0.0, value=10.0, step=1.0, key="cost_fn"
    )
    fp_cost = col2.number_input(
        "False-positive cost", min_value=0.0, value=1.0, step=1.0, key="cost_fp"
    )
    min_recall = col3.number_input(
        "Minimum recall", min_value=0.0, max_value=1.0, value=0.8, step=0.05, key="cost_recall"
    )
    if st.button("Compare cost", key="run_cost"):
        result = _try(
            lambda: client.get_cost_comparison(selected, fn_cost, fp_cost, min_recall),
            "Cost comparison",
        )
        if isinstance(result, dict):
            st.session_state["cost_result"] = result

    result = st.session_state.get("cost_result")
    if isinstance(result, dict):
        _render_cost_result(result)


def _render_cost_result(result: dict[str, Any]) -> None:
    _kv("Experiment", result.get("experiment_id"))
    _badge(str(result.get("metric_best") or "--"), "info")
    st.markdown("metric-best")
    _badge(str(result.get("cost_best") or "--"), "ok")
    st.markdown("cost-best")
    rows = R.cost_comparison_rows(result)
    if rows:
        st.dataframe(rows)
    else:
        st.info("No cost rows returned.")
    notes = result.get("notes") or []
    if notes:
        st.markdown("**Notes**")
        for note in notes:
            st.markdown(f"- {_escape(note)}")
    assumptions = result.get("assumptions")
    if isinstance(assumptions, dict):
        _kv("FN cost", assumptions.get("fn_cost"))
        _kv("FP cost", assumptions.get("fp_cost"))
        _kv("Currency", assumptions.get("currency_label"))
    _kv("Minimum recall", R.fmt_num(result.get("minimum_recall")))
    st.markdown(
        f'<div class="maintai-disclaimer">'
        f"{_escape(result.get('disclaimer') or R.COST_ASSUMPTIONS_DISCLAIMER)}</div>",
        unsafe_allow_html=True,
    )


# -- tab: Promotion & approvals ------------------------------------------------


def _render_promotion_tab(client: APIClient) -> None:
    st.markdown(
        f'<div class="maintai-disclaimer">{_escape(R.PROMOTION_DISCLAIMER)}</div>',
        unsafe_allow_html=True,
    )

    st.subheader("1. Request promotion")
    models = _models(client)
    if st.button("Refresh models", key="refresh_models_promo"):
        _reload(client, "models_cache", client.list_models)
        models = _models(client)
    if not models:
        st.info("No registered models. Register a candidate on Registry & Deploy first.")
        return
    ids = [m.get("id") for m in models if m.get("id")]
    if not ids:
        return
    selected = st.selectbox(
        "Registered model",
        options=ids,
        key="promo_model",
        format_func=lambda i: _model_label(models, i),
    )
    if st.button("Request promotion (challenger)", key="request_promotion"):
        result = _try(lambda: client.request_promotion(selected), "Promotion request")
        if isinstance(result, dict):
            st.session_state["promotion_request"] = result
            st.session_state.pop("approvals_cache", None)
    req = st.session_state.get("promotion_request")
    if isinstance(req, dict):
        _kv("Approval id", req.get("approval_id"))
        _kv("Approval status", req.get("approval_status"))
        _kv("Lifecycle status", req.get("lifecycle_status"))

    st.subheader("2. Human decision gate")
    token, actor = _gate_inputs("promo")

    st.subheader("3. Approvals")
    approvals = _load(client, "approvals_cache", client.list_approvals)
    if st.button("Refresh approvals", key="refresh_approvals"):
        _reload(client, "approvals_cache", client.list_approvals)
        approvals = _load(client, "approvals_cache", client.list_approvals)
    if not approvals:
        st.info("No approval requests yet.")
    else:
        st.dataframe(R.approval_rows(approvals))
        pending = [a for a in approvals if isinstance(a, dict) and a.get("status") == "pending"]
        pending_ids = [a.get("id") for a in pending if a.get("id")]
        if pending_ids:
            decision_id = st.selectbox(
                "Pending approval",
                options=pending_ids,
                key="promo_approval",
                format_func=lambda i: i[:8],
            )
            version = next(
                (a.get("version", 1) for a in pending if a.get("id") == decision_id), 1
            )
            col1, col2 = st.columns(2)
            if col1.button("Approve", key="approve_btn", disabled=not (token and actor)):
                result = _try(
                    lambda: client.approve_approval(
                        decision_id, version, token=token, actor_id=actor
                    ),
                    "Approve",
                )
                if isinstance(result, dict):
                    st.session_state.pop("approvals_cache", None)
                    _badge("OK", "ok")
                    st.rerun()
            reject_reason = col2.text_input("Reject reason", key="reject_reason")
            if st.button(
                "Reject",
                key="reject_btn",
                disabled=not (token and actor and reject_reason.strip()),
            ):
                result = _try(
                    lambda: client.reject_approval(
                        decision_id, version, reject_reason.strip(), token=token, actor_id=actor
                    ),
                    "Reject",
                )
                if isinstance(result, dict):
                    st.session_state.pop("approvals_cache", None)
                    _badge("OK", "ok")
                    st.rerun()
        else:
            st.info("No pending approvals to decide.")

    st.subheader("4. Execute promotion (separate, explicit)")
    st.caption("Executes only an already approved/modified promotion; it never auto-deploys.")
    exec_approval_id = st.text_input(
        "Approval id to execute",
        key="promo_exec_approval",
        value=(req or {}).get("approval_id") if isinstance(req, dict) else "",
    )
    if st.button(
        "Execute promotion",
        key="execute_promotion",
        disabled=not (token and actor and exec_approval_id.strip()),
    ):
        result = _try(
            lambda: client.promote_model(
                selected, exec_approval_id.strip(), token=token, actor_id=actor
            ),
            "Promotion",
        )
        if isinstance(result, dict):
            st.session_state.pop("models_cache", None)
            _badge("OK", "ok")
            st.markdown(
                f"Promoted **{_escape(result.get('name'))}** v{_escape(result.get('version'))} "
                f"(executed={result.get('executed')})"
            )
            _kv("Lifecycle status", result.get("lifecycle_status"))


# -- tab: Feedback & Mock CMMS -------------------------------------------------


def _render_feedback_tab(client: APIClient) -> None:
    st.subheader("Technician feedback")
    _section(
        "Submit or review technician feedback",
        "Requires a technician id (X-Human-Actor-ID); no bearer token. Feedback is "
        "append-only.",
    )
    col1, col2, col3 = st.columns(3)
    prediction_id = col1.text_input("Prediction event id", key="feedback_prediction_id")
    actor = col2.text_input("Technician id", key="feedback_actor")
    outcome = col3.selectbox(
        "Outcome",
        options=R.FEEDBACK_OUTCOMES,
        key="feedback_outcome",
    )
    comment = st.text_input("Comment (optional)", key="feedback_comment")
    if st.button(
        "Submit feedback",
        key="submit_feedback",
        disabled=not (prediction_id.strip() and actor.strip()),
    ):
        result = _try(
            lambda: client.submit_feedback(
                prediction_id.strip(),
                outcome,
                actor_id=actor.strip(),
                comment=comment.strip() or None,
            ),
            "Feedback",
        )
        if isinstance(result, dict):
            _badge("OK", "ok")
            st.markdown(f"Feedback **{_escape(result.get('id'))}** recorded.")
    if st.button("List feedback", key="list_feedback", disabled=not prediction_id.strip()):
        feedback = _try(lambda: client.list_feedback(prediction_id.strip()), "List feedback")
        if isinstance(feedback, list):
            if feedback:
                st.dataframe(R.feedback_rows(feedback))
            else:
                st.info("No feedback for this prediction id yet.")

    st.subheader("Mock CMMS")
    st.markdown(
        f'<div class="maintai-disclaimer">{_escape(R.CMMS_MOCK_DISCLAIMER)}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Propose a CMMS work order**")
    col1, col2 = st.columns(2)
    asset_id = col1.text_input("Asset id", key="cmms_asset_id")
    priority = col2.selectbox(
        "Priority", options=("low", "medium", "high", "critical"), key="cmms_priority"
    )
    event_id = st.text_input("Prediction event id (entity)", key="cmms_event_id")
    action = st.text_input("Recommended action", key="cmms_action")
    risk = st.number_input(
        "Risk score (optional)", min_value=0.0, value=0.0, step=0.1, key="cmms_risk"
    )
    if st.button(
        "Propose CMMS approval",
        key="cmms_propose",
        disabled=not (asset_id.strip() and event_id.strip() and action.strip()),
    ):
        payload: dict[str, Any] = {
            "asset_id": asset_id.strip(),
            "priority": priority,
            "recommended_action": action.strip(),
            "prediction_event_id": event_id.strip(),
        }
        if risk > 0:
            payload["risk_score"] = risk
        result = _try(
            lambda: client.propose_approval(
                action_type="cmms_work_order",
                entity_type="prediction_event",
                entity_id=event_id.strip(),
                requested_by_type="user",
                proposed_payload=payload,
            ),
            "CMMS proposal",
        )
        if isinstance(result, dict):
            st.session_state["cmms_approval_id"] = result.get("id")
            st.session_state.pop("approvals_cache", None)
            _badge("OK", "ok")
            st.markdown(
                f"Proposed approval **{_escape(result.get('id'))}** "
                f"(status {_escape(result.get('status'))})."
            )

    st.markdown("**Execute an approved CMMS approval into a mock work order**")
    token, _ = _gate_inputs("cmms")
    approval_id = st.text_input(
        "Approval id",
        key="cmms_exec_approval",
        value=st.session_state.get("cmms_approval_id") or "",
    )
    if st.button(
        "Execute mock CMMS draft",
        key="cmms_exec",
        disabled=not (token and approval_id.strip()),
    ):
        result = _try(
            lambda: client.draft_work_order(approval_id.strip(), token=token), "CMMS draft"
        )
        if isinstance(result, dict):
            st.session_state.pop("work_orders_cache", None)
            _badge("OK", "ok")
            st.markdown(
                f"Work order **{_escape(result.get('id'))}** "
                f"({_escape(result.get('status'))})"
            )
            _kv("Asset", result.get("asset_id"))
            _kv("Priority", result.get("priority"))
            disclaimer = result.get("disclaimer") or R.CMMS_MOCK_DISCLAIMER
            st.markdown(
                f'<div class="maintai-disclaimer">{_escape(disclaimer)}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("**Mock work-order drafts**")
    work_orders = _load(client, "work_orders_cache", client.list_work_orders)
    if st.button("Refresh drafts", key="refresh_work_orders"):
        _reload(client, "work_orders_cache", client.list_work_orders)
        work_orders = _load(client, "work_orders_cache", client.list_work_orders)
    if work_orders:
        st.dataframe(R.work_order_rows(work_orders))
    else:
        st.info("No mock work-order drafts yet.")


# -- dispatch ------------------------------------------------------------------


def _render_page(key: str, client: APIClient) -> None:
    pages: dict[str, Callable[[APIClient], None]] = {
        "home": _page_home,
        "datasets": _page_datasets,
        "task": _page_task,
        "experiments": _page_experiments,
        "explainability": _page_explainability,
        "registry": _page_registry,
        "predict": _page_predict,
        "copilot": _page_copilot,
        "p1_monitor": _page_p1_monitor,
    }
    pages[key](client)


def main() -> None:
    st.set_page_config(page_title="MaintAI Studio", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    _init_state()
    client = _get_client()
    page_key = _render_sidebar()
    _render_page(page_key, client)


if __name__ == "__main__":
    main()

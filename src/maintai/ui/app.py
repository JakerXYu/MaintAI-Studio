"""Streamlit health dashboard (Phase A).

The UI talks to the FastAPI API **only** over HTTP; it never imports ML
internals or the database. Set ``MAINTAI_API_URL`` to point at the API
(default ``http://localhost:8000``).
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("MAINTAI_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="MaintAI Studio", layout="wide")
st.title("MaintAI Studio — Health")

st.caption(f"API: {API_URL}")

statuses: dict[str, tuple[bool, str]] = {}
for name, path in [
    ("liveness (/health/live)", "/health/live"),
    ("readiness (/health/ready)", "/health/ready"),
    ("service (/health)", "/health"),
]:
    try:
        resp = httpx.get(f"{API_URL}{path}", timeout=5.0)
        ok = resp.status_code == 200
        statuses[name] = (ok, f"HTTP {resp.status_code}")
    except httpx.HTTPError as exc:  # noqa: BLE001 - show connectivity failures
        statuses[name] = (False, f"unreachable: {exc}")

for name, (ok, detail) in statuses.items():
    icon = "✅" if ok else "❌"
    st.markdown(f"{icon} **{name}** — {detail}")

if all(ok for ok, _ in statuses.values()):
    st.success("API is healthy.")
else:
    st.error("API is not fully healthy. Check `docker compose ps`.")

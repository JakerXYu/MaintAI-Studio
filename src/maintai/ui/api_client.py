"""Typed-ish HTTP client for the MaintAI FastAPI backend.

The Streamlit UI talks to the API **only** over HTTP using ``httpx``. This
module never imports ML internals, the database, or application services —
its only dependency is ``httpx`` (plus ``urllib`` for safe path quoting).

Every method maps the backend's stable error shape (``{"detail": ...}`` and the
``X-Request-ID`` header) onto a single :class:`UIAPIError` so the UI can render
connectivity / validation failures gracefully without leaking internal details.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

import httpx

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 30.0

_API_PREFIX = "/api/v1"


class UIAPIError(Exception):
    """Stable error raised for any API failure (connectivity, timeout, or 4xx/5xx).

    ``detail`` carries the backend's readable message when present; ``request_id``
    echoes the backend ``X-Request-ID`` so an operator can correlate a failure.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        detail: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail
        self.request_id = request_id

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


def _path_segment(value: Any) -> str:
    """Quote a URL path segment so ids can never inject slashes or traversal."""
    if value is None:
        return ""
    return quote(str(value), safe="")


def _auth_headers(token: str | None, actor_id: str | None = None) -> dict[str, str]:
    """Build local-demo auth headers without caching, rendering, or logging the token.

    The bearer token is only ever placed into the outgoing ``Authorization``
    header for the single request that needs it; it is never stored on the
    client, echoed back, or persisted anywhere.
    """
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if actor_id:
        headers["X-Human-Actor-ID"] = actor_id
    return headers


class APIClient:
    """Synchronous HTTP client bound to a single base URL and timeout."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("MAINTAI_API_URL", DEFAULT_BASE_URL)).rstrip(
            "/"
        )
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> APIClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals -----------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise UIAPIError(
                f"API timed out after {self.timeout:g}s",
                status_code=None,
            ) from exc
        except httpx.HTTPError as exc:
            raise UIAPIError(f"API unreachable: {exc}") from exc
        return self._decode(response)

    def _decode(self, response: httpx.Response) -> Any:
        payload: Any = None
        try:
            payload = response.json()
        except ValueError:
            payload = None

        request_id = response.headers.get("x-request-id")

        if response.is_error:
            detail: str | None = None
            if isinstance(payload, dict):
                raw_detail = payload.get("detail")
                if isinstance(raw_detail, str):
                    detail = raw_detail
            message = f"API error {response.status_code}"
            if detail:
                message = f"{message}: {detail}"
            raise UIAPIError(
                message,
                status_code=response.status_code,
                detail=detail,
                request_id=request_id,
            )
        return payload

    # -- health --------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def health_live(self) -> dict[str, Any]:
        return self._request("GET", "/health/live")

    def health_ready(self) -> dict[str, Any]:
        return self._request("GET", "/health/ready")

    # -- datasets ------------------------------------------------------------

    def upload_dataset(self, filename: str, content: bytes) -> dict[str, Any]:
        return self._request(
            "POST",
            f"{_API_PREFIX}/datasets/upload",
            files={"file": (filename, content, "application/octet-stream")},
        )

    def list_datasets(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return self._request(
            "GET", f"{_API_PREFIX}/datasets", params={"limit": limit, "offset": offset}
        )

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        return self._request("GET", f"{_API_PREFIX}/datasets/{_path_segment(dataset_id)}")

    def profile_dataset(self, dataset_id: str) -> dict[str, Any]:
        return self._request("POST", f"{_API_PREFIX}/datasets/{_path_segment(dataset_id)}/profile")

    def get_quality(self, dataset_id: str) -> dict[str, Any]:
        return self._request("GET", f"{_API_PREFIX}/datasets/{_path_segment(dataset_id)}/quality")

    def recommend_task(
        self,
        dataset_id: str,
        target_column: str,
        asset_id_column: str | None = None,
        timestamp_column: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"target_column": target_column}
        if asset_id_column:
            body["asset_id_column"] = asset_id_column
        if timestamp_column:
            body["timestamp_column"] = timestamp_column
        return self._request(
            "POST",
            f"{_API_PREFIX}/datasets/{_path_segment(dataset_id)}/task-recommendation",
            json=body,
        )

    # -- experiments ---------------------------------------------------------

    def create_experiment(
        self,
        dataset_id: str,
        model_names: list[str] | None = None,
        minimum_recall: float | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"dataset_id": dataset_id}
        if model_names:
            body["model_names"] = model_names
        if minimum_recall is not None:
            body["minimum_recall"] = minimum_recall
        return self._request("POST", f"{_API_PREFIX}/experiments", json=body)

    def list_experiments(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return self._request(
            "GET", f"{_API_PREFIX}/experiments", params={"limit": limit, "offset": offset}
        )

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        return self._request("GET", f"{_API_PREFIX}/experiments/{_path_segment(experiment_id)}")

    def get_comparison(self, experiment_id: str) -> dict[str, Any]:
        return self._request(
            "GET", f"{_API_PREFIX}/experiments/{_path_segment(experiment_id)}/comparison"
        )

    # -- models / registry ---------------------------------------------------

    def list_models(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return self._request(
            "GET", f"{_API_PREFIX}/models", params={"limit": limit, "offset": offset}
        )

    def get_model(self, registered_id: str) -> dict[str, Any]:
        return self._request("GET", f"{_API_PREFIX}/models/{_path_segment(registered_id)}")

    def register_model(self, model_run_id: str, name: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if name:
            body["name"] = name
        return self._request(
            "POST",
            f"{_API_PREFIX}/models/{_path_segment(model_run_id)}/register",
            json=body,
        )

    def deploy_demo(self, registered_id: str) -> dict[str, Any]:
        return self._request(
            "POST", f"{_API_PREFIX}/models/{_path_segment(registered_id)}/deploy-demo"
        )

    # -- predictions ---------------------------------------------------------

    def predict_single(self, model_id: str, record: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "POST", f"{_API_PREFIX}/predict", json={"model_id": model_id, "records": [record]}
        )

    def predict_batch(self, model_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request(
            "POST", f"{_API_PREFIX}/predict/batch", json={"model_id": model_id, "records": records}
        )

    # -- copilot -------------------------------------------------------------

    def copilot_chat(
        self,
        user_request: str,
        *,
        dataset_id: str | None = None,
        experiment_id: str | None = None,
        model_id: str | None = None,
        record: dict[str, Any] | None = None,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"user_request": user_request}
        if dataset_id:
            body["dataset_id"] = dataset_id
        if experiment_id:
            body["experiment_id"] = experiment_id
        if model_id:
            body["model_id"] = model_id
        if record is not None:
            body["record"] = record
        if conversation_id:
            body["conversation_id"] = conversation_id
        return self._request("POST", f"{_API_PREFIX}/copilot/chat", json=body)

    # -- monitoring (P1) ----------------------------------------------------

    def create_monitoring_run(
        self,
        model_id: str,
        *,
        replay_kind: str | None = None,
        production_dataset_id: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model_id": model_id}
        if replay_kind is not None:
            body["replay_kind"] = replay_kind
        if production_dataset_id is not None:
            body["production_dataset_id"] = production_dataset_id
        return self._request("POST", f"{_API_PREFIX}/monitoring/runs", json=body)

    def list_monitoring_runs(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return self._request(
            "GET", f"{_API_PREFIX}/monitoring/runs", params={"limit": limit, "offset": offset}
        )

    def get_monitoring_run(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"{_API_PREFIX}/monitoring/runs/{_path_segment(run_id)}")

    # -- cost-aware comparison (P1) -----------------------------------------

    def get_cost_comparison(
        self,
        experiment_id: str,
        fn_cost: float,
        fp_cost: float,
        minimum_recall: float,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"{_API_PREFIX}/experiments/{_path_segment(experiment_id)}/cost-comparison",
            json={"fn_cost": fn_cost, "fp_cost": fp_cost, "minimum_recall": minimum_recall},
        )

    # -- approvals (P1) -----------------------------------------------------

    def propose_approval(
        self,
        *,
        action_type: str,
        entity_type: str,
        entity_id: str,
        requested_by_type: str,
        requested_by_id: str | None = None,
        proposed_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "action_type": action_type,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "requested_by_type": requested_by_type,
        }
        if requested_by_id:
            body["requested_by_id"] = requested_by_id
        if proposed_payload is not None:
            body["proposed_payload"] = proposed_payload
        return self._request("POST", f"{_API_PREFIX}/approvals", json=body)

    def list_approvals(
        self,
        *,
        status: str | None = None,
        action_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if action_type:
            params["action_type"] = action_type
        return self._request("GET", f"{_API_PREFIX}/approvals", params=params)

    def get_approval(self, approval_id: str, *, token: str | None = None) -> dict[str, Any]:
        return self._request(
            "GET",
            f"{_API_PREFIX}/approvals/{_path_segment(approval_id)}",
            headers=_auth_headers(token),
        )

    def approve_approval(
        self,
        approval_id: str,
        expected_version: int,
        *,
        token: str | None = None,
        actor_id: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"{_API_PREFIX}/approvals/{_path_segment(approval_id)}/approve",
            json={"expected_version": expected_version},
            headers=_auth_headers(token, actor_id),
        )

    def reject_approval(
        self,
        approval_id: str,
        expected_version: int,
        reason: str,
        *,
        token: str | None = None,
        actor_id: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"{_API_PREFIX}/approvals/{_path_segment(approval_id)}/reject",
            json={"expected_version": expected_version, "reason": reason},
            headers=_auth_headers(token, actor_id),
        )

    # -- champion/challenger promotion (P1) ---------------------------------

    def request_promotion(
        self,
        registered_id: str,
        *,
        requested_by_type: str = "agent",
        requested_by_id: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"requested_by_type": requested_by_type}
        if requested_by_id:
            body["requested_by_id"] = requested_by_id
        return self._request(
            "POST",
            f"{_API_PREFIX}/models/{_path_segment(registered_id)}/promotion-request",
            json=body,
        )

    def promote_model(
        self,
        registered_id: str,
        approval_id: str,
        *,
        token: str | None = None,
        actor_id: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"{_API_PREFIX}/models/{_path_segment(registered_id)}/promote",
            json={"approval_id": approval_id},
            headers=_auth_headers(token, actor_id),
        )

    # -- technician feedback (P1) -------------------------------------------

    def submit_feedback(
        self,
        prediction_id: str,
        outcome: str,
        *,
        actor_id: str,
        comment: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"outcome": outcome}
        if comment:
            body["comment"] = comment
        return self._request(
            "POST",
            f"{_API_PREFIX}/predictions/{_path_segment(prediction_id)}/feedback",
            json=body,
            headers={"X-Human-Actor-ID": actor_id},
        )

    def list_feedback(
        self, prediction_id: str, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        return self._request(
            "GET",
            f"{_API_PREFIX}/predictions/{_path_segment(prediction_id)}/feedback",
            params={"limit": limit, "offset": offset},
        )

    # -- mock CMMS (P1) -----------------------------------------------------

    def draft_work_order(self, approval_id: str, *, token: str | None = None) -> dict[str, Any]:
        return self._request(
            "POST",
            f"{_API_PREFIX}/cmms/work-orders/draft",
            json={"approval_id": approval_id},
            headers=_auth_headers(token),
        )

    def list_work_orders(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return self._request(
            "GET", f"{_API_PREFIX}/cmms/work-orders", params={"limit": limit, "offset": offset}
        )

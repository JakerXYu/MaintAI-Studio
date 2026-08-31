"""Smoke the P1 routers through the real FastAPI composition root."""


def test_p1_routes_are_registered_on_main_app(client):
    assert client.get("/api/v1/monitoring/runs").status_code == 200
    assert client.get("/api/v1/approvals").status_code == 200
    assert client.get("/api/v1/cmms/work-orders").status_code == 200
    assert client.get("/api/v1/predictions/missing/feedback").json() == []

    missing_cost = client.post(
        "/api/v1/experiments/missing/cost-comparison",
        json={"fn_cost": 10, "fp_cost": 1, "minimum_recall": 0.8},
    )
    assert missing_cost.status_code == 404

    missing_model = client.post(
        "/api/v1/models/missing/promotion-request",
        json={"requested_by_type": "user", "requested_by_id": "engineer-1"},
    )
    assert missing_model.status_code == 404

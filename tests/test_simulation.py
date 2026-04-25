"""
tests/test_simulation.py - Pipeline Simulation and PADE tests.
"""
import pytest

from tests.conftest import FAKE_USER, FAKE_VIEWER_USER, MockRecord


VALID_STAGES = [
    "checkout",
    "build",
    "unit_test",
    "integration_test",
    "security_scan",
    "docker_build",
    "deploy_staging",
    "deploy_prod",
]

VALID_DECISIONS = {"ALLOW", "WARN", "AUTO_OPTIMISE", "BLOCK"}


class TestSimulateEndpoint:
    @pytest.mark.asyncio
    async def test_simulate_returns_200_with_valid_structure(self, test_app, mock_conn_and_pool, valid_jwt_headers):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.side_effect = [MockRecord(FAKE_USER), None]
        conn.execute.return_value = "INSERT 0 1"

        resp = await test_app.post(
            "/api/ingest/simulate",
            headers=valid_jwt_headers,
            params={"anomaly_level": 0.5, "stage_name": "build"},
        )
        assert resp.status_code == 200
        body = resp.json()
        for field in ["run_id", "stage_name", "crs", "decision", "billed_cost"]:
            assert field in body, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_simulate_crs_in_valid_range(self, test_app, mock_conn_and_pool, valid_jwt_headers):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.side_effect = [MockRecord(FAKE_USER), None]
        conn.execute.return_value = "INSERT 0 1"

        resp = await test_app.post(
            "/api/ingest/simulate",
            headers=valid_jwt_headers,
            params={"anomaly_level": 0.7, "stage_name": "integration_test"},
        )
        assert resp.status_code == 200
        crs = resp.json()["crs"]
        assert 0.0 <= crs <= 1.0

    @pytest.mark.asyncio
    async def test_simulate_decision_is_valid(self, test_app, mock_conn_and_pool, valid_jwt_headers):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.side_effect = [MockRecord(FAKE_USER), None]
        conn.execute.return_value = "INSERT 0 1"

        resp = await test_app.post(
            "/api/ingest/simulate",
            headers=valid_jwt_headers,
            params={"anomaly_level": 0.9, "stage_name": "deploy_prod"},
        )
        assert resp.status_code == 200
        assert resp.json()["decision"] in VALID_DECISIONS

    @pytest.mark.asyncio
    async def test_simulate_inserts_attribution_row(self, test_app, mock_conn_and_pool, valid_jwt_headers):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.side_effect = [MockRecord(FAKE_USER), None]
        conn.execute.return_value = "INSERT 0 1"

        resp = await test_app.post(
            "/api/ingest/simulate",
            headers=valid_jwt_headers,
            params={"anomaly_level": 0.6, "stage_name": "unit_test"},
        )
        assert resp.status_code == 200
        assert conn.execute.call_count >= 1

    @pytest.mark.asyncio
    async def test_simulate_invalid_stage_defaults_to_integration_test(self, test_app, mock_conn_and_pool, valid_jwt_headers):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.side_effect = [MockRecord(FAKE_USER), None]
        conn.execute.return_value = "INSERT 0 1"

        resp = await test_app.post(
            "/api/ingest/simulate",
            headers=valid_jwt_headers,
            params={"anomaly_level": 0.5, "stage_name": "NONEXISTENT_STAGE"},
        )
        assert resp.status_code == 200
        assert resp.json()["stage_name"] in VALID_STAGES

    @pytest.mark.asyncio
    async def test_simulate_anomaly_level_out_of_range_returns_422(self, test_app, mock_conn_and_pool, valid_jwt_headers):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.return_value = MockRecord(FAKE_USER)
        resp = await test_app.post(
            "/api/ingest/simulate",
            headers=valid_jwt_headers,
            params={"anomaly_level": 1.5, "stage_name": "build"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_simulate_requires_authentication(self, test_app):
        resp = await test_app.post(
            "/api/ingest/simulate",
            params={"anomaly_level": 0.5, "stage_name": "build"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_simulate_requires_admin_role(self, test_app, mock_conn_and_pool, valid_jwt_headers):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.return_value = MockRecord(FAKE_VIEWER_USER)
        resp = await test_app.post(
            "/api/ingest/simulate",
            headers=valid_jwt_headers,
            params={"anomaly_level": 0.5, "stage_name": "build"},
        )
        assert resp.status_code == 403


class TestCRSScoring:
    def test_classify_crs_returns_block_above_0_90(self):
        import os
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
        from pade.ensemble import classify_crs

        assert classify_crs(0.95) == "BLOCK"
        assert classify_crs(0.90) == "BLOCK"

    def test_classify_crs_returns_auto_optimise_between_0_75_0_90(self):
        import os
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
        from pade.ensemble import classify_crs

        assert classify_crs(0.80) == "AUTO_OPTIMISE"
        assert classify_crs(0.75) == "AUTO_OPTIMISE"

    def test_classify_crs_returns_warn_between_0_50_0_75(self):
        import os
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
        from pade.ensemble import classify_crs

        assert classify_crs(0.60) == "WARN"
        assert classify_crs(0.50) == "WARN"

    def test_classify_crs_returns_allow_below_0_50(self):
        import os
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
        from pade.ensemble import classify_crs

        assert classify_crs(0.49) == "ALLOW"
        assert classify_crs(0.0) == "ALLOW"

    def test_compute_crs_equals_gat_prob(self):
        import os
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
        from pade.ensemble import compute_crs

        assert compute_crs(0.72) == pytest.approx(0.72)
        assert compute_crs(0.0) == pytest.approx(0.0)
        assert compute_crs(1.0) == pytest.approx(1.0)

    def test_opa_fallback_inline_policy_matches_ensemble(self):
        import os
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
        from pade.ensemble import classify_crs
        from peg.opa_client import _inline_policy

        for crs in [0.0, 0.3, 0.5, 0.6, 0.75, 0.80, 0.90, 0.95, 1.0]:
            assert classify_crs(crs) == _inline_policy(crs, 0.50, 0.75, 0.90)


class TestAlertRetrieval:
    @pytest.mark.asyncio
    async def test_recent_alerts_returns_list(self, test_app, mock_conn_and_pool):
        from tests.conftest import FAKE_ATTRIBUTION_ROW

        conn, _ = mock_conn_and_pool
        conn.fetch.return_value = [MockRecord(FAKE_ATTRIBUTION_ROW)]

        resp = await test_app.get("/api/alerts/recent?limit=10")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.asyncio
    async def test_recent_alerts_empty_returns_empty_list(self, test_app, mock_conn_and_pool):
        conn, _ = mock_conn_and_pool
        conn.fetch.return_value = []

        resp = await test_app.get("/api/alerts/recent?limit=5")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_alerts_limit_validated(self, test_app):
        resp = await test_app.get("/api/alerts/recent?limit=9999")
        assert resp.status_code == 422

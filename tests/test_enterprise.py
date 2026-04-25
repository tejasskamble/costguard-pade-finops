"""
tests/test_enterprise.py - Enterprise feature tests
===================================================
Tests the three enterprise feature groups:
  1. Budget guardrails (/api/budget)
  2. Async job queue (/api/jobs)
  3. Multi-cloud analytics (/api/providers)
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from tests.conftest import FAKE_USER, FAKE_VIEWER_USER, MockRecord


# -------------------------------------------------------------------
# FEATURE 1: Budget Guardrails
# -------------------------------------------------------------------

class TestBudgetGuardrails:

    @pytest.mark.asyncio
    async def test_configure_budget_creates_record(self, test_app, mock_conn_and_pool, valid_jwt_headers):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.return_value = MockRecord(FAKE_USER)
        conn.execute.return_value = "INSERT 0 1"

        resp = await test_app.post("/api/budget/configure", headers=valid_jwt_headers, json={
            "team_name":       "backend-team",
            "monthly_cap_usd": 50.00,
            "alert_at_pct":    80.0,
            "block_at_pct":    110.0,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["team"] == "backend-team"
        assert body["cap_usd"] == 50.00

    @pytest.mark.asyncio
    async def test_get_budget_status_returns_list(self, test_app, mock_conn_and_pool):
        conn, _ = mock_conn_and_pool
        conn.fetch.return_value = [MockRecord({
            "id":              1,
            "team_name":       "backend-team",
            "monthly_cap_usd": 50.0,
            "alert_at_pct":    80.0,
            "block_at_pct":    110.0,
            "webhook_url":     None,
        })]
        # Second call for monthly spend
        conn.fetchrow.return_value = MockRecord({"total": 12.5})

        resp = await test_app.get("/api/budget/status")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)

    @pytest.mark.asyncio
    async def test_budget_check_allows_under_cap(self, test_app, mock_conn_and_pool):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.side_effect = [
            MockRecord({
                "id":              1,
                "team_name":       "ml-team",
                "monthly_cap_usd": 100.0,
                "alert_at_pct":    80.0,
                "block_at_pct":    110.0,
                "webhook_url":     None,
            }),
            MockRecord({"total": 40.0}),   # 40% utilisation -> allowed
        ]

        resp = await test_app.get("/api/budget/check/ml-team")
        assert resp.status_code == 200
        body = resp.json()
        assert body["allowed"] is True

    @pytest.mark.asyncio
    async def test_budget_check_blocks_when_exceeded(self, test_app, mock_conn_and_pool):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.side_effect = [
            MockRecord({
                "id":              1,
                "team_name":       "ml-team",
                "monthly_cap_usd": 100.0,
                "alert_at_pct":    80.0,
                "block_at_pct":    110.0,
                "webhook_url":     None,
            }),
            MockRecord({"total": 115.0}),   # 115% -> blocked
        ]

        resp = await test_app.get("/api/budget/check/ml-team")
        assert resp.status_code == 402   # Payment Required
        body = resp.json()
        assert body["detail"]["allowed"] is False
        assert "BLOCKED" in body["detail"]["message"]

    @pytest.mark.asyncio
    async def test_budget_check_no_config_always_allows(self, test_app, mock_conn_and_pool):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.return_value = None   # no budget config -> always allow

        resp = await test_app.get("/api/budget/check/unconfigured-team")
        assert resp.status_code == 200
        assert resp.json()["allowed"] is True

    def test_budget_utilisation_calculation(self):
        """Unit test for status classification thresholds."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
        from api.budget import _classify_status
        assert _classify_status(50.0,  80.0, 110.0) == "OK"
        assert _classify_status(82.0,  80.0, 110.0) == "WARNING"
        assert _classify_status(101.0, 80.0, 110.0) == "CRITICAL"
        assert _classify_status(115.0, 80.0, 110.0) == "EXCEEDED"

    @pytest.mark.asyncio
    async def test_budget_mutations_require_authentication(self, test_app):
        resp = await test_app.post("/api/budget/configure", json={
            "team_name": "backend-team",
            "monthly_cap_usd": 50.00,
            "alert_at_pct": 80.0,
            "block_at_pct": 110.0,
        })
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_budget_mutations_require_admin_role(self, test_app, mock_conn_and_pool, valid_jwt_headers):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.return_value = MockRecord(FAKE_VIEWER_USER)
        resp = await test_app.post("/api/budget/configure", headers=valid_jwt_headers, json={
            "team_name": "backend-team",
            "monthly_cap_usd": 50.00,
            "alert_at_pct": 80.0,
            "block_at_pct": 110.0,
        })
        assert resp.status_code == 403


# -------------------------------------------------------------------
# FEATURE 2: Async Job Queue
# -------------------------------------------------------------------

class TestAsyncJobQueue:

    @pytest.mark.asyncio
    async def test_submit_job_returns_job_id(self, test_app, mock_conn_and_pool, valid_jwt_headers):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.return_value = MockRecord(FAKE_USER)
        resp = await test_app.post("/api/jobs/submit", headers=valid_jwt_headers, json={
            "job_type": "report",
            "priority": 3,
            "params":   {"days": 7},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "job_id" in body
        assert body["status"] == "QUEUED"
        assert body["job_type"] == "report"

    @pytest.mark.asyncio
    async def test_submit_unknown_job_type_returns_400(self, test_app, mock_conn_and_pool, valid_jwt_headers):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.return_value = MockRecord(FAKE_USER)
        resp = await test_app.post("/api/jobs/submit", headers=valid_jwt_headers, json={
            "job_type": "nonexistent_job",
            "priority": 3,
            "params":   {},
        })
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_get_job_status_after_submit(self, test_app, mock_conn_and_pool, valid_jwt_headers):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.return_value = MockRecord(FAKE_USER)
        # Submit
        submit_resp = await test_app.post("/api/jobs/submit", headers=valid_jwt_headers, json={
            "job_type": "export",
            "priority": 4,
            "params":   {"days": 7},
        })
        assert submit_resp.status_code == 200
        job_id = submit_resp.json()["job_id"]

        # Poll status
        status_resp = await test_app.get(f"/api/jobs/{job_id}", headers=valid_jwt_headers)
        assert status_resp.status_code == 200
        body = status_resp.json()
        assert body["job_id"] == job_id
        assert body["status"] in ("QUEUED", "RUNNING", "DONE")

    @pytest.mark.asyncio
    async def test_get_nonexistent_job_returns_404(self, test_app, mock_conn_and_pool, valid_jwt_headers):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.return_value = MockRecord(FAKE_USER)
        resp = await test_app.get("/api/jobs/00000000-0000-0000-0000-000000000000", headers=valid_jwt_headers)
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_jobs_returns_list(self, test_app, mock_conn_and_pool, valid_jwt_headers):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.return_value = MockRecord(FAKE_USER)
        resp = await test_app.get("/api/jobs/", headers=valid_jwt_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.asyncio
    async def test_cancel_queued_job(self, test_app, mock_conn_and_pool, valid_jwt_headers):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.return_value = MockRecord(FAKE_USER)
        # Submit a low-priority job
        submit = await test_app.post("/api/jobs/submit", headers=valid_jwt_headers, json={
            "job_type": "export",
            "priority": 4,
            "params":   {},
        })
        job_id = submit.json()["job_id"]

        # Cancel it immediately (before worker processes it)
        cancel = await test_app.delete(f"/api/jobs/{job_id}", headers=valid_jwt_headers)
        # May be QUEUED (cancellable) or already RUNNING (not cancellable)
        assert cancel.status_code in (200, 409)

    @pytest.mark.asyncio
    async def test_job_routes_require_authentication(self, test_app):
        resp = await test_app.post("/api/jobs/submit", json={
            "job_type": "report",
            "priority": 3,
            "params": {"days": 7},
        })
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_job_routes_require_admin_role(self, test_app, mock_conn_and_pool, valid_jwt_headers):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.return_value = MockRecord(FAKE_VIEWER_USER)
        resp = await test_app.post("/api/jobs/submit", headers=valid_jwt_headers, json={
            "job_type": "report",
            "priority": 3,
            "params": {"days": 7},
        })
        assert resp.status_code == 403

    def test_job_priority_ordering(self):
        """Critical jobs must have lower priority number than LOW jobs."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
        from api.jobs import Priority
        assert Priority.CRITICAL < Priority.HIGH
        assert Priority.HIGH     < Priority.NORMAL
        assert Priority.NORMAL   < Priority.LOW


# -------------------------------------------------------------------
# FEATURE 3: Multi-Cloud Provider Analytics
# -------------------------------------------------------------------

class TestMultiCloudProviders:

    @pytest.mark.asyncio
    async def test_compare_providers_returns_valid_structure(self, test_app, mock_conn_and_pool):
        conn, _ = mock_conn_and_pool
        conn.fetch.side_effect = [
            # provider_rows
            [MockRecord({
                "provider":       "aws",
                "total_cost":     0.0842,
                "total_runs":     10,
                "anomaly_count":  2,
                "total_records":  20,
                "avg_crs":        0.55,
                "top_region":     "ap-south-1",
            })],
            # prev_rows (week-over-week)
            [MockRecord({"provider": "aws", "total_cost": 0.07})],
            # hotspot_rows
            [MockRecord({
                "provider":      "aws",
                "region":        "ap-south-1",
                "total_cost":    0.0842,
                "anomaly_count": 2,
                "avg_crs":       0.55,
            })],
        ]

        resp = await test_app.get("/api/providers/compare?days=30")
        assert resp.status_code == 200
        body = resp.json()
        assert "providers"     in body
        assert "total_cost_usd" in body
        assert "hotspots"      in body
        assert "recommendation" in body
        assert isinstance(body["providers"], list)

    @pytest.mark.asyncio
    async def test_provider_trend_valid_provider(self, test_app, mock_conn_and_pool):
        conn, _ = mock_conn_and_pool
        conn.fetch.return_value = []  # no data - returns empty series

        resp = await test_app.get("/api/providers/trend/aws?days=7")
        assert resp.status_code == 200
        body = resp.json()
        assert body["provider"] == "aws"
        assert "series" in body

    @pytest.mark.asyncio
    async def test_provider_trend_invalid_provider_returns_400(self, test_app):
        resp = await test_app.get("/api/providers/trend/ibm?days=7")
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_provider_efficiency_returns_ranked_list(self, test_app, mock_conn_and_pool):
        conn, _ = mock_conn_and_pool
        conn.fetch.return_value = [
            MockRecord({
                "provider":      "aws",
                "runs":          10,
                "total_cost":    0.08,
                "anomalies":     1,
            }),
            MockRecord({
                "provider":      "gcp",
                "runs":          5,
                "total_cost":    0.03,
                "anomalies":     0,
            }),
        ]

        resp = await test_app.get("/api/providers/efficiency?days=7")
        assert resp.status_code == 200
        body = resp.json()
        assert "providers" in body
        # Ranked by efficiency_score descending
        assert len(body["providers"]) == 2

    def test_provider_recommendation_logic(self):
        """Unit test for the recommendation text generator."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
        from api.providers import _make_recommendation, ProviderSummary

        providers = [
            ProviderSummary(
                provider="aws", display_name="AWS", color="#F97316",
                total_cost_usd=0.08, total_runs=10, anomaly_count=5,
                anomaly_rate_pct=25.0, avg_crs_score=0.65,
                cost_share_pct=60.0, top_region="ap-south-1",
                week_over_week_pct=5.0,
            ),
            ProviderSummary(
                provider="gcp", display_name="GCP", color="#6366F1",
                total_cost_usd=0.03, total_runs=5, anomaly_count=0,
                anomaly_rate_pct=0.0, avg_crs_score=0.25,
                cost_share_pct=40.0, top_region="europe-west1",
                week_over_week_pct=2.0,
            ),
        ]

        rec = _make_recommendation(providers)
        assert isinstance(rec, str)
        assert len(rec) > 0
        # AWS has >20% anomaly rate so recommendation should mention it
        assert "aws" in rec.lower() or "AWS" in rec

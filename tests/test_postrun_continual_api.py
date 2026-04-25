"""Focused API tests for post-run graph endpoints and continual-capture routes."""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest

from tests.conftest import FAKE_POLICY, FAKE_USER, MockRecord


@dataclass
class _FakeScore:
    crs: float = 0.82
    decision: str = "AUTO_OPTIMISE"
    gat_prob: float = 0.71
    ai_recommendation: str = "Use reserved concurrency for high-cost stage."


class TestPostRunContinualRoutes:
    @pytest.mark.asyncio
    async def test_routes_are_registered(self, test_app):
        resp = await test_app.get("/openapi.json")
        assert resp.status_code == 200
        paths = resp.json().get("paths", {})
        assert "/api/postrun/graphs/seed-metrics" in paths
        assert "/api/postrun/graphs/domain-metrics" in paths
        assert "/api/postrun/graphs/anomaly-counts" in paths
        assert "/api/postrun/graphs/dataset-summaries" in paths
        assert "/api/continual/observe" in paths
        assert "/api/continual/feedback" in paths
        assert "/api/continual/retraining-readiness" in paths
        assert "/api/continual/retraining-export" in paths

    @pytest.mark.asyncio
    async def test_postrun_graph_missing_table_is_safe(
        self,
        test_app,
        mock_conn_and_pool,
        valid_jwt_headers,
    ):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.return_value = MockRecord(FAKE_USER)
        conn.fetch.side_effect = asyncpg.UndefinedTableError("missing")

        resp = await test_app.get(
            "/api/postrun/graphs/seed-metrics",
            headers=valid_jwt_headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "migration_missing"
        assert body["rows"] == []

    @pytest.mark.asyncio
    async def test_continual_observe_captures_event_and_queues_retraining(
        self,
        test_app,
        mock_conn_and_pool,
        valid_jwt_headers,
    ):
        conn, _ = mock_conn_and_pool
        conn.fetchrow = AsyncMock(
            side_effect=[
                MockRecord(FAKE_USER),
                MockRecord({"id": 101, "created_at": "2026-04-24T00:00:00+00:00"}),
                MockRecord(FAKE_POLICY),
                MockRecord({"id": 202, "created_at": "2026-04-24T00:00:01+00:00"}),
            ]
        )
        conn.execute = AsyncMock(return_value="INSERT 0 1")

        with patch("api.continual.score_pipeline", AsyncMock(return_value=_FakeScore())), patch(
            "api.continual.evaluate_policy",
            AsyncMock(
                return_value={
                    "decision": "WARN",
                    "reasons": ["Synthetic policy response."],
                    "policy_source": "inline",
                }
            ),
        ), patch(
            "api.continual.get_pade_runtime_status",
            return_value={"model_type": "canonical-gatv2-backend-adapter", "checkpoint_path": "/tmp/gat_best.pt"},
        ):
            resp = await test_app.post(
                "/api/continual/observe",
                headers=valid_jwt_headers,
                json={
                    "stage_name": "integration_test",
                    "provider": "aws",
                    "region": "ap-south-1",
                    "billed_cost": 1.25,
                    "payload": {"cost_deviation_pct": 33.0},
                    "feedback_label": "true_positive",
                    "feedback_notes": "Manual reviewer confirmed anomaly.",
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "captured"
        assert body["observation_id"] == 101
        assert body["inference_event_id"] == 202
        assert body["pending_retraining"] is True
        assert body["mode"] == "capture_only_no_online_training"

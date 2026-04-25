"""Regression tests for the production hardening pass."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import FAKE_USER, FAKE_VIEWER_USER, MockRecord

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from config import Settings


@pytest.mark.asyncio
async def test_checkpoint_upload_requires_authentication(test_app):
    resp = await test_app.post(
        "/api/pade/load-checkpoint",
        files={"file": ("gat.pt", b"x" * (1024 * 1024), "application/octet-stream")},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_checkpoint_upload_requires_admin_role(test_app, mock_conn_and_pool, valid_jwt_headers):
    conn, _ = mock_conn_and_pool
    conn.fetchrow.return_value = MockRecord(FAKE_VIEWER_USER)

    resp = await test_app.post(
        "/api/pade/load-checkpoint",
        headers=valid_jwt_headers,
        files={"file": ("gat.pt", b"x" * (1024 * 1024), "application/octet-stream")},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_invalid_checkpoint_upload_does_not_replace_active_checkpoint(test_app, mock_conn_and_pool, valid_jwt_headers, tmp_path):
    from pade import inference

    conn, _ = mock_conn_and_pool
    conn.fetchrow.return_value = MockRecord(FAKE_USER)

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    active_path = checkpoint_dir / "gat_best.pt"
    original_bytes = b"original-checkpoint"
    active_path.write_bytes(original_bytes)

    with patch.object(inference, "CHECKPOINT_DIR", checkpoint_dir), \
         patch.object(inference, "ACTIVE_CHECKPOINT_PATH", active_path), \
         patch.object(inference, "_build_model_from_checkpoint", return_value=None):
        resp = await test_app.post(
            "/api/pade/load-checkpoint",
            headers=valid_jwt_headers,
            files={"file": ("gat.pt", b"x" * (1024 * 1024), "application/octet-stream")},
        )

    assert resp.status_code == 422
    assert active_path.read_bytes() == original_bytes


@pytest.mark.asyncio
async def test_activity_logging_failure_returns_non_200(test_app, mock_conn_and_pool, valid_jwt_headers):
    conn, _ = mock_conn_and_pool
    conn.fetchrow.return_value = MockRecord(FAKE_USER)
    conn.execute.side_effect = Exception("db down")

    resp = await test_app.post(
        "/api/users/activity",
        headers=valid_jwt_headers,
        json={"action": "page_visit", "page": "dashboard"},
    )
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_cancelled_job_is_skipped_by_worker_loop():
    import api.jobs as jobs

    original_queue = jobs._job_queue
    original_registry = jobs._job_registry
    original_handlers = dict(jobs._HANDLERS)

    handler = AsyncMock(return_value={"ok": True})
    jobs._job_queue = asyncio.PriorityQueue()
    jobs._job_registry = {
        "job-1": {
            "job_id": "job-1",
            "job_type": "skip_test",
            "priority": 3,
            "status": "CANCELLED",
            "queued_at": "2026-01-01T00:00:00+00:00",
            "started_at": None,
            "finished_at": "2026-01-01T00:00:01+00:00",
            "result": None,
            "error": None,
            "progress_pct": 0,
        }
    }
    jobs._HANDLERS["skip_test"] = handler

    try:
        await jobs._job_queue.put((3, "job-1", "skip_test", {}))
        worker = asyncio.create_task(jobs._worker_loop(pool=None))
        await asyncio.sleep(0.1)
        worker.cancel()
        await worker

        handler.assert_not_awaited()
        assert jobs._job_registry["job-1"]["status"] == "CANCELLED"
    finally:
        jobs._job_queue = original_queue
        jobs._job_registry = original_registry
        jobs._HANDLERS.clear()
        jobs._HANDLERS.update(original_handlers)


@pytest.mark.asyncio
async def test_budget_status_schedules_alert_dispatch(test_app, mock_conn_and_pool):
    conn, _ = mock_conn_and_pool
    conn.fetch.return_value = [MockRecord({
        "id": 1,
        "team_name": "backend-team",
        "monthly_cap_usd": 50.0,
        "alert_at_pct": 80.0,
        "block_at_pct": 110.0,
        "webhook_url": "https://example.invalid/webhook",
    })]
    conn.fetchrow.return_value = MockRecord({"total": 45.0})

    scheduled = []

    def _schedule(coro, **kwargs):
        scheduled.append(kwargs["label"])
        return asyncio.create_task(coro)

    with patch("api.budget.safe_create_task", side_effect=_schedule), \
         patch("api.budget._fire_budget_webhook", new=AsyncMock()):
        resp = await test_app.get("/api/budget/status")

    assert resp.status_code == 200
    assert scheduled == ["budget alert notifications"]


@pytest.mark.asyncio
async def test_health_reports_degraded_database(test_app, mock_conn_and_pool):
    conn, _ = mock_conn_and_pool
    conn.execute.side_effect = Exception("db unavailable")

    resp = await test_app.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["database"] == "degraded"
    assert body["status"] == "degraded"


@pytest.mark.asyncio
async def test_health_reports_pade_component_status(test_app):
    fake_status = {
        "status": "ok",
        "model_loaded": True,
        "inference_ready": True,
        "checkpoint_path": "C:/tmp/gat_best.pt",
        "checkpoint_source": "manifest",
    }
    with patch("main.get_pade_runtime_status", return_value=fake_status):
        resp = await test_app.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["pade"]["checkpoint_source"] == "manifest"
    assert body["components"]["pade"] == "ok"


def test_settings_reject_placeholder_secrets_outside_local():
    settings = Settings(
        _env_file=None,
        ENVIRONMENT="production",
        DB_PASSWORD="CHANGE_ME_DB_PASSWORD",
        JWT_SECRET="CHANGE_ME_JWT_SECRET",
        GRAFANA_ADMIN_PASSWORD="CHANGE_ME_GRAFANA_PASSWORD",
        API_BASE_URL="https://api.costguard.example",
        DASHBOARD_BASE_URL="https://dashboard.costguard.example",
        ALLOWED_ORIGINS=["https://dashboard.costguard.example"],
    )
    with pytest.raises(RuntimeError):
        settings.validate_runtime_requirements()


def test_settings_parse_runtime_urls_and_origins():
    settings = Settings(
        _env_file=None,
        ENVIRONMENT="production",
        DB_PASSWORD="real-db-secret",
        JWT_SECRET="real-jwt-secret",
        GRAFANA_ADMIN_PASSWORD="grafana-secret",
        API_BASE_URL="https://api.costguard.example/",
        DASHBOARD_BASE_URL="https://dashboard.costguard.example/",
        ALLOWED_ORIGINS='["https://dashboard.costguard.example","https://admin.costguard.example"]',
        OPA_URL="https://opa.costguard.example/v1/data/costguard/result",
    )

    assert settings.api_http_base == "https://api.costguard.example"
    assert settings.dashboard_http_base == "https://dashboard.costguard.example"
    assert settings.ALLOWED_ORIGINS == [
        "https://dashboard.costguard.example",
        "https://admin.costguard.example",
    ]
    settings.validate_runtime_requirements()

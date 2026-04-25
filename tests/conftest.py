"""
tests/conftest.py - CostGuard v17.0 Enterprise Test Suite
=========================================================
Pytest fixtures providing:
  - isolated in-memory test doubles for database access
  - FastAPI test client via httpx ASGI transport
  - mock DB connection that never touches a real Postgres instance
  - pre-seeded test data for users, pipeline runs, and attribution rows
"""
import asyncio
import os
import pytest
import pytest_asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

# ── Test configuration ────────────────────────────────────────────────────────

pytest_plugins = ["pytest_asyncio"]


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks tests that need a real Docker DB (slow)"
    )


@pytest.hookimpl(wrapper=True)
def pytest_sessionfinish(session, exitstatus):
    """
    Windows/OneDrive can raise PermissionError while pytest cleans its temp tree.
    Ignore that teardown-only cleanup failure so real test results remain visible.
    """
    try:
        yield
    except PermissionError as exc:
        message = str(exc)
        if "Access is denied" in message and "pytest" in message.lower():
            reporter = session.config.pluginmanager.getplugin("terminalreporter")
            if reporter is not None:
                reporter.write_line(
                    f"[pytest] Ignored Windows temp cleanup error: {message}"
                )
            return
        raise


# ── Fake DB records ───────────────────────────────────────────────────────────

FAKE_USER = {
    "id":            1,
    "email":         "test@costguard.dev",
    "full_name":     "Test Engineer",
    "password_hash": "$2b$12$0bN48oCMkAQPODBJ6kwt8OCyOs7DzwLjDnvaS5WXmBOt4dzJDEl6K",  # "password"
    # FIX: role and is_active added — get_current_user() reads these columns
    # after Fix 2 added them to the schema.  Without them MockRecord raises KeyError.
    "role":          "admin",
    "is_active":     True,
    "created_at":    "2026-01-01T00:00:00+00:00",
}

FAKE_VIEWER_USER = {
    **FAKE_USER,
    "role": "viewer",
}

FAKE_ATTRIBUTION_ROW = {
    "id":                  1,
    "run_id":              "aaaa-bbbb-cccc-dddd",
    "stage_name":          "integration_test",
    "resource_type":       "compute",
    "billed_cost":         0.0240,
    "effective_cost":      0.0204,
    "billing_currency":    "USD",
    "usage_quantity":      24.0,
    "usage_unit":          "CPU-hours",
    "provider":            "aws",
    "region":              "ap-south-1",
    "cost_deviation_pct":  15.5,
    "historical_avg_cost": 0.0208,
    "crs_score":           0.82,
    "pade_decision":       "AUTO_OPTIMISE",
    "ai_recommendation":   "This stage has a CRS of 0.820. Apply spot instances. Saves ~45%.",
    "window_start":        "2026-01-01T10:00:00+00:00",
    "window_end":          "2026-01-01T11:00:00+00:00",
    "timestamp_start":     "2026-01-01T10:00:00+00:00",
    "timestamp_end":       "2026-01-01T10:05:00+00:00",
    "created_at":          "2026-01-01T10:00:00+00:00",
}

FAKE_POLICY = {
    "id":                      1,
    "warn_threshold":          0.50,
    "auto_optimise_threshold": 0.75,
    "block_threshold":         0.90,
    "policy_bundle": {
        "version": "v17.0",
        "thresholds": {
            "warn_threshold": 0.50,
            "auto_optimise_threshold": 0.75,
            "block_threshold": 0.90,
        },
        "rules": {
            "protected_branches": ["main", "release", "production"],
            "sensitive_stages": ["security_scan", "deploy_staging", "deploy_prod"],
            "block_pr_prod_deploys": True,
            "require_core_team_for_sensitive_stages": True,
            "stage_cost_ceiling_usd": {
                "build": 0.05,
                "integration_test": 0.08,
                "security_scan": 0.04,
                "docker_build": 0.09,
                "deploy_staging": 0.06,
                "deploy_prod": 0.08,
            },
        },
    },
    "updated_at":              "2026-01-01T00:00:00+00:00",
}

FAKE_PIPELINE_RUN = {
    "run_id":        "aaaa-bbbb-cccc-dddd",
    "user_id":       1,
    "branch_type":   "feature",
    "executor_type": "github_actions",
    "provider":      "aws",
    "region":        "ap-south-1",
    "total_cost_usd": 0.0240,
    "stage_count":   1,
    "is_anomalous":  True,
    "created_at":    "2026-01-01T10:00:00+00:00",
}


# ── Mock asyncpg connection ───────────────────────────────────────────────────

class MockRecord(dict):
    """Behaves like an asyncpg Record for dict(row) calls."""
    def keys(self):
        return super().keys()
    def values(self):
        return super().values()
    def __getitem__(self, key):
        return super().__getitem__(key)


def make_mock_conn(
    fetch_returns=None,
    fetchrow_returns=None,
    fetchval_returns=None,
    execute_returns="INSERT 0 1",
):
    """Build a mock asyncpg connection that returns pre-set data."""
    conn = AsyncMock()
    conn.fetch     = AsyncMock(return_value=[MockRecord(r) for r in (fetch_returns or [])])
    conn.fetchrow  = AsyncMock(return_value=MockRecord(fetchrow_returns) if fetchrow_returns else None)
    conn.fetchval  = AsyncMock(return_value=fetchval_returns)
    conn.execute   = AsyncMock(return_value=execute_returns)
    conn.add_listener    = AsyncMock()
    conn.remove_listener = AsyncMock()

    # Make conn usable as async context manager: `async with pool.acquire() as conn`
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__  = AsyncMock(return_value=False)
    return conn, cm


def make_mock_pool(conn, conn_cm):
    """Build a mock asyncpg Pool that yields our mock connection."""
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=conn_cm)
    pool.close   = AsyncMock()
    return pool


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_conn_and_pool():
    """
    Returns (conn, pool) where conn is a pre-configured mock asyncpg connection.
    Override conn.fetch.return_value / conn.fetchrow.return_value in each test.
    """
    conn, cm = make_mock_conn(
        fetch_returns   = [FAKE_ATTRIBUTION_ROW],
        fetchrow_returns = FAKE_POLICY,
    )
    pool = make_mock_pool(conn, cm)
    return conn, pool


@pytest.fixture(autouse=True)
def disable_external_notifications():
    """
    Force notification credentials off during tests so no background Slack or SMTP
    coroutines are spawned from environment-specific .env settings.
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
    from config import settings

    original = {
        "SLACK_BOT_TOKEN": settings.SLACK_BOT_TOKEN,
        "SLACK_DEFAULT_CHANNEL": settings.SLACK_DEFAULT_CHANNEL,
        "SMTP_HOST": settings.SMTP_HOST,
        "SMTP_USER": settings.SMTP_USER,
        "SMTP_PASSWORD": settings.SMTP_PASSWORD,
        "SMTP_FROM": settings.SMTP_FROM,
    }
    settings.SLACK_BOT_TOKEN = None
    settings.SLACK_DEFAULT_CHANNEL = "#costguard-alerts"
    settings.SMTP_HOST = None
    settings.SMTP_USER = None
    settings.SMTP_PASSWORD = None
    settings.SMTP_FROM = None
    try:
        yield
    finally:
        for key, value in original.items():
            setattr(settings, key, value)


@pytest_asyncio.fixture
async def test_app(mock_conn_and_pool):
    """
    FastAPI app fixture with the DB pool swapped for a mock.
    Uses httpx.ASGITransport so requests stay in-process and never hit localhost.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

    _, pool = mock_conn_and_pool

    # Patch create_pool so no real DB is contacted
    with patch("database.create_pool", AsyncMock(return_value=pool)), \
         patch("database.run_migration", AsyncMock()), \
         patch("pade.inference.bootstrap_gat_checkpoint", MagicMock()), \
         patch("api.ingest.pcam_attribution_loop", AsyncMock()), \
         patch("api.jobs.start_worker", MagicMock(return_value=AsyncMock())):

        from main import app
        app.state.db = pool

        import httpx
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


@pytest.fixture
def valid_jwt_headers():
    """Return Authorization headers with a valid test JWT."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
    from datetime import datetime, timedelta, timezone
    from jose import jwt
    from config import settings

    token = jwt.encode(
        {"sub": FAKE_USER["email"], "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM,
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use the default asyncio event loop policy (required for pytest-asyncio)."""
    return asyncio.DefaultEventLoopPolicy()

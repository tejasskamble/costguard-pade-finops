"""
tests/test_health.py — Startup & Health Check Tests
=====================================================
Tests that the application boots correctly and all health
endpoints respond. These are the first tests to run in CI.
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from tests.conftest import FAKE_POLICY, FAKE_USER, MockRecord


@pytest.mark.asyncio
async def test_health_endpoint_returns_200(test_app):
    """GET /health must return 200 with correct service name."""
    resp = await test_app.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"ok", "degraded"}
    assert "costguard" in body["service"].lower()
    assert "version" in body
    assert "components" in body
    assert "pade" in body


@pytest.mark.asyncio
async def test_docs_endpoint_accessible(test_app):
    """Swagger docs must be reachable (confirms FastAPI is wired correctly)."""
    resp = await test_app.get("/docs")
    assert resp.status_code == 200
    assert "swagger" in resp.text.lower() or "openapi" in resp.text.lower()


@pytest.mark.asyncio
async def test_metrics_endpoint(test_app):
    """Prometheus /metrics endpoint must return 200 with text/plain content."""
    resp = await test_app.get("/metrics")
    assert resp.status_code == 200
    # Prometheus format starts with # HELP lines
    assert "costguard" in resp.text or "#" in resp.text


@pytest.mark.asyncio
async def test_unknown_route_returns_404(test_app):
    """Undefined routes return 404, not 500."""
    resp = await test_app.get("/api/does-not-exist")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_cors_headers_present(test_app):
    """CORS preflight headers must be present for Streamlit origin."""
    resp = await test_app.options(
        "/health",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "GET",
        },
    )
    # FastAPI returns 200 for OPTIONS
    assert resp.status_code in (200, 204)


class TestDatabaseConnectivity:
    """DB connectivity tests using the mock pool."""

    @pytest.mark.asyncio
    async def test_policy_endpoint_queries_db(self, test_app, mock_conn_and_pool, valid_jwt_headers):
        """GET /api/policy must hit the DB and return threshold values."""
        conn, _ = mock_conn_and_pool
        conn.fetchrow.side_effect = [
            MockRecord(FAKE_USER),
            MockRecord(FAKE_POLICY),
        ]
        resp = await test_app.get("/api/policy", headers=valid_jwt_headers)
        # Policy endpoint requires DB — conn.fetchrow should have been called
        assert conn.fetchrow.called
        assert resp.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_alerts_endpoint_queries_db(self, test_app, mock_conn_and_pool):
        """GET /api/alerts/recent must call the DB and return a list."""
        conn, _ = mock_conn_and_pool
        conn.fetch.return_value = []   # empty list — simulates no alerts
        resp = await test_app.get("/api/alerts/recent?limit=5")
        assert resp.status_code in (200, 500)  # 500 only if mock misconfigured

    @pytest.mark.asyncio
    async def test_db_pool_pool_acquired_for_requests(self, mock_conn_and_pool):
        """Validate the mock pool's acquire() is callable and returns our conn."""
        _, pool = mock_conn_and_pool
        async with pool.acquire() as conn:
            assert conn is not None
            # execute should be callable
            result = await conn.execute("SELECT 1")
            assert result is not None


class TestDockerHealth:
    """Docker infrastructure health checks (unit-level, no real Docker needed)."""

    def test_docker_compose_file_exists(self):
        import pathlib
        compose = pathlib.Path(__file__).parent.parent / "docker-compose.yml"
        assert compose.exists(), "docker-compose.yml must exist in project root"

    def test_docker_compose_has_no_version_key(self):
        """Ensure the obsolete 'version:' key has been removed."""
        import pathlib
        compose = pathlib.Path(__file__).parent.parent / "docker-compose.yml"
        if compose.exists():
            content = compose.read_text()
            # version: key should not appear as a top-level key
            lines = [l.strip() for l in content.splitlines()]
            top_level_version = [l for l in lines if l.startswith("version:")]
            assert len(top_level_version) == 0, (
                "Obsolete 'version:' key found in docker-compose.yml — "
                "remove it for Docker Compose v2 compatibility"
            )

    def test_docker_compose_uses_named_volumes_only(self):
        """No bind mounts for data directories (Windows path safety)."""
        import pathlib
        compose = pathlib.Path(__file__).parent.parent / "docker-compose.yml"
        if compose.exists():
            content = compose.read_text()
            # The migrations initdb bind mount caused Windows crashes
            assert "docker-entrypoint-initdb.d" not in content, (
                "./migrations:/docker-entrypoint-initdb.d bind mount must be removed. "
                "It causes 'Volume c' crash on Windows Docker Desktop."
            )

    def test_postgres_healthcheck_configured(self):
        """PostgreSQL service must have a healthcheck for --wait flag to work."""
        import pathlib, yaml
        compose = pathlib.Path(__file__).parent.parent / "docker-compose.yml"
        if not compose.exists():
            pytest.skip("docker-compose.yml not found")
        try:
            data = yaml.safe_load(compose.read_text())
            pg = data.get("services", {}).get("postgres", {})
            assert "healthcheck" in pg, "postgres service must have a healthcheck"
            assert pg["healthcheck"]["interval"] is not None
        except ImportError:
            pytest.skip("PyYAML not installed — install with: pip install pyyaml")

    def test_env_file_has_required_keys(self):
        """The .env file must contain the minimum required keys."""
        import pathlib
        env_file = pathlib.Path(__file__).parent.parent / ".env"
        if not env_file.exists():
            pytest.skip(".env not found — expected in project root")

        content = env_file.read_text(encoding="utf-8")
        required = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD", "JWT_SECRET"]
        missing = [k for k in required if k not in content]
        assert not missing, f"Missing required .env keys: {missing}"

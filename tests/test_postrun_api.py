"""Focused API tests for /api/postrun/* routes."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import FAKE_USER, MockRecord


def _touch(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestPostRunRoutes:
    @pytest.mark.asyncio
    async def test_postrun_routes_are_registered(self, test_app):
        resp = await test_app.get("/openapi.json")
        assert resp.status_code == 200
        paths = resp.json().get("paths", {})
        assert "/api/postrun/summary" in paths
        assert "/api/postrun/models" in paths
        assert "/api/postrun/import" in paths
        assert "/api/postrun/import/history" in paths

    @pytest.mark.asyncio
    async def test_postrun_import_dry_run_has_zero_db_writes(
        self,
        test_app,
        mock_conn_and_pool,
        valid_jwt_headers,
        tmp_path,
    ):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.return_value = MockRecord(FAKE_USER)
        conn.execute.reset_mock()

        results_root = tmp_path / "results"
        results_root.mkdir(parents=True, exist_ok=True)

        resp = await test_app.post(
            "/api/postrun/import",
            headers=valid_jwt_headers,
            json={
                "dry_run": True,
                "results_root": str(results_root),
                "chunk_size": 100_000,
                "min_ensemble_f1": 0.80,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "dry_run"
        assert body["summary"]["dry_run"] is True
        assert body["seed_rows"] == 10
        assert conn.execute.await_count == 0

    @pytest.mark.asyncio
    async def test_postrun_models_handles_missing_and_corrupt_artifacts_safely(
        self,
        test_app,
        mock_conn_and_pool,
        valid_jwt_headers,
        tmp_path,
    ):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.return_value = MockRecord(FAKE_USER)

        results_root = tmp_path / "results"
        # Corrupt manifest for seed 42
        _touch(results_root / "trials" / "seed_42" / "trial_manifest.json", "{not-valid-json")
        # Valid non-empty checkpoint artifact to ensure positive detection still works
        ckpt = (
            results_root
            / "trials"
            / "seed_42"
            / "synthetic"
            / "run_1"
            / "checkpoints"
            / "lstm_best.pt"
        )
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(b"checkpoint-bytes")

        resp = await test_app.get(
            "/api/postrun/models",
            headers=valid_jwt_headers,
            params={"results_root": str(results_root)},
        )
        assert resp.status_code == 200
        body = resp.json()
        seed_models = body["seed_models"]
        assert len(seed_models) == 10

        seed_42 = next(item for item in seed_models if item.get("seed") == 42)
        seed_52 = next(item for item in seed_models if item.get("seed") == 52)

        assert seed_42["status"] == "invalid_manifest"
        assert seed_52["status"] == "missing"

        synthetic_entry = next(domain for domain in seed_42["domains"] if domain.get("domain") == "synthetic")
        assert synthetic_entry["artifacts"]["lstm_best.pt"]["exists"] is True

    @pytest.mark.asyncio
    async def test_postrun_quality_gate_is_inline_when_opa_is_unavailable(
        self,
        test_app,
        mock_conn_and_pool,
        valid_jwt_headers,
        tmp_path,
    ):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.return_value = MockRecord(FAKE_USER)

        results_root = tmp_path / "results"
        results_root.mkdir(parents=True, exist_ok=True)

        from config import settings

        original_opa_postrun_url = settings.OPA_POSTRUN_URL
        settings.OPA_POSTRUN_URL = "http://127.0.0.1:65535/v1/data/costguard/postrun_result"
        try:
            resp = await test_app.get(
                "/api/postrun/summary",
                headers=valid_jwt_headers,
                params={"results_root": str(results_root), "min_ensemble_f1": 0.80},
            )
        finally:
            settings.OPA_POSTRUN_URL = original_opa_postrun_url

        assert resp.status_code == 200
        quality_gate = resp.json()["quality_gate"]
        assert quality_gate["source"] == "inline"
        assert quality_gate["decision"] == "BLOCK"
        assert any("Incomplete trials detected" in reason for reason in quality_gate["reasons"])


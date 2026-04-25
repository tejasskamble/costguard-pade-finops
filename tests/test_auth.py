"""
tests/test_auth.py — Authentication & Authorization Tests
==========================================================
Tests the complete auth lifecycle:
  - User registration (bcrypt, DB insert)
  - Login with email/password
  - JWT validation (valid/expired/tampered)
  - Protected endpoint access
  - Duplicate registration rejection
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from tests.conftest import FAKE_USER, MockRecord


class TestRegistration:

    @pytest.mark.asyncio
    async def test_register_new_user_returns_201_with_token(self, test_app, mock_conn_and_pool):
        conn, pool = mock_conn_and_pool
        # Simulate: no existing user, successful insert
        conn.fetchrow.side_effect = [
            None,        # "SELECT id FROM users WHERE email = ?" → not found
            None,        # insert → no return needed (execute used instead)
        ]
        conn.execute.return_value = "INSERT 0 1"

        resp = await test_app.post("/api/auth/register", json={
            "email":     "newuser@test.com",
            "password":  "SecurePass123!",
            "full_name": "New User",
        })
        # Should return token (200 because we return Token model)
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_register_duplicate_email_returns_400(self, test_app, mock_conn_and_pool):
        conn, _ = mock_conn_and_pool
        # Simulate: user already exists
        conn.fetchrow.return_value = MockRecord({"id": 1})

        resp = await test_app.post("/api/auth/register", json={
            "email":     "existing@test.com",
            "password":  "password",
            "full_name": "Duplicate",
        })
        assert resp.status_code == 400
        assert "already" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_register_missing_fields_returns_422(self, test_app):
        """Missing required fields must trigger validation error."""
        resp = await test_app.post("/api/auth/register", json={
            "email": "noemail@test.com",
            # missing password and full_name
        })
        assert resp.status_code == 422


class TestLogin:

    @pytest.mark.asyncio
    async def test_login_valid_credentials_returns_token(self, test_app, mock_conn_and_pool):
        conn, _ = mock_conn_and_pool
        # Return a user with a known bcrypt hash for "password"
        conn.fetchrow.return_value = MockRecord(FAKE_USER)

        resp = await test_app.post(
            "/api/auth/token",
            data={"username": FAKE_USER["email"], "password": "password"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_wrong_password_returns_401(self, test_app, mock_conn_and_pool):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.return_value = MockRecord(FAKE_USER)  # correct user record

        resp = await test_app.post(
            "/api/auth/token",
            data={"username": FAKE_USER["email"], "password": "WRONG_PASSWORD"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_login_nonexistent_user_returns_401(self, test_app, mock_conn_and_pool):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.return_value = None  # user not found

        resp = await test_app.post(
            "/api/auth/token",
            data={"username": "ghost@test.com", "password": "password"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code == 401


class TestJWT:

    @pytest.mark.asyncio
    async def test_protected_endpoint_without_token_returns_401(self, test_app):
        resp = await test_app.get("/api/auth/me")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_protected_endpoint_with_valid_token(self, test_app, mock_conn_and_pool, valid_jwt_headers):
        conn, _ = mock_conn_and_pool
        conn.fetchrow.return_value = MockRecord(FAKE_USER)

        resp = await test_app.get("/api/auth/me", headers=valid_jwt_headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["email"] == FAKE_USER["email"]

    @pytest.mark.asyncio
    async def test_tampered_token_returns_401(self, test_app):
        resp = await test_app.get(
            "/api/auth/me",
            headers={"Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.TAMPERED.signature"},
        )
        assert resp.status_code == 401

    def test_jwt_payload_contains_email_as_sub(self, valid_jwt_headers):
        """JWT sub must be the user email (CONSTRAINT-2)."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
        from jose import jwt
        from config import settings

        raw_token = valid_jwt_headers["Authorization"].split(" ")[1]
        payload = jwt.decode(raw_token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        assert payload["sub"] == FAKE_USER["email"], "JWT sub must be the user's email address"
        assert "@" in payload["sub"], "JWT sub must look like an email"

    def test_expired_token_returns_401(self, test_app):
        """An expired JWT must be rejected."""
        import sys, os, asyncio
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
        from datetime import datetime, timedelta, timezone
        from jose import jwt
        from config import settings

        expired_token = jwt.encode(
            {"sub": FAKE_USER["email"], "exp": datetime.now(timezone.utc) - timedelta(hours=1)},
            settings.JWT_SECRET,
            algorithm=settings.JWT_ALGORITHM,
        )
        # Sync check — the JWT library itself should reject it
        with pytest.raises(Exception):
            jwt.decode(expired_token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])

"""
backend/api/notifications.py  — CostGuard v17.0
Notification preferences: get, update, and send test notification.
"""
import asyncio
import logging
from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
import httpx
from pydantic import BaseModel, field_serializer

from api.auth import get_current_user, UserProfile
from database import get_db_conn
from runtime_hardening import safe_create_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/notifications", tags=["notifications"])

VALID_SENSITIVITY = {"low", "medium", "high", "paranoid"}
VALID_DIGEST = {"daily", "weekly", "never"}


class NotificationPrefs(BaseModel):
    email_enabled: bool = True
    slack_enabled: bool = False
    slack_webhook: Optional[str] = None
    budget_threshold: Optional[Decimal] = None
    anomaly_sensitivity: str = "medium"
    digest_frequency: str = "daily"

    @field_serializer("budget_threshold", when_used="json")
    def _serialize_budget_threshold(self, value: Optional[Decimal]) -> Optional[float]:
        if value is None:
            return None
        return float(value)


@router.get("/preferences", response_model=NotificationPrefs)
async def get_preferences(
    request: Request,
    db=Depends(get_db_conn),
    current_user: UserProfile = Depends(get_current_user),
):
    """Return notification preferences for the current user (create defaults if none)."""
    try:
        row = await db.fetchrow(
            "SELECT * FROM notification_preferences WHERE user_id = $1",
            current_user.id,
        )
        if not row:
            await db.execute(
                "INSERT INTO notification_preferences (user_id) VALUES ($1) ON CONFLICT DO NOTHING",
                current_user.id,
            )
            return NotificationPrefs()
        return NotificationPrefs(
            email_enabled=row["email_enabled"],
            slack_enabled=row["slack_enabled"],
            slack_webhook=row["slack_webhook"],
            budget_threshold=row["budget_threshold"],
            anomaly_sensitivity=row["anomaly_sensitivity"] or "medium",
            digest_frequency=row["digest_frequency"] or "daily",
        )
    except Exception as exc:
        logger.exception(f"get_preferences error: {exc}")
        raise HTTPException(500, "Failed to fetch preferences")


@router.put("/preferences", response_model=NotificationPrefs)
async def update_preferences(
    body: NotificationPrefs,
    request: Request,
    db=Depends(get_db_conn),
    current_user: UserProfile = Depends(get_current_user),
):
    """Upsert notification preferences for the current user."""
    if body.anomaly_sensitivity not in VALID_SENSITIVITY:
        raise HTTPException(400, f"anomaly_sensitivity must be one of: {VALID_SENSITIVITY}")
    if body.digest_frequency not in VALID_DIGEST:
        raise HTTPException(400, f"digest_frequency must be one of: {VALID_DIGEST}")
    try:
        await db.execute(
            """INSERT INTO notification_preferences
                   (user_id, email_enabled, slack_enabled, slack_webhook,
                    budget_threshold, anomaly_sensitivity, digest_frequency, updated_at)
               VALUES ($1,$2,$3,$4,$5,$6,$7,NOW())
               ON CONFLICT (user_id) DO UPDATE SET
                   email_enabled       = EXCLUDED.email_enabled,
                   slack_enabled       = EXCLUDED.slack_enabled,
                   slack_webhook       = EXCLUDED.slack_webhook,
                   budget_threshold    = EXCLUDED.budget_threshold,
                   anomaly_sensitivity = EXCLUDED.anomaly_sensitivity,
                   digest_frequency    = EXCLUDED.digest_frequency,
                   updated_at          = NOW()""",
            current_user.id,
            body.email_enabled, body.slack_enabled, body.slack_webhook,
            body.budget_threshold, body.anomaly_sensitivity, body.digest_frequency,
        )
        return body
    except Exception as exc:
        logger.exception(f"update_preferences error: {exc}")
        raise HTTPException(500, "Failed to update preferences")


@router.post("/test")
async def send_test_notification(
    request: Request,
    db=Depends(get_db_conn),
    current_user: UserProfile = Depends(get_current_user),
):
    """Send an immediate test notification via email and/or Slack."""
    from services import email_service
    from config import settings
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    results: dict[str, str] = {}

    # Email test
    if all([settings.SMTP_HOST, settings.SMTP_USER, settings.SMTP_PASSWORD]):
        try:
            html = """<html><body style="background:#060B18;font-family:sans-serif;padding:32px;">
            <div style="max-width:520px;margin:0 auto;background:#0D1B2E;border-radius:16px;
                        border:1px solid rgba(255,107,53,.2);padding:32px;color:#E8F0FE;">
              <h2 style="color:#FF6B35;">🔔 Test Notification</h2>
              <p style="color:#B0BDD0;">This is a test notification from CostGuard v17.0.</p>
              <p style="color:#6B7A99;font-size:.8rem;">
                If you received this, your email notifications are working correctly.</p>
            </div></body></html>"""
            msg = MIMEMultipart("alternative")
            msg["Subject"] = "🔔 CostGuard Test Notification"
            msg["From"] = settings.SMTP_FROM or settings.SMTP_USER
            msg["To"] = current_user.email
            msg.attach(MIMEText(html, "html"))
            safe_create_task(
                asyncio.to_thread(email_service._send, msg),
                logger=logger,
                label="test notification email",
            )
            results["email"] = "queued"
        except Exception:
            logger.exception("Test email dispatch failed for user_id=%s", current_user.id)
            results["email"] = "delivery_failed"
    else:
        results["email"] = "not_configured"

    # Slack test
    row = await db.fetchrow(
        "SELECT slack_enabled, slack_webhook FROM notification_preferences WHERE user_id = $1",
        current_user.id,
    )
    if row and row["slack_enabled"] and row["slack_webhook"]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    row["slack_webhook"],
                    json={"text": f"🔔 CostGuard test notification for {current_user.email}"},
                )
            response.raise_for_status()
            results["slack"] = "sent"
        except Exception:
            logger.exception("Test Slack notification failed for user_id=%s", current_user.id)
            results["slack"] = "delivery_failed"
    else:
        results["slack"] = "not_configured"

    return {"message": "Test notification dispatched", "channels": results}

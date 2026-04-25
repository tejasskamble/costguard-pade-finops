"""
backend/peg/notifier.py
Fixes GAP-5: send_email_alert() routes to recipient_email (not settings.SMTP_USER).
Fixes GAP-6: ai_recommendation included in email body.
CONSTRAINT-6: STARTTLS on port 587.
"""
import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

import aiosmtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from config import settings

logger = logging.getLogger(__name__)
_MONEY_QUANT = Decimal("0.0001")


def _to_decimal(value: Decimal | float | int) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _format_cost(value: Decimal | float | int) -> str:
    amount = _to_decimal(value).quantize(_MONEY_QUANT, rounding=ROUND_HALF_UP)
    return f"${amount:.4f}"

# Decision → color mapping for both Slack and email badges
DECISION_COLORS = {
    "ALLOW":         "#10B981",   # emerald
    "WARN":          "#F59E0B",   # amber
    "AUTO_OPTIMISE": "#8B5CF6",   # violet
    "BLOCK":         "#F43F5E",   # rose
}

DECISION_EMOJI = {
    "ALLOW":         "✅",
    "WARN":          "⚠️",
    "AUTO_OPTIMISE": "⚙️",
    "BLOCK":         "🚫",
}


# ─── Slack alert ──────────────────────────────────────────────────────────────

async def send_slack_alert(
    decision: str,
    run_id: str,
    crs: float,
    stage_name: str,
    cost: Decimal | float | int,
    optimisation: Optional[str] = None,
    ai_recommendation: str = "",
) -> None:
    """Send a formatted Slack alert via chat.postMessage (not webhook)."""
    if not settings.SLACK_BOT_TOKEN:
        logger.warning("SLACK_BOT_TOKEN not set — skipping Slack alert")
        return

    client = WebClient(token=settings.SLACK_BOT_TOKEN)
    color = DECISION_COLORS.get(decision, "#6366F1")
    emoji = DECISION_EMOJI.get(decision, "🔔")

    fields = [
        {"title": "Stage",     "value": stage_name,      "short": True},
        {"title": "CRS Score", "value": f"{crs:.3f}",    "short": True},
        {"title": "Cost",      "value": _format_cost(cost),  "short": True},
        {"title": "Decision",  "value": decision,         "short": True},
    ]
    if optimisation:
        fields.append({"title": "Auto-Optimisation", "value": optimisation, "short": False})
    if ai_recommendation:
        fields.append({"title": "AI Recommendation", "value": ai_recommendation, "short": False})

    attachment = {
        "color":  color,
        "title":  f"{emoji} CostGuard {decision}: Run {run_id[:8]}",
        "fields": fields,
        "footer": "CostGuard PADE Engine v17.0",
        "ts":     int(datetime.now(timezone.utc).timestamp()),
    }

    try:
        await asyncio.to_thread(
            client.chat_postMessage,
            channel=settings.SLACK_DEFAULT_CHANNEL,
            text=f"Pipeline Cost Alert: {decision} — CRS {crs:.3f}",
            attachments=[attachment],
        )
        logger.info(f"Slack alert sent for run {run_id}")
    except SlackApiError as exc:
        logger.error(f"Slack API error: {exc.response.get('error', 'unknown')}")
    except Exception as exc:
        logger.warning(f"Slack alert failed (non-fatal): {exc}")


# ─── Email alert ──────────────────────────────────────────────────────────────

async def send_email_alert(
    decision: str,
    run_id: str,
    crs: float,
    stage_name: str,
    cost: Decimal | float | int,
    recipient_email: str,           # GAP-5 fix: no longer hardcoded to settings.SMTP_USER
    optimisation: Optional[str] = None,
    ai_recommendation: str = "",    # GAP-6 fix: GPT-4o-mini recommendation
) -> None:
    """
    Send a dark-themed HTML anomaly alert to the pipeline owner's actual email.
    GAP-5 fix: recipient_email is passed explicitly — not settings.SMTP_USER.
    GAP-6 fix: ai_recommendation included in the email body.
    CONSTRAINT-6: STARTTLS on port 587.
    """
    if not all([settings.SMTP_HOST, settings.SMTP_USER, settings.SMTP_PASSWORD]):
        logger.warning("SMTP settings incomplete — skipping email alert")
        return

    badge_color = DECISION_COLORS.get(decision, "#6366F1")
    badge_bg    = badge_color + "1A"  # 10% opacity hex
    emoji       = DECISION_EMOJI.get(decision, "🔔")

    opt_row = ""
    if optimisation:
        opt_row = f"""
        <tr>
          <td style="padding:10px 16px;color:#6B7A99;font-size:0.88rem;">Action Taken</td>
          <td style="padding:10px 16px;color:#8B5CF6;font-weight:600;">{optimisation}</td>
        </tr>"""

    ai_section = ""
    if ai_recommendation:
        ai_section = f"""
        <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);
                    border-left:3px solid #6366F1;border-radius:12px;
                    padding:16px 20px;margin:20px 0;">
          <div style="color:#A5B4FC;font-size:0.78rem;font-weight:700;
                      letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">
            🤖 AI Recommendation
          </div>
          <div style="color:#E8EAF6;font-size:0.9rem;line-height:1.7;">
            {ai_recommendation}
          </div>
        </div>"""

    html = f"""
    <html>
    <body style="font-family:'DM Sans',sans-serif;background:#050B1A;
                 color:#E8EAF6;margin:0;padding:32px;">
      <div style="max-width:600px;margin:0 auto;background:#0C1428;
                  border-radius:20px;border:1px solid rgba(99,102,241,0.18);
                  padding:40px;box-shadow:0 40px 80px rgba(0,0,0,0.5);">

        <!-- Header -->
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:28px;">
          <div style="font-size:2.5rem;">🛡️</div>
          <div>
            <div style="font-size:1.6rem;font-weight:800;
                        background:linear-gradient(135deg,#fff,#A5B4FC);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                        background-clip:text;">CostGuard</div>
            <div style="color:#6B7A99;font-size:0.82rem;">Pipeline Cost Intelligence</div>
          </div>
        </div>

        <!-- Decision badge -->
        <div style="background:{badge_bg};border:1px solid {badge_color}33;
                    border-radius:14px;padding:16px 20px;margin-bottom:24px;
                    display:flex;align-items:center;gap:12px;">
          <span style="font-size:1.8rem;">{emoji}</span>
          <div>
            <div style="color:{badge_color};font-weight:800;font-size:1.1rem;
                        letter-spacing:0.03em;">{decision}</div>
            <div style="color:#94A3B8;font-size:0.85rem;margin-top:2px;">
              Pipeline cost anomaly detected — immediate attention required
            </div>
          </div>
        </div>

        <!-- Details table -->
        <table style="width:100%;border-collapse:collapse;background:rgba(255,255,255,0.02);
                      border-radius:12px;overflow:hidden;margin-bottom:8px;">
          <tr style="border-bottom:1px solid rgba(99,102,241,0.1);">
            <td style="padding:10px 16px;color:#6B7A99;font-size:0.88rem;">Run ID</td>
            <td style="padding:10px 16px;color:#E8EAF6;font-family:monospace;
                       font-size:0.82rem;">{run_id}</td>
          </tr>
          <tr style="border-bottom:1px solid rgba(99,102,241,0.1);">
            <td style="padding:10px 16px;color:#6B7A99;font-size:0.88rem;">Stage</td>
            <td style="padding:10px 16px;color:#E8EAF6;font-weight:600;">{stage_name}</td>
          </tr>
          <tr style="border-bottom:1px solid rgba(99,102,241,0.1);">
            <td style="padding:10px 16px;color:#6B7A99;font-size:0.88rem;">CRS Score</td>
            <td style="padding:10px 16px;color:{badge_color};font-weight:800;
                       font-size:1.1rem;font-family:monospace;">{crs:.3f}</td>
          </tr>
          <tr style="border-bottom:1px solid rgba(99,102,241,0.1);">
            <td style="padding:10px 16px;color:#6B7A99;font-size:0.88rem;">Billed Cost</td>
            <td style="padding:10px 16px;color:#E8EAF6;">{_format_cost(cost)} USD</td>
          </tr>
          <tr style="border-bottom:1px solid rgba(99,102,241,0.1);">
            <td style="padding:10px 16px;color:#6B7A99;font-size:0.88rem;">Decision</td>
            <td style="padding:10px 16px;">
              <span style="background:{badge_bg};color:{badge_color};
                           border:1px solid {badge_color}55;border-radius:20px;
                           padding:3px 12px;font-size:0.78rem;font-weight:700;">
                {decision}
              </span>
            </td>
          </tr>
          {opt_row}
        </table>

        {ai_section}

        <!-- CTA button -->
        <div style="text-align:center;margin:28px 0 20px;">
          <a href="{settings.dashboard_http_base}"
             style="display:inline-block;background:linear-gradient(135deg,#4F46E5,#6366F1);
                    color:#fff;text-decoration:none;padding:14px 32px;border-radius:12px;
                    font-weight:700;font-size:0.95rem;
                    box-shadow:0 8px 24px rgba(99,102,241,0.35);">
            View in Dashboard →
          </a>
        </div>

        <!-- Footer -->
        <div style="border-top:1px solid rgba(99,102,241,0.15);padding-top:20px;
                    color:#4B5563;font-size:0.78rem;text-align:center;line-height:1.7;">
          Alert sent to <span style="color:#6366F1;">{recipient_email}</span>.<br>
          Powered by CostGuard PADE Engine v17.0.
        </div>
      </div>
    </body>
    </html>
    """

    message = MIMEMultipart("alternative")
    message["Subject"] = f"[CostGuard {decision}] {stage_name} — CRS {crs:.3f}"
    message["From"]    = settings.SMTP_FROM or settings.SMTP_USER
    message["To"]      = recipient_email   # GAP-5 fix: user's actual email
    message.attach(MIMEText(html, "html"))

    try:
        await aiosmtplib.send(
            message,
            hostname=settings.SMTP_HOST,
            port=settings.SMTP_PORT,
            username=settings.SMTP_USER,
            password=settings.SMTP_PASSWORD,
            start_tls=True,   # CONSTRAINT-6: STARTTLS on 587
            timeout=10,
        )
        logger.info(f"Email alert sent to {recipient_email} for run {run_id}")
    except Exception as exc:
        logger.warning(f"Email alert failed (non-fatal): {exc}")

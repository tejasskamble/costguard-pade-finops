"""
backend/services/email_service.py
CostGuard v17.0 — HTML email service for OTP, confirmations, and support enquiries.

All emails use Gmail STARTTLS (port 587) with a 10-second socket timeout.
Never crash the calling endpoint — all SMTP errors are caught and logged.
"""
import logging
import smtplib
import socket
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from config import settings

logger = logging.getLogger(__name__)

# ── Brand constants ───────────────────────────────────────────────────────────
_BRAND = "CostGuard"
_VERSION = "v17.0"
_COLLEGE = "Sir Parshurambhau College (Autonomous), Pune"
_SUPPORT_EMAIL = settings.support_email

_BASE_STYLE = """
<style>
  body { margin:0; padding:0; background:#060B18; font-family:'Segoe UI',Arial,sans-serif; }
  .wrapper { max-width:560px; margin:40px auto; background:#0D1B2E;
             border-radius:16px; overflow:hidden;
             border:1px solid rgba(255,107,53,.2); }
  .header { background:linear-gradient(135deg,#FF6B35 0%,#C084FC 100%);
            padding:32px; text-align:center; }
  .header h1 { margin:0; color:#fff; font-size:1.6rem; font-weight:800;
               letter-spacing:-0.02em; }
  .header span { color:rgba(255,255,255,.7); font-size:.85rem; }
  .body { padding:32px; color:#E8F0FE; }
  .body p { color:#B0BDD0; line-height:1.6; margin:0 0 16px; }
  .otp-box { display:flex; gap:12px; justify-content:center; margin:28px 0; }
  .otp-digit { width:52px; height:64px; background:#111D35;
               border:2px solid rgba(255,107,53,.5); border-radius:12px;
               display:flex; align-items:center; justify-content:center;
               font-size:2rem; font-weight:800; color:#FF6B35;
               font-family:'Courier New',monospace; }
  .alert-box { background:rgba(255,107,53,.08); border:1px solid rgba(255,107,53,.25);
               border-left:4px solid #FF6B35; border-radius:10px;
               padding:16px 20px; margin:20px 0; color:#FF6B35;
               font-size:.88rem; line-height:1.5; }
  .success-box { background:rgba(0,212,170,.08); border:1px solid rgba(0,212,170,.25);
                 border-left:4px solid #00D4AA; border-radius:10px;
                 padding:16px 20px; margin:20px 0; color:#00D4AA;
                 font-size:.88rem; }
  .info-row { display:flex; gap:12px; padding:10px 0;
              border-bottom:1px solid rgba(255,255,255,.05); }
  .info-label { color:#6B7A99; font-size:.82rem; min-width:120px; }
  .info-value { color:#E8F0FE; font-size:.82rem; font-weight:600; }
  .footer { background:#060B18; padding:20px 32px; text-align:center;
            font-size:.75rem; color:#444D5E; border-top:1px solid rgba(255,107,53,.1); }
  .badge { display:inline-block; padding:3px 10px; border-radius:40px;
           font-size:.7rem; font-weight:700; letter-spacing:.08em;
           text-transform:uppercase; }
  .badge-critical { background:rgba(255,59,92,.1); color:#FF3B5C;
                    border:1px solid rgba(255,59,92,.3); }
  .badge-high { background:rgba(255,149,0,.1); color:#FF9500;
                border:1px solid rgba(255,149,0,.3); }
  .badge-medium { background:rgba(0,122,255,.1); color:#4A9EFF;
                  border:1px solid rgba(0,122,255,.3); }
  .badge-low { background:rgba(0,212,170,.1); color:#00D4AA;
               border:1px solid rgba(0,212,170,.3); }
</style>
"""

_FOOTER_HTML = f"""
<div class="footer">
  <strong style="color:#FF6B35;">{_BRAND} {_VERSION}</strong> &nbsp;|&nbsp;
  {_COLLEGE}<br>
  This is an automated message. Do not reply to this email.
</div>
"""


def _make_message(to: str, subject: str, html: str, text: str) -> MIMEMultipart:
    """Build a MIME multipart email with HTML + plaintext fallback."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = settings.SMTP_FROM or f"{_BRAND} <{_SUPPORT_EMAIL}>"
    msg["To"] = to
    msg.attach(MIMEText(text, "plain"))
    msg.attach(MIMEText(html, "html"))
    return msg


def _send(msg: MIMEMultipart) -> bool:
    """Send email via Gmail STARTTLS. Returns True on success."""
    if not all([settings.SMTP_HOST, settings.SMTP_USER, settings.SMTP_PASSWORD]):
        logger.warning("SMTP not configured — email not sent.")
        return False
    try:
        # LOGIC-4 fix: socket timeout prevents indefinite hang
        socket.setdefaulttimeout(10)
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.sendmail(msg["From"], msg["To"], msg.as_string())
        logger.info(f"Email sent to {msg['To'][:6]}***")
        return True
    except smtplib.SMTPException as exc:
        logger.error(f"SMTPException sending to {msg['To'][:6]}***: {exc}")
        return False
    except OSError as exc:
        logger.error(f"Network error sending email: {exc}")
        return False
    finally:
        socket.setdefaulttimeout(None)


# ── OTP Email ─────────────────────────────────────────────────────────────────

def send_otp_email(to: str, otp: str) -> bool:
    """
    Send 6-digit OTP email for password reset.
    OTP is displayed in a large digit-box layout.
    SECURITY: otp is displayed in email but NEVER logged here.
    """
    digits = "".join(
        f'<div class="otp-digit">{d}</div>' for d in otp
    )
    html = f"""<!DOCTYPE html><html><head>{_BASE_STYLE}</head><body>
<div class="wrapper">
  <div class="header">
    <h1>🔐 {_BRAND} Password Reset</h1>
    <span>{_VERSION} Security Service</span>
  </div>
  <div class="body">
    <p>You requested a password reset for your CostGuard account.
    Use the OTP below to proceed. <strong>This code expires in 10 minutes.</strong></p>
    <div class="otp-box">{digits}</div>
    <div class="alert-box">
      ⏱️ &nbsp;This OTP is valid for <strong>10 minutes only</strong>.<br>
      🔒 &nbsp;Never share this code with anyone.<br>
      🚫 &nbsp;If you did not request a password reset, you can safely ignore this email.
    </div>
    <p style="font-size:.82rem; color:#6B7A99;">
      For security, we do not store your OTP — it is one-time use only.
      After 5 incorrect attempts, the code will be automatically invalidated.
    </p>
  </div>
  {_FOOTER_HTML}
</div>
</body></html>"""

    text = (
        f"CostGuard Password Reset OTP\n\n"
        f"Your OTP is: {otp}\n\n"
        f"This code expires in 10 minutes.\n"
        f"If you did not request this, ignore this email.\n\n"
        f"-- {_BRAND} {_VERSION}"
    )
    subject = f"🔐 {_BRAND} Password Reset OTP (expires in 10 min)"
    return _send(_make_message(to, subject, html, text))


# ── Password Reset Confirmation Email ─────────────────────────────────────────

def send_password_reset_confirmation(to: str, ip: str = "unknown") -> bool:
    """Send confirmation email after a successful password reset."""
    reset_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!DOCTYPE html><html><head>{_BASE_STYLE}</head><body>
<div class="wrapper">
  <div class="header">
    <h1>✅ Password Reset Successful</h1>
    <span>{_BRAND} {_VERSION}</span>
  </div>
  <div class="body">
    <div class="success-box">
      ✅ Your password has been reset successfully.
    </div>
    <p>Here are the details of this security event:</p>
    <div class="info-row">
      <span class="info-label">Time</span>
      <span class="info-value">{reset_time}</span>
    </div>
    <div class="info-row">
      <span class="info-label">IP Address</span>
      <span class="info-value">{ip}</span>
    </div>
    <div class="info-row">
      <span class="info-label">Account</span>
      <span class="info-value">{to}</span>
    </div>
    <div class="alert-box" style="border-left-color:#FF3B5C; color:#FF3B5C;
         background:rgba(255,59,92,.06); border-color:rgba(255,59,92,.25);">
      ⚠️ &nbsp;If you did not perform this reset, please contact support immediately
      at <strong>{_SUPPORT_EMAIL}</strong> and change your password right away.
    </div>
  </div>
  {_FOOTER_HTML}
</div>
</body></html>"""

    text = (
        f"CostGuard — Password Reset Successful\n\n"
        f"Your password was reset on {reset_time} from IP {ip}.\n"
        f"If this wasn't you, contact support immediately: {_SUPPORT_EMAIL}\n\n"
        f"-- {_BRAND} {_VERSION}"
    )
    subject = f"✅ {_BRAND} — Password Reset Successful"
    return _send(_make_message(to, subject, html, text))


# ── Support Enquiry Acknowledgement ───────────────────────────────────────────

def send_enquiry_ack(to: str, enquiry_id: int, subject: str) -> bool:
    """Send auto-acknowledgement email to the user after enquiry submission."""
    html = f"""<!DOCTYPE html><html><head>{_BASE_STYLE}</head><body>
<div class="wrapper">
  <div class="header">
    <h1>📨 Enquiry Received</h1>
    <span>Reference #{enquiry_id:05d}</span>
  </div>
  <div class="body">
    <div class="success-box">
      ✅ We've received your enquiry and will respond within <strong>2 hours</strong>.
    </div>
    <p>Thank you for reaching out to {_BRAND} Support.</p>
    <div class="info-row">
      <span class="info-label">Reference ID</span>
      <span class="info-value">#{enquiry_id:05d}</span>
    </div>
    <div class="info-row">
      <span class="info-label">Subject</span>
      <span class="info-value">{subject}</span>
    </div>
    <div class="info-row">
      <span class="info-label">Submitted</span>
      <span class="info-value">{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}</span>
    </div>
    <p style="margin-top:20px; font-size:.88rem;">
      Keep this reference number for follow-up. You can track the status of
      your enquiry in the <strong>Support</strong> section of your dashboard.
    </p>
  </div>
  {_FOOTER_HTML}
</div>
</body></html>"""

    text = (
        f"CostGuard Support — Enquiry #{enquiry_id:05d} Received\n\n"
        f"Subject: {subject}\n"
        f"We'll respond within 2 hours.\n\n"
        f"-- {_BRAND} {_VERSION}"
    )
    return _send(_make_message(to, f"📨 Enquiry #{enquiry_id:05d} Received — {_BRAND}", html, text))


# ── Admin Notification for New Enquiry ────────────────────────────────────────

def send_admin_notification(enquiry: Dict[str, Any]) -> bool:
    """Notify admin (SMTP_USER) of a new support enquiry."""
    admin_email = settings.SMTP_USER
    if not admin_email:
        return False

    priority = enquiry.get("priority", "medium").lower()
    badge_cls = {
        "critical": "badge-critical", "high": "badge-high",
        "medium": "badge-medium", "low": "badge-low",
    }.get(priority, "badge-medium")

    html = f"""<!DOCTYPE html><html><head>{_BASE_STYLE}</head><body>
<div class="wrapper">
  <div class="header">
    <h1>🔔 New Support Enquiry</h1>
    <span>Admin Notification</span>
  </div>
  <div class="body">
    <p>A new support enquiry has been submitted:</p>
    <div class="info-row">
      <span class="info-label">Enquiry ID</span>
      <span class="info-value">#{enquiry.get('id',0):05d}</span>
    </div>
    <div class="info-row">
      <span class="info-label">From</span>
      <span class="info-value">{enquiry.get('name','')} &lt;{enquiry.get('email','')}&gt;</span>
    </div>
    <div class="info-row">
      <span class="info-label">Subject</span>
      <span class="info-value">{enquiry.get('subject','')}</span>
    </div>
    <div class="info-row">
      <span class="info-label">Category</span>
      <span class="info-value">{enquiry.get('category','').title()}</span>
    </div>
    <div class="info-row">
      <span class="info-label">Priority</span>
      <span class="info-value">
        <span class="badge {badge_cls}">{priority.upper()}</span>
      </span>
    </div>
    <p style="margin-top:20px; font-size:.85rem; color:#6B7A99;">
      Log into the CostGuard dashboard → Support to manage this enquiry.
    </p>
  </div>
  {_FOOTER_HTML}
</div>
</body></html>"""

    text = (
        f"New CostGuard Enquiry #{enquiry.get('id',0):05d}\n"
        f"From: {enquiry.get('name','')} <{enquiry.get('email','')}>\n"
        f"Subject: {enquiry.get('subject','')}\n"
        f"Priority: {priority.upper()}\n"
    )
    return _send(_make_message(
        admin_email,
        f"🔔 New [{priority.upper()}] Enquiry #{enquiry.get('id',0):05d} — {_BRAND}",
        html, text,
    ))


# ── Team Invite Email ─────────────────────────────────────────────────────────

def send_team_invite(to: str, inviter_name: str, temp_password: str, role: str) -> bool:
    """Send team invitation with a temporary password."""
    html = f"""<!DOCTYPE html><html><head>{_BASE_STYLE}</head><body>
<div class="wrapper">
  <div class="header">
    <h1>👋 You're Invited to {_BRAND}</h1>
    <span>Team Invitation</span>
  </div>
  <div class="body">
    <p><strong>{inviter_name}</strong> has invited you to join the {_BRAND}
    FinOps platform as a <strong>{role.title()}</strong>.</p>
    <div class="info-row">
      <span class="info-label">Your Email</span>
      <span class="info-value">{to}</span>
    </div>
    <div class="info-row">
      <span class="info-label">Temp Password</span>
      <span class="info-value" style="font-family:monospace; color:#FF6B35;">{temp_password}</span>
    </div>
    <div class="info-row">
      <span class="info-label">Role</span>
      <span class="info-value">{role.title()}</span>
    </div>
    <div class="alert-box">
      🔐 &nbsp;Please change your password immediately after first login.
    </div>
  </div>
  {_FOOTER_HTML}
</div>
</body></html>"""

    text = (
        f"You've been invited to {_BRAND}.\n"
        f"Email: {to}\nTemporary password: {temp_password}\nRole: {role.title()}\n"
        f"Please change your password on first login.\n"
    )
    return _send(_make_message(to, f"👋 You're invited to {_BRAND}", html, text))

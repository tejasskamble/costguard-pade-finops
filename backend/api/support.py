"""
backend/api/support.py - CostGuard v17.0

Support and enquiry endpoints for the active enterprise platform.
"""
import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel
from datetime import datetime

from api.auth import get_current_user_optional, UserProfile
from database import get_db_conn
from runtime_hardening import safe_create_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/support", tags=["support"])

UPLOAD_DIR = Path(__file__).parent.parent / "uploads" / "enquiries"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024  # 5 MB
VALID_PRIORITIES = {"low", "medium", "high", "critical"}
VALID_CATEGORIES = {"billing", "technical", "feature_request", "bug_report", "other"}


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp), str(path))
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


# ─── Pydantic models ──────────────────────────────────────────────────────────

class EnquiryResponse(BaseModel):
    id: int
    status: str
    message: str
    created_at: datetime

class EnquiryListItem(BaseModel):
    id: int
    subject: str
    category: str
    priority: str
    status: str
    created_at: datetime


# ─── FAQ data ─────────────────────────────────────────────────────────────────

FAQ_LIST = [
    {
        "q": "How does CostGuard detect anomalies?",
        "a": "CostGuard uses a Graph Attention Network (GAT) + LSTM ensemble — "
             "the PADE engine — to detect cost anomalies in your CI/CD pipeline runs. "
             "It assigns a Cost Risk Score (CRS) from 0–1 and automatically classifies "
             "each run as ALLOW / WARN / AUTO_OPTIMISE / BLOCK.",
    },
    {
        "q": "What cloud providers are supported?",
        "a": "CostGuard supports AWS, GCP, Azure, and self-hosted infrastructure. "
             "Multi-cloud cost attribution is available in the Enterprise edition.",
    },
    {
        "q": "How do I connect my cloud billing data?",
        "a": "Use the Pipeline Ingest API (/api/ingest) to POST pipeline run data. "
             "CostGuard accepts FOCUS-schema compatible JSON payloads.",
    },
    {
        "q": "Can I export cost reports?",
        "a": "Yes. Navigate to Pipeline Costs or Run History pages and click 'Export CSV'. "
             "PDF export is available in the Forecasting Studio page.",
    },
    {
        "q": "How do I reset my password?",
        "a": "Click 'Forgot Password?' on the login page. Enter your email address, "
             "and you'll receive a 6-digit OTP valid for 10 minutes. "
             "Enter the OTP to set a new password.",
    },
    {
        "q": "What is the CRS (Cost Risk Score)?",
        "a": "CRS is a 0–1 score produced by the PADE ensemble model. "
             "A score below 0.3 is ALLOW (safe), 0.3–0.5 is WARN, "
             "0.5–0.75 triggers AUTO_OPTIMISE, and above 0.75 is BLOCK.",
    },
    {
        "q": "How do I invite team members?",
        "a": "Go to Team & Users (Page 07 in the sidebar). Click 'Invite Team Member', "
             "enter their email and select a role. They'll receive a temporary password via email.",
    },
    {
        "q": "Can I set budget alerts?",
        "a": "Yes. Go to Notifications & Alerts Config and set a budget threshold. "
             "You'll receive Slack and/or email alerts when pipeline costs exceed your limit.",
    },
    {
        "q": "What are the user roles?",
        "a": "CostGuard has three roles: Admin (full access + user management), "
             "Analyst (read/write access to all analytics), and Viewer (read-only).",
    },
    {
        "q": "How do I contact support?",
        "a": "Use this enquiry form! Fill in your issue details and we'll respond "
             "within 2 business hours. For critical issues, select 'Critical' priority "
             "for immediate escalation.",
    },
]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/enquiry", response_model=EnquiryResponse, status_code=201)
async def create_enquiry(
    name: str = Form(...),
    email: str = Form(...),
    phone: Optional[str] = Form(None),
    subject: str = Form(...),
    category: str = Form(...),
    priority: str = Form("medium"),
    message: str = Form(...),
    attachment: Optional[UploadFile] = File(None),
    request: Request = None,
    db=Depends(get_db_conn),
    current_user: Optional[UserProfile] = Depends(get_current_user_optional),
):
    """
    Create a support enquiry. Authenticated users get their user_id linked.
    Sends acknowledgment email to user + notification to admin.
    Attachment (image only, ≤5MB) stored in backend/uploads/enquiries/.
    """
    # Validate inputs
    priority = priority.lower()
    category = category.lower()

    if priority not in VALID_PRIORITIES:
        raise HTTPException(400, f"priority must be one of: {VALID_PRIORITIES}")
    if not name.strip():
        raise HTTPException(400, "name is required")
    if not email.strip() or "@" not in email:
        raise HTTPException(400, "valid email is required")
    if not subject.strip():
        raise HTTPException(400, "subject is required")
    if len(message.strip()) < 20:
        raise HTTPException(400, "message must be at least 20 characters")

    # Handle file attachment
    attachment_path: Optional[str] = None
    if attachment and attachment.filename:
        content = await attachment.read()
        if len(content) > MAX_ATTACHMENT_BYTES:
            raise HTTPException(400, "Attachment must be under 5MB")
        if attachment.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(400, "Only PNG, JPEG, WEBP images are accepted")
        safe_name = f"{uuid.uuid4()}_{attachment.filename}"
        dest = UPLOAD_DIR / safe_name
        _atomic_write_bytes(dest, content)
        attachment_path = str(dest)

    # Persist to DB
    try:
        row = await db.fetchrow(
            """INSERT INTO support_enquiries
                   (user_id, name, email, phone, subject, category, priority, message, attachment_path)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
               RETURNING id, status, created_at""",
            getattr(current_user, "id", None),
            name.strip(), email.strip(), phone or None,
            subject.strip(), category, priority,
            message.strip(), attachment_path,
        )
    except Exception as exc:
        logger.exception(f"create_enquiry DB error: {exc}")
        raise HTTPException(500, "Failed to save enquiry")

    enquiry_id = row["id"]

    # Fire-and-forget emails
    from services import email_service
    safe_create_task(
        asyncio.to_thread(email_service.send_enquiry_ack, email, enquiry_id, subject),
        logger=logger,
        label="support enquiry acknowledgement email",
    )
    safe_create_task(
        asyncio.to_thread(email_service.send_admin_notification, {
            "id": enquiry_id, "name": name, "email": email,
            "subject": subject, "priority": priority, "category": category,
        }),
        logger=logger,
        label="support enquiry admin notification",
    )

    return EnquiryResponse(
        id=enquiry_id,
        status="open",
        message="Your enquiry has been received. We'll respond within 2 hours.",
        created_at=row["created_at"],
    )


@router.get("/enquiries", response_model=List[EnquiryListItem])
async def list_my_enquiries(
    request: Request,
    db=Depends(get_db_conn),
    current_user: Optional[UserProfile] = Depends(get_current_user_optional),
    page: int = 1,
    limit: int = 20,
):
    """List all enquiries for the authenticated user, newest first."""
    if not current_user:
        raise HTTPException(401, "Authentication required")
    limit = min(limit, 100)
    offset = (page - 1) * limit
    try:
        rows = await db.fetch(
            """SELECT id, subject, category, priority, status, created_at
               FROM support_enquiries
               WHERE user_id = $1
               ORDER BY created_at DESC
               LIMIT $2 OFFSET $3""",
            current_user.id, limit, offset,
        )
        return [EnquiryListItem(**dict(r)) for r in rows]
    except Exception as exc:
        logger.exception(f"list_my_enquiries error: {exc}")
        raise HTTPException(500, "Failed to fetch enquiries")


@router.get("/enquiries/{enquiry_id}")
async def get_enquiry_detail(
    enquiry_id: int,
    request: Request,
    db=Depends(get_db_conn),
    current_user: Optional[UserProfile] = Depends(get_current_user_optional),
):
    """Get a single enquiry detail. Users can only see their own enquiries."""
    if not current_user:
        raise HTTPException(401, "Authentication required")
    try:
        row = await db.fetchrow(
            """SELECT id, name, email, phone, subject, category, priority,
                      message, attachment_path, status, admin_notes, created_at, updated_at
               FROM support_enquiries
               WHERE id = $1 AND (user_id = $2 OR $3 = 'admin')""",
            enquiry_id, current_user.id, current_user.role,
        )
        if not row:
            raise HTTPException(404, "Enquiry not found")
        return dict(row)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"get_enquiry_detail error: {exc}")
        raise HTTPException(500, "Failed to fetch enquiry")


@router.get("/faq")
async def get_faq():
    """Return the FAQ list."""
    return FAQ_LIST

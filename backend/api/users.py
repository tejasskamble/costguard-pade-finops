"""
backend/api/users.py - CostGuard v17.0

User and team management endpoints for CostGuard.
"""
import asyncio
import hashlib
import logging
import secrets
import string
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr

from api.auth import get_current_user, UserProfile, pwd_context, _validate_password_strength
from database import get_db_conn
from runtime_hardening import safe_create_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/users", tags=["users"])


# ─── Pydantic models ──────────────────────────────────────────────────────────

class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None

class InviteRequest(BaseModel):
    email: EmailStr
    full_name: str
    role: str = "viewer"

class RoleUpdateRequest(BaseModel):
    role: str

class UserListItem(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    is_active: bool
    last_login_at: Optional[datetime]
    created_at: datetime
    gravatar_url: str


def _gravatar(email: str, size: int = 40) -> str:
    """Return Gravatar URL for the given email (identicon fallback)."""
    h = hashlib.md5(email.lower().strip().encode()).hexdigest()
    return f"https://www.gravatar.com/avatar/{h}?d=identicon&s={size}"


def _gen_temp_password(length: int = 12) -> str:
    """Generate a random temporary password meeting strength requirements."""
    chars = string.ascii_letters + string.digits + "!@#$"
    while True:
        pw = "".join(secrets.choice(chars) for _ in range(length))
        if _validate_password_strength(pw) is None:
            return pw


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/me")
async def get_my_profile(
    request: Request,
    db=Depends(get_db_conn),
    current_user: UserProfile = Depends(get_current_user),
):
    """Return extended profile: role, avatar, last_login, activity count."""
    try:
        row = await db.fetchrow(
            """SELECT id, email, full_name, role, avatar_url, phone,
                      last_login_at, is_active, created_at
               FROM users WHERE id = $1""",
            current_user.id,
        )
        if not row:
            raise HTTPException(404, "User not found")
        activity_count = await db.fetchval(
            "SELECT COUNT(*) FROM user_activity_log WHERE user_id = $1", current_user.id
        ) or 0
        data = dict(row)
        data["gravatar_url"] = _gravatar(row["email"])
        data["activity_count"] = int(activity_count)
        return data
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"get_my_profile error: {exc}")
        raise HTTPException(500, "Failed to fetch profile")


@router.put("/me")
async def update_my_profile(
    body: ProfileUpdate,
    request: Request,
    db=Depends(get_db_conn),
    current_user: UserProfile = Depends(get_current_user),
):
    """Update the authenticated user's name and/or phone."""
    updates = {}
    if body.full_name is not None:
        updates["full_name"] = body.full_name.strip()
    if body.phone is not None:
        updates["phone"] = body.phone.strip()

    if not updates:
        raise HTTPException(400, "No fields to update")

    try:
        if "full_name" in updates and "phone" in updates:
            await db.execute(
                "UPDATE users SET full_name = $1, phone = $2 WHERE id = $3",
                updates["full_name"],
                updates["phone"],
                current_user.id,
            )
        elif "full_name" in updates:
            await db.execute(
                "UPDATE users SET full_name = $1 WHERE id = $2",
                updates["full_name"],
                current_user.id,
            )
        else:
            await db.execute(
                "UPDATE users SET phone = $1 WHERE id = $2",
                updates["phone"],
                current_user.id,
            )
        return {"message": "Profile updated successfully"}
    except Exception as exc:
        logger.exception(f"update_my_profile error: {exc}")
        raise HTTPException(500, "Failed to update profile")


@router.post("/invite")
async def invite_team_member(
    body: InviteRequest,
    request: Request,
    db=Depends(get_db_conn),
    current_user: UserProfile = Depends(get_current_user),
):
    """Admin-only: invite a new team member with a temporary password."""
    if current_user.role != "admin":
        raise HTTPException(403, "Only admins can invite team members")

    valid_roles = {"admin", "analyst", "viewer"}
    if body.role not in valid_roles:
        raise HTTPException(400, f"role must be one of: {valid_roles}")

    try:
        existing = await db.fetchrow(
            "SELECT id FROM users WHERE email = $1", body.email
        )
        if existing:
            raise HTTPException(400, "User with this email already exists")

        temp_password = _gen_temp_password()
        pw_hash = pwd_context.hash(temp_password)

        await db.execute(
            """INSERT INTO users (email, full_name, password_hash, role)
               VALUES ($1, $2, $3, $4)""",
            body.email, body.full_name, pw_hash, body.role,
        )

        from services import email_service
        safe_create_task(
            asyncio.to_thread(
                email_service.send_team_invite,
                body.email, current_user.full_name, temp_password, body.role,
            ),
            logger=logger,
            label="team invitation email",
        )

        return {"message": f"Invitation sent to {body.email}", "role": body.role}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"invite_team_member error: {exc}")
        raise HTTPException(500, "Failed to send invitation")


@router.get("/list", response_model=List[UserListItem])
async def list_users(
    request: Request,
    db=Depends(get_db_conn),
    current_user: UserProfile = Depends(get_current_user),
    page: int = 1,
    limit: int = 20,
):
    """Admin-only: list all users with roles and activity."""
    if current_user.role != "admin":
        raise HTTPException(403, "Admin access required")
    limit = min(limit, 100)
    offset = (page - 1) * limit
    try:
        rows = await db.fetch(
            """SELECT id, email, full_name, role, is_active, last_login_at, created_at
               FROM users ORDER BY created_at DESC LIMIT $1 OFFSET $2""",
            limit, offset,
        )
        return [
            UserListItem(
                **{k: v for k, v in dict(r).items() if k != "gravatar_url"},
                gravatar_url=_gravatar(r["email"]),
            )
            for r in rows
        ]
    except Exception as exc:
        logger.exception(f"list_users error: {exc}")
        raise HTTPException(500, "Failed to list users")


@router.put("/{user_id}/role")
async def update_user_role(
    user_id: int,
    body: RoleUpdateRequest,
    request: Request,
    db=Depends(get_db_conn),
    current_user: UserProfile = Depends(get_current_user),
):
    """Admin-only: update a user's role."""
    if current_user.role != "admin":
        raise HTTPException(403, "Admin access required")
    valid_roles = {"admin", "analyst", "viewer"}
    if body.role not in valid_roles:
        raise HTTPException(400, f"role must be one of: {valid_roles}")
    try:
        result = await db.execute(
            "UPDATE users SET role = $1 WHERE id = $2", body.role, user_id
        )
        if result == "UPDATE 0":
            raise HTTPException(404, "User not found")
        return {"message": f"Role updated to {body.role}"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"update_user_role error: {exc}")
        raise HTTPException(500, "Failed to update role")


@router.delete("/{user_id}")
async def deactivate_user(
    user_id: int,
    request: Request,
    db=Depends(get_db_conn),
    current_user: UserProfile = Depends(get_current_user),
):
    """Admin or self: soft-delete a user account (is_active=False)."""
    if current_user.role != "admin" and current_user.id != user_id:
        raise HTTPException(403, "You can only delete your own account")
    if current_user.id == user_id and current_user.role == "admin":
        # Prevent admin from accidentally locking themselves out
        admin_count = await db.fetchval(
            "SELECT COUNT(*) FROM users WHERE role='admin' AND is_active=TRUE"
        ) or 0
        if int(admin_count) <= 1:
            raise HTTPException(400, "Cannot deactivate the last admin account")
    try:
        await db.execute(
            "UPDATE users SET is_active = FALSE WHERE id = $1", user_id
        )
        return {"message": "Account deactivated"}
    except Exception as exc:
        logger.exception(f"deactivate_user error: {exc}")
        raise HTTPException(500, "Failed to deactivate account")


@router.get("/activity")
async def get_activity_log(
    request: Request,
    db=Depends(get_db_conn),
    current_user: UserProfile = Depends(get_current_user),
    limit: int = 50,
):
    """Return the current user's recent activity log."""
    limit = min(limit, 200)
    try:
        rows = await db.fetch(
            """SELECT id, action, page, metadata, ip_address, created_at
               FROM user_activity_log
               WHERE user_id = $1
               ORDER BY created_at DESC LIMIT $2""",
            current_user.id, limit,
        )
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.exception(f"get_activity_log error: {exc}")
        raise HTTPException(500, "Failed to fetch activity log")


@router.post("/activity")
async def log_activity(
    request: Request,
    db=Depends(get_db_conn),
    current_user: UserProfile = Depends(get_current_user),
):
    """Log a page visit or action for the current user."""
    try:
        import json as _json
        body = await request.json()
        # FIX: str() on a dict produces Python syntax e.g. "{'k': 'v'}" which
        # fails the ::jsonb cast. json.dumps() produces valid JSON.
        metadata_raw = body.get("metadata")
        metadata_json = _json.dumps(metadata_raw) if metadata_raw else "{}"
        # FIX: behind a reverse proxy, client.host may be "10.0.0.1, 172.31.0.1"
        # which fails PostgreSQL's ::inet cast.  Take only the first address.
        raw_ip = request.client.host if request.client else None
        ip_addr = raw_ip.split(",")[0].strip() if raw_ip else None
        await db.execute(
            """INSERT INTO user_activity_log (user_id, action, page, metadata, ip_address)
               VALUES ($1, $2, $3, $4::jsonb, $5::inet)""",
            current_user.id,
            body.get("action", "page_visit"),
            body.get("page", ""),
            metadata_json,
            ip_addr,
        )
        return {"logged": True}
    except Exception as exc:
        logger.warning("log_activity failed: %s", exc)
        raise HTTPException(status_code=503, detail="Failed to log activity.")

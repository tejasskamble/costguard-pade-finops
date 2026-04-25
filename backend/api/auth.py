"""
backend/api/auth.py  — CostGuard v17.0
Real auth: bcrypt hashing, email-based JWT (sub=email), DB-backed users.
Active additions: forgot-password OTP flow (B1-B5), get_current_user_optional,
password-strength validation, activity logging on login.
"""
import asyncio
import logging
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field

from config import settings
from runtime_hardening import safe_create_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")
oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="/api/auth/token", auto_error=False)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ─── Pydantic models ──────────────────────────────────────────────────────────

class Token(BaseModel):
    access_token: str
    token_type: str

class UserCreate(BaseModel):
    email: str
    password: str
    full_name: str

class UserProfile(BaseModel):
    id: Optional[int] = None
    email: str
    full_name: str
    role: str = "viewer"
    created_at: datetime

class MessageResponse(BaseModel):
    message: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class VerifyOTPRequest(BaseModel):
    email: EmailStr
    otp: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")

class ResetPasswordRequest(BaseModel):
    new_password: str = Field(min_length=8)
    confirm_password: str


# ─── Password strength ────────────────────────────────────────────────────────

def _validate_password_strength(password: str) -> Optional[str]:
    """Return error message if password fails requirements, else None."""
    if len(password) < 8:
        return "Password must be at least 8 characters"
    if not re.search(r"[A-Z]", password):
        return "Password must contain at least one uppercase letter"
    if not re.search(r"\d", password):
        return "Password must contain at least one digit"
    return None


# ─── JWT helpers ──────────────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta if expires_delta
        else timedelta(minutes=settings.JWT_EXPIRY_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def create_reset_token(email: str) -> str:
    """Create 15-min password-reset JWT with purpose='password_reset' claim."""
    return create_access_token(
        {"sub": email, "purpose": "password_reset"},
        expires_delta=timedelta(minutes=settings.RESET_TOKEN_EXPIRY_MINUTES),
    )


async def get_current_user(
    request: Request,
    token: str = Depends(oauth2_scheme),
) -> UserProfile:
    """
    Decode JWT and fetch user from DB.
    CONSTRAINT-2: JWT sub is always the user's email.
    """
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM]
        )
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exc
    except JWTError:
        raise credentials_exc

    pool = getattr(request.app.state, "db", None)
    if pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is unavailable while the database is degraded.",
        )

    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, email, full_name, created_at, role FROM users WHERE email = $1 AND is_active = TRUE",
                email,
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("User lookup failed during authentication: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is temporarily unavailable.",
        )
    if row is None:
        raise credentials_exc
    return UserProfile(
        id=row["id"],
        email=row["email"],
        full_name=row["full_name"],
        role=row["role"] if row["role"] else "viewer",
        created_at=row["created_at"],
    )


async def get_current_user_optional(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme_optional),
) -> Optional[UserProfile]:
    """Returns None instead of 401 when token is absent/invalid."""
    if not token:
        return None
    try:
        return await get_current_user(request, token)
    except HTTPException as exc:
        if exc.status_code == status.HTTP_401_UNAUTHORIZED:
            return None
        raise


async def get_current_user_id(
    request: Request,
    token: str = Depends(oauth2_scheme),
) -> int:
    """Return DB user id for the current JWT token."""
    user = await get_current_user(request, token)
    if user.id is None:
        raise HTTPException(status_code=401, detail="Could not resolve user id")
    return user.id


async def require_admin(
    current_user: UserProfile = Depends(get_current_user),
) -> UserProfile:
    """Require an authenticated admin user."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return current_user


def _get_auth_pool(request: Request):
    pool = getattr(request.app.state, "db", None)
    if pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is unavailable while the database is degraded.",
        )
    return pool


# ─── Welcome email ────────────────────────────────────────────────────────────

async def send_welcome_email(recipient_email: str, full_name: str) -> None:
    """Fire-and-forget welcome email — never raises."""
    if not all([settings.SMTP_HOST, settings.SMTP_USER, settings.SMTP_PASSWORD]):
        return
    try:
        import aiosmtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        html = f"""<html><body style="background:#060B18;font-family:sans-serif;padding:32px;">
        <div style="max-width:520px;margin:0 auto;background:#0D1B2E;border-radius:16px;
                    border:1px solid rgba(255,107,53,.2);padding:40px;color:#E8F0FE;">
          <h2 style="background:linear-gradient(135deg,#FF6B35,#C084FC);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            🛡️ Welcome to CostGuard v17.0</h2>
          <p style="color:#B0BDD0;">Hi <strong>{full_name}</strong>, your account is ready!</p>
          <p style="color:#6B7A99;font-size:.85rem;">
            Sir Parshurambhau College (Autonomous), Pune</p>
        </div></body></html>"""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Welcome to CostGuard, {full_name}!"
        msg["From"] = settings.SMTP_FROM or settings.SMTP_USER
        msg["To"] = recipient_email
        msg.attach(MIMEText(html, "html"))
        await aiosmtplib.send(
            msg, hostname=settings.SMTP_HOST, port=settings.SMTP_PORT,
            username=settings.SMTP_USER, password=settings.SMTP_PASSWORD,
            start_tls=True, timeout=10,
        )
    except Exception as exc:
        logger.warning(f"Welcome email failed (non-fatal): {exc}")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/register", response_model=Token)
async def register(user: UserCreate, request: Request):
    """Create a new user account. Returns JWT. Sub = email."""
    pool = _get_auth_pool(request)
    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT id FROM users WHERE email = $1", user.email
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="An account with this email already exists.",
            )
        err = _validate_password_strength(user.password)
        if err:
            raise HTTPException(status_code=400, detail=err)
        pw_hash = pwd_context.hash(user.password)
        await conn.execute(
            "INSERT INTO users (email, full_name, password_hash) VALUES ($1, $2, $3)",
            user.email, user.full_name, pw_hash,
        )

    if all([settings.SMTP_HOST, settings.SMTP_USER, settings.SMTP_PASSWORD]):
        safe_create_task(
            send_welcome_email(user.email, user.full_name),
            logger=logger,
            label="welcome email delivery",
        )
    token = create_access_token({"sub": user.email})
    logger.info(f"New user registered: {user.email}")
    return {"access_token": token, "token_type": "bearer"}


@router.post("/token", response_model=Token)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    """OAuth2 token endpoint. Accepts email in 'username' field."""
    pool = _get_auth_pool(request)
    email = form_data.username

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, email, password_hash FROM users WHERE email = $1 AND is_active = TRUE",
            email,
        )

    if row is None or not pwd_context.verify(form_data.password, row["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET last_login_at = NOW() WHERE id = $1", row["id"]
        )

    token = create_access_token({"sub": row["email"]})
    logger.info(f"User logged in: {email}")
    return {"access_token": token, "token_type": "bearer"}


@router.get("/me", response_model=UserProfile)
async def get_me(current_user: UserProfile = Depends(get_current_user)):
    """Return the current user's profile."""
    return current_user


# ── Forgot-Password OTP Flow ────────────────────────────────────────────

@router.post("/forgot-password", response_model=MessageResponse)
async def forgot_password(
    body: ForgotPasswordRequest,
    request: Request,
) -> MessageResponse:
    """
    Send 6-digit OTP to registered email for password reset.
    Identical response whether email exists or not (prevents enumeration).
    Rate limit: 3 requests per email per hour.
    """
    neutral = MessageResponse(message="If that email exists, an OTP has been sent.")
    pool = _get_auth_pool(request)
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, email FROM users WHERE email = $1 AND is_active = TRUE",
                body.email,
            )
            if not row:
                return neutral

            # Rate limit: 3 OTPs per email per hour
            recent_count = await conn.fetchval(
                """SELECT COUNT(*) FROM password_reset_otps
                   WHERE email = $1 AND created_at > NOW() - INTERVAL '1 hour'""",
                body.email,
            )
            if (recent_count or 0) >= settings.OTP_RATE_LIMIT_PER_HOUR:
                raise HTTPException(
                    status_code=429,
                    detail="Too many OTP requests. Please wait 1 hour.",
                )

            # Generate cryptographically secure 6-digit OTP
            otp = str(secrets.randbelow(900000) + 100000)
            # Hash before storing — raw OTP is never persisted
            otp_hash = pwd_context.hash(otp)

            await conn.execute(
                """INSERT INTO password_reset_otps (email, otp_hash, expires_at)
                   VALUES ($1, $2, NOW() + INTERVAL '10 minutes')""",
                body.email, otp_hash,
            )

        # Fire-and-forget email in thread pool (SMTP is blocking I/O)
        from services import email_service
        safe_create_task(
            asyncio.to_thread(email_service.send_otp_email, body.email, otp),
            logger=logger,
            label="password reset OTP email",
        )

        logger.info(f"OTP issued for {body.email[:3]}***")
        return neutral

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"forgot_password error: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/verify-otp", response_model=dict)
async def verify_otp(
    body: VerifyOTPRequest,
    request: Request,
) -> dict:
    """
    Verify 6-digit OTP. Returns short-lived reset_token on success.
    5 failed attempts invalidates the OTP.
    """
    pool = _get_auth_pool(request)
    try:
        async with pool.acquire() as conn:
            # FIX 3B: read fail_count directly from OTP row (activity_log INSERT
            # was missing user_id so COUNT(*) always returned 0 — brute-force
            # protection was completely bypassed)
            row = await conn.fetchrow(
                """SELECT id, otp_hash, fail_count FROM password_reset_otps
                   WHERE email = $1 AND used = FALSE AND expires_at > NOW()
                   ORDER BY created_at DESC LIMIT 1""",
                body.email,
            )
            if not row:
                raise HTTPException(
                    status_code=400,
                    detail="OTP is invalid, expired, or already used. Please request a new one.",
                )

            # Check failure count — lock out BEFORE attempting verify
            if row["fail_count"] >= settings.OTP_MAX_ATTEMPTS:
                await conn.execute(
                    "UPDATE password_reset_otps SET used = TRUE WHERE id = $1", row["id"]
                )
                raise HTTPException(
                    status_code=429,
                    detail="Too many failed attempts. Please request a new OTP.",
                )

            if not pwd_context.verify(body.otp, row["otp_hash"]):
                await conn.execute(
                    "UPDATE password_reset_otps SET fail_count = fail_count + 1 WHERE id = $1",
                    row["id"],
                )
                remaining = settings.OTP_MAX_ATTEMPTS - row["fail_count"] - 1
                raise HTTPException(
                    status_code=400,
                    detail=f"Incorrect OTP. {remaining} attempt(s) remaining.",
                )

            # Mark as used (single-use)
            await conn.execute(
                "UPDATE password_reset_otps SET used = TRUE WHERE id = $1", row["id"]
            )

        reset_token = create_reset_token(body.email)
        logger.info(f"OTP verified for {body.email[:3]}***, reset token issued")
        return {"reset_token": reset_token, "valid": True}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"verify_otp error: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/reset-password", response_model=MessageResponse)
async def reset_password(
    body: ResetPasswordRequest,
    request: Request,
    token: str = Depends(oauth2_scheme),
) -> MessageResponse:
    """
    Reset password using a valid reset_token (from /verify-otp).
    Validates token purpose='password_reset', password strength, and confirmation.
    """
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM]
        )
        email: str = payload.get("sub")
        purpose: str = payload.get("purpose", "")
        if not email or purpose != "password_reset":
            raise HTTPException(status_code=401, detail="Invalid or expired reset token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired reset token")

    if body.new_password != body.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    err = _validate_password_strength(body.new_password)
    if err:
        raise HTTPException(status_code=400, detail=err)

    pool = _get_auth_pool(request)
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM users WHERE email = $1 AND is_active = TRUE", email
            )
            if not row:
                raise HTTPException(status_code=404, detail="User not found")
            new_hash = pwd_context.hash(body.new_password)
            await conn.execute(
                "UPDATE users SET password_hash = $1 WHERE email = $2",
                new_hash, email,
            )

        client_ip = request.client.host if request.client else "unknown"
        from services import email_service
        safe_create_task(
            asyncio.to_thread(email_service.send_password_reset_confirmation, email, client_ip),
            logger=logger,
            label="password reset confirmation email",
        )

        logger.info(f"Password reset successful for {email[:3]}***")
        return MessageResponse(message="Password reset successfully. Please log in with your new password.")

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"reset_password error: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")

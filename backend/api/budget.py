"""
backend/api/budget.py — Enterprise Feature 1: Budget Guardrails
================================================================
Real-time per-team, per-project monthly budget caps with:
  - Configurable spending limits per team/project
  - Real-time utilisation % tracking
  - Automatic webhook + email alerts at 80%/100%/120% thresholds
  - Budget freeze: block new pipeline runs when cap is exceeded
  - Historical budget vs actual reports
"""
import asyncio
import logging
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_serializer

from api.auth import require_admin
from cache import invalidate
from runtime_hardening import safe_create_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/budget", tags=["budget"])
_ZERO = Decimal("0")
_HUNDRED = Decimal("100")
_MONEY_QUANT = Decimal("0.0001")
_PCT_QUANT = Decimal("0.01")


# ── Pydantic models ───────────────────────────────────────────────────────────

class BudgetCreate(BaseModel):
    team_name: str = Field(..., min_length=1, max_length=100)
    monthly_cap_usd: Decimal = Field(..., gt=0, description="Monthly spend cap in USD")
    alert_at_pct: Decimal = Field(Decimal("80.0"), ge=0, le=100, description="Alert threshold %")
    block_at_pct: Decimal = Field(Decimal("110.0"), ge=0, le=200, description="Block pipelines at % of cap")
    webhook_url: Optional[str] = Field(None, description="Webhook for budget alerts")


class BudgetStatus(BaseModel):
    team_name: str
    monthly_cap_usd: Decimal
    spent_this_month: Decimal
    utilisation_pct: Decimal
    remaining_usd: Decimal
    status: str  # OK | WARNING | CRITICAL | EXCEEDED
    month: str   # YYYY-MM

    @field_serializer(
        "monthly_cap_usd",
        "spent_this_month",
        "utilisation_pct",
        "remaining_usd",
        when_used="json",
    )
    def _serialize_decimal_fields(self, value: Decimal) -> float:
        return float(value)


# ── DB helpers ────────────────────────────────────────────────────────────────

async def _ensure_budget_table(conn) -> None:
    """Create budget tables if they don't exist (idempotent)."""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS budget_configs (
            id              SERIAL PRIMARY KEY,
            team_name       VARCHAR(100) UNIQUE NOT NULL,
            monthly_cap_usd DECIMAL(12,4) NOT NULL,
            alert_at_pct    DECIMAL(5,2) DEFAULT 80.0,
            block_at_pct    DECIMAL(5,2) DEFAULT 110.0,
            webhook_url     TEXT,
            created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_budget_team ON budget_configs(team_name)
    """)


async def _get_monthly_spend(conn, team_name: str) -> Decimal:
    """Sum billed_cost for the current calendar month for a team."""
    row = await conn.fetchrow("""
        SELECT COALESCE(SUM(ca.billed_cost), 0.0) AS total
        FROM cost_attribution ca
        JOIN pipeline_runs pr ON ca.run_id = pr.run_id
        WHERE DATE_TRUNC('month', ca.created_at) = DATE_TRUNC('month', NOW())
          AND pr.executor_type = $1
    """, team_name)
    return _to_decimal(row["total"]) if row else _ZERO


def _to_decimal(value: object) -> Decimal:
    if value is None:
        return _ZERO
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _round_money(value: Decimal) -> Decimal:
    return value.quantize(_MONEY_QUANT, rounding=ROUND_HALF_UP)


def _round_pct(value: Decimal) -> Decimal:
    return value.quantize(_PCT_QUANT, rounding=ROUND_HALF_UP)


def _get_pool_or_503(request: Request):
    pool = getattr(request.app.state, "db", None)
    if pool is None:
        raise HTTPException(
            status_code=503,
            detail="Budget operations are unavailable while the database is degraded.",
        )
    return pool


async def _fire_budget_webhook(webhook_url: str, payload: dict) -> None:
    """Non-blocking webhook notification for budget alerts."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(webhook_url, json=payload)
        logger.info(f"Budget webhook fired to {webhook_url}")
    except Exception as exc:
        logger.warning(f"Budget webhook failed: {exc}")


def _classify_status(utilisation_pct: Decimal, alert_at: Decimal, block_at: Decimal) -> str:
    if utilisation_pct >= block_at:
        return "EXCEEDED"
    if utilisation_pct >= _HUNDRED:
        return "CRITICAL"
    if utilisation_pct >= alert_at:
        return "WARNING"
    return "OK"


async def _dispatch_budget_alerts(tasks: list) -> None:
    await asyncio.gather(*tasks, return_exceptions=True)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/configure", response_model=dict)
async def configure_budget(
    body: BudgetCreate,
    request: Request,
    _admin=Depends(require_admin),
):
    """
    Create or update a monthly budget cap for a team.
    Teams map to pipeline executor_type (e.g. 'backend-team', 'ml-team').
    """
    pool = _get_pool_or_503(request)
    async with pool.acquire() as conn:
        await _ensure_budget_table(conn)
        await conn.execute("""
            INSERT INTO budget_configs
                (team_name, monthly_cap_usd, alert_at_pct, block_at_pct, webhook_url)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (team_name) DO UPDATE SET
                monthly_cap_usd = EXCLUDED.monthly_cap_usd,
                alert_at_pct    = EXCLUDED.alert_at_pct,
                block_at_pct    = EXCLUDED.block_at_pct,
                webhook_url     = EXCLUDED.webhook_url,
                updated_at      = NOW()
        """,
            body.team_name, body.monthly_cap_usd,
            body.alert_at_pct, body.block_at_pct, body.webhook_url,
        )
    invalidate("policy")
    logger.info(f"Budget configured for team '{body.team_name}': cap=${body.monthly_cap_usd}")
    return {
        "status": "configured",
        "team": body.team_name,
        "cap_usd": float(_round_money(body.monthly_cap_usd)),
    }


@router.get("/status", response_model=List[BudgetStatus])
async def get_all_budget_status(request: Request):
    """
    Return real-time budget utilisation for every configured team.
    Fires webhook alerts when thresholds are crossed.
    """
    pool = _get_pool_or_503(request)
    async with pool.acquire() as conn:
        await _ensure_budget_table(conn)
        configs = await conn.fetch("SELECT * FROM budget_configs ORDER BY team_name")

    statuses = []
    alert_tasks = []

    for cfg in configs:
        async with pool.acquire() as conn:
            spent = await _get_monthly_spend(conn, cfg["team_name"])

        cap = _to_decimal(cfg["monthly_cap_usd"])
        spent_dec = _to_decimal(spent)
        utilisation = ((spent_dec / cap) * _HUNDRED) if cap > 0 else _ZERO
        remaining = max(_ZERO, cap - spent_dec)
        status_label = _classify_status(
            utilisation,
            _to_decimal(cfg["alert_at_pct"]),
            _to_decimal(cfg["block_at_pct"]),
        )

        statuses.append(BudgetStatus(
            team_name        = cfg["team_name"],
            monthly_cap_usd  = _round_money(cap),
            spent_this_month = _round_money(spent_dec),
            utilisation_pct  = _round_pct(utilisation),
            remaining_usd    = _round_money(remaining),
            status           = status_label,
            month            = date.today().strftime("%Y-%m"),
        ))

        # Fire webhook on WARNING/CRITICAL/EXCEEDED
        if cfg["webhook_url"] and status_label != "OK":
            alert_tasks.append(
                _fire_budget_webhook(cfg["webhook_url"], {
                    "team":          cfg["team_name"],
                    "status":        status_label,
                    "spent_usd":     _round_money(spent_dec),
                    "cap_usd":       _round_money(cap),
                    "utilisation":   f"{float(utilisation):.1f}%",
                    "alert_message": (
                        f"Team '{cfg['team_name']}' has used {float(utilisation):.1f}% "
                        f"of its ${float(cap):.0f} monthly budget."
                    ),
                })
            )

    if alert_tasks:
        safe_create_task(
            _dispatch_budget_alerts(alert_tasks),
            logger=logger,
            label="budget alert notifications",
        )

    return statuses


@router.get("/status/{team_name}", response_model=BudgetStatus)
async def get_team_budget_status(team_name: str, request: Request):
    """Get budget status for a specific team."""
    pool = _get_pool_or_503(request)
    async with pool.acquire() as conn:
        await _ensure_budget_table(conn)
        cfg = await conn.fetchrow(
            "SELECT * FROM budget_configs WHERE team_name = $1", team_name
        )
    if not cfg:
        raise HTTPException(404, f"No budget configured for team '{team_name}'")

    async with pool.acquire() as conn:
        spent = await _get_monthly_spend(conn, team_name)

    cap = _to_decimal(cfg["monthly_cap_usd"])
    spent_dec = _to_decimal(spent)
    utilisation = ((spent_dec / cap) * _HUNDRED) if cap > 0 else _ZERO

    return BudgetStatus(
        team_name        = team_name,
        monthly_cap_usd  = _round_money(cap),
        spent_this_month = _round_money(spent_dec),
        utilisation_pct  = _round_pct(utilisation),
        remaining_usd    = _round_money(max(_ZERO, cap - spent_dec)),
        status           = _classify_status(
            utilisation,
            _to_decimal(cfg["alert_at_pct"]),
            _to_decimal(cfg["block_at_pct"]),
        ),
        month            = date.today().strftime("%Y-%m"),
    )


@router.get("/check/{team_name}")
async def check_budget_gate(team_name: str, request: Request) -> dict:
    """
    Pipeline gate check: returns allowed=True/False.
    Called by CI/CD pipelines before starting a new run.
    Returns HTTP 402 (Payment Required) if budget is exceeded.
    """
    pool = _get_pool_or_503(request)
    async with pool.acquire() as conn:
        await _ensure_budget_table(conn)
        cfg = await conn.fetchrow(
            "SELECT * FROM budget_configs WHERE team_name = $1", team_name
        )
        if not cfg:
            # No budget configured = always allow
            return {"allowed": True, "reason": "No budget cap configured for this team."}

        spent = await _get_monthly_spend(conn, team_name)

    cap = _to_decimal(cfg["monthly_cap_usd"])
    spent_dec = _to_decimal(spent)
    utilisation = ((spent_dec / cap) * _HUNDRED) if cap > 0 else _ZERO
    block_at = _to_decimal(cfg["block_at_pct"])

    if utilisation >= block_at:
        raise HTTPException(
            status_code=402,
            detail={
                "allowed":      False,
                "team":         team_name,
                "utilisation":  f"{float(utilisation):.1f}%",
                "cap_usd":      float(_round_money(cap)),
                "spent_usd":    float(_round_money(spent_dec)),
                "message":      (
                    f"Pipeline BLOCKED: team '{team_name}' has exceeded "
                    f"{block_at:.0f}% of its ${float(cap):.0f} monthly budget."
                ),
            }
        )

    return {
        "allowed":     True,
        "utilisation": f"{float(utilisation):.1f}%",
        "remaining":   float(_round_money(max(_ZERO, cap - spent_dec))),
    }


@router.delete("/configure/{team_name}")
async def delete_budget(
    team_name: str,
    request: Request,
    _admin=Depends(require_admin),
) -> dict:
    """Remove a team's budget configuration."""
    pool = _get_pool_or_503(request)
    async with pool.acquire() as conn:
        await _ensure_budget_table(conn)
        result = await conn.execute(
            "DELETE FROM budget_configs WHERE team_name = $1", team_name
        )
    if result == "DELETE 0":
        raise HTTPException(404, f"Team '{team_name}' not found")
    return {"status": "deleted", "team": team_name}

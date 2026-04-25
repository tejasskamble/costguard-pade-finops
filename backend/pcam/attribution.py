import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional
import asyncpg

logger = logging.getLogger(__name__)
_ZERO = Decimal("0")
_MONEY_QUANT = Decimal("0.000001")
_PCT_QUANT = Decimal("0.0001")
_USAGE_QUANT = Decimal("0.0001")


def _to_decimal(value: object) -> Decimal:
    if value is None:
        return _ZERO
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _quantize_money(value: object) -> Decimal:
    return _to_decimal(value).quantize(_MONEY_QUANT, rounding=ROUND_HALF_UP)


def _quantize_pct(value: object) -> Decimal:
    return _to_decimal(value).quantize(_PCT_QUANT, rounding=ROUND_HALF_UP)


def _quantize_usage(value: object) -> Decimal:
    return _to_decimal(value).quantize(_USAGE_QUANT, rounding=ROUND_HALF_UP)

async def compute_cost_deviation(
    pool: asyncpg.Pool,
    stage_name: str,
    branch_type: str,
    executor_type: str,
    current_cost: Decimal | float
) -> float:
    """
    Compute cost deviation percentage:
    ((current_cost - historical_avg) / historical_avg) * 100
    If historical_avg is 0, return 0.0 to avoid division by zero.
    """
    async with pool.acquire() as conn:
        # Query historical average for same (stage, branch, executor)
        row = await conn.fetchrow("""
            SELECT AVG(billed_cost) AS avg_cost
            FROM cost_attribution ca
            JOIN pipeline_runs pr ON pr.run_id = ca.run_id
            WHERE ca.stage_name = $1
              AND pr.branch_type = $2
              AND pr.executor_type = $3
        """, stage_name, branch_type, executor_type)

        hist_avg = _to_decimal(row["avg_cost"]) if row and row["avg_cost"] is not None else _ZERO

        if hist_avg == _ZERO:
            return 0.0

        current_cost_dec = _to_decimal(current_cost)
        deviation = ((current_cost_dec - hist_avg) / hist_avg) * Decimal("100")
        return float(_quantize_pct(deviation))

async def write_attribution_snapshot(
    pool: asyncpg.Pool,
    snapshot: dict
) -> None:
    """
    Insert a single cost attribution snapshot into the database.
    Expected keys match cost_attribution columns.
    """
    prepared = {
        **snapshot,
        "billed_cost": _quantize_money(snapshot.get("billed_cost")),
        "effective_cost": _quantize_money(snapshot.get("effective_cost")),
        "usage_quantity": _quantize_usage(snapshot.get("usage_quantity")),
        "cost_deviation_pct": _quantize_pct(snapshot.get("cost_deviation_pct")),
        "historical_avg_cost": _quantize_money(snapshot.get("historical_avg_cost")),
    }
    columns = [
        "run_id", "stage_name", "resource_type", "billed_cost",
        "effective_cost", "billing_currency", "usage_quantity", "usage_unit",
        "provider", "region", "cost_deviation_pct", "historical_avg_cost",
        "crs_score", "pade_decision", "window_start", "window_end",
        "timestamp_start", "timestamp_end",
    ]
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO cost_attribution (
                run_id, stage_name, resource_type, billed_cost,
                effective_cost, billing_currency, usage_quantity, usage_unit,
                provider, region, cost_deviation_pct, historical_avg_cost,
                crs_score, pade_decision, window_start, window_end,
                timestamp_start, timestamp_end
            )
            VALUES (
                $1, $2, $3, $4,
                $5, $6, $7, $8,
                $9, $10, $11, $12,
                $13, $14, $15, $16,
                $17, $18
            )
            """,
            *[prepared.get(col) for col in columns],
        )

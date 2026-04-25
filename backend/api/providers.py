"""
backend/api/providers.py — Enterprise Feature 3: Multi-Cloud Cost Aggregation
===============================================================================
Unified cost intelligence across AWS, GCP, and Azure:
  - Per-provider spend breakdown and anomaly rate comparison
  - Provider efficiency score (cost per pipeline success)
  - Regional cost hotspot detection
  - Provider-level budget recommendations
  - Cost spike attribution by provider (did AWS costs spike this week?)
"""
import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, field_serializer

from database import get_db_conn
from cache import cached

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/providers", tags=["providers"])
_ZERO = Decimal("0")
_HUNDRED = Decimal("100")
_MONEY_QUANT = Decimal("0.0001")
_PCT_QUANT = Decimal("0.1")

KNOWN_PROVIDERS = {"aws", "gcp", "azure", "unknown"}

PROVIDER_DISPLAY = {
    "aws":     {"name": "Amazon Web Services", "color": "#F97316", "icon": "☁️"},
    "gcp":     {"name": "Google Cloud",        "color": "#6366F1", "icon": "🌐"},
    "azure":   {"name": "Microsoft Azure",     "color": "#10B981", "icon": "⚡"},
    "unknown": {"name": "Unknown Provider",    "color": "#6B7A99", "icon": "❓"},
}


# ── Pydantic models ───────────────────────────────────────────────────────────

class ProviderSummary(BaseModel):
    provider:           str
    display_name:       str
    color:              str
    total_cost_usd:     Decimal
    total_runs:         int
    anomaly_count:      int
    anomaly_rate_pct:   float
    avg_crs_score:      float
    cost_share_pct:     float
    top_region:         Optional[str]
    week_over_week_pct: float   # positive = increase, negative = decrease

    @field_serializer("total_cost_usd", when_used="json")
    def _serialize_total_cost(self, value: Decimal) -> float:
        return float(value)


class RegionHotspot(BaseModel):
    provider: str
    region:   str
    total_cost_usd: Decimal
    anomaly_count:  int
    avg_crs:        float

    @field_serializer("total_cost_usd", when_used="json")
    def _serialize_total_cost(self, value: Decimal) -> float:
        return float(value)


class ProviderComparison(BaseModel):
    period_days:        int
    total_cost_usd:     Decimal
    providers:          List[ProviderSummary]
    hotspots:           List[RegionHotspot]
    cheapest_provider:  Optional[str]
    riskiest_provider:  Optional[str]
    recommendation:     str

    @field_serializer("total_cost_usd", when_used="json")
    def _serialize_total_cost(self, value: Decimal) -> float:
        return float(value)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_recommendation(providers: List[ProviderSummary]) -> str:
    """Generate a plain-English cost recommendation from provider data."""
    if not providers:
        return "No pipeline data yet. Simulate a run to see multi-cloud analysis."

    riskiest = max(providers, key=lambda p: p.anomaly_rate_pct)
    cheapest = min(providers, key=lambda p: p.total_cost_usd)
    costliest = max(providers, key=lambda p: p.total_cost_usd)

    rec = []
    if riskiest.anomaly_rate_pct > 20:
        rec.append(
            f"{riskiest.provider.upper()} has the highest anomaly rate "
            f"({riskiest.anomaly_rate_pct:.0f}%) — review spot instance interruptions "
            f"or quota limits on that provider."
        )
    if costliest.week_over_week_pct > 20:
        rec.append(
            f"{costliest.provider.upper()} costs rose {costliest.week_over_week_pct:.0f}% "
            f"week-over-week — check for new workloads or pricing changes."
        )
    if cheapest.provider != costliest.provider:
        savings = costliest.total_cost_usd - cheapest.total_cost_usd
        if savings > Decimal("0.01"):
            rec.append(
                f"Migrating {costliest.provider.upper()} workloads to "
                f"{cheapest.provider.upper()} could save approx ${savings:.2f} "
                f"over this period."
            )
    return " ".join(rec) if rec else "All providers are performing within normal cost ranges."


def _to_decimal(value: object) -> Decimal:
    if value is None:
        return _ZERO
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _money(value: Decimal) -> Decimal:
    return value.quantize(_MONEY_QUANT, rounding=ROUND_HALF_UP)


def _pct(value: Decimal) -> float:
    return float(value.quantize(_PCT_QUANT, rounding=ROUND_HALF_UP))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/compare", response_model=ProviderComparison)
async def compare_providers(
    request: Request,
    days: int = Query(30, ge=1, le=365),
) -> ProviderComparison:
    """
    Multi-cloud cost comparison with anomaly rates, regional hotspots,
    and actionable recommendations. Core FinOps analytics endpoint.
    """
    pool = request.app.state.db

    async with pool.acquire() as conn:
        # Per-provider aggregates
        provider_rows = await conn.fetch("""
            SELECT
                LOWER(COALESCE(provider, 'unknown'))  AS provider,
                SUM(billed_cost)                       AS total_cost,
                COUNT(DISTINCT run_id)                 AS total_runs,
                COUNT(*) FILTER (
                    WHERE pade_decision IN ('WARN','AUTO_OPTIMISE','BLOCK')
                )                                      AS anomaly_count,
                COUNT(*)                               AS total_records,
                AVG(crs_score)                         AS avg_crs,
                MODE() WITHIN GROUP (ORDER BY region)  AS top_region
            FROM cost_attribution
            WHERE created_at >= NOW() - ($1 || ' days')::INTERVAL
            GROUP BY LOWER(COALESCE(provider, 'unknown'))
            ORDER BY total_cost DESC NULLS LAST
        """, str(days))

        # Previous-period costs for week-over-week comparison
        prev_rows = await conn.fetch("""
            SELECT
                LOWER(COALESCE(provider, 'unknown')) AS provider,
                SUM(billed_cost) AS total_cost
            FROM cost_attribution
            WHERE created_at >= NOW() - ($1 || ' days')::INTERVAL * 2
              AND created_at <  NOW() - ($1 || ' days')::INTERVAL
            GROUP BY LOWER(COALESCE(provider, 'unknown'))
        """, str(days))

        # Regional hotspots
        hotspot_rows = await conn.fetch("""
            SELECT
                LOWER(COALESCE(provider, 'unknown')) AS provider,
                region,
                SUM(billed_cost)                     AS total_cost,
                COUNT(*) FILTER (
                    WHERE pade_decision IN ('WARN','AUTO_OPTIMISE','BLOCK')
                )                                    AS anomaly_count,
                AVG(crs_score)                       AS avg_crs
            FROM cost_attribution
            WHERE created_at >= NOW() - ($1 || ' days')::INTERVAL
              AND region IS NOT NULL
            GROUP BY LOWER(COALESCE(provider, 'unknown')), region
            ORDER BY total_cost DESC NULLS LAST
            LIMIT 10
        """, str(days))

    # Build previous-period lookup
    prev_by_provider = {r["provider"]: _to_decimal(r["total_cost"]) for r in prev_rows}
    total_cost = sum((_to_decimal(r["total_cost"]) for r in provider_rows), _ZERO)

    providers = []
    for row in provider_rows:
        prov         = row["provider"]
        curr_cost = _to_decimal(row["total_cost"])
        prev_cost = prev_by_provider.get(prov, curr_cost)
        wow_pct = (((curr_cost - prev_cost) / prev_cost) * _HUNDRED) if prev_cost > 0 else _ZERO
        total_recs   = int(row["total_records"] or 1)
        anomalies    = int(row["anomaly_count"] or 0)
        display      = PROVIDER_DISPLAY.get(prov, PROVIDER_DISPLAY["unknown"])
        anomaly_rate = (Decimal(anomalies) / Decimal(total_recs) * _HUNDRED) if total_recs > 0 else _ZERO
        cost_share = (curr_cost / total_cost * _HUNDRED) if total_cost > 0 else _ZERO

        providers.append(ProviderSummary(
            provider           = prov,
            display_name       = display["name"],
            color              = display["color"],
            total_cost_usd     = _money(curr_cost),
            total_runs         = int(row["total_runs"] or 0),
            anomaly_count      = anomalies,
            anomaly_rate_pct   = _pct(anomaly_rate),
            avg_crs_score      = round(float(row["avg_crs"] or 0), 3),
            cost_share_pct     = _pct(cost_share),
            top_region         = row["top_region"],
            week_over_week_pct = _pct(wow_pct),
        ))

    hotspots = [
        RegionHotspot(
            provider      = r["provider"],
            region        = r["region"] or "unknown",
            total_cost_usd= _money(_to_decimal(r["total_cost"])),
            anomaly_count = int(r["anomaly_count"] or 0),
            avg_crs       = round(float(r["avg_crs"] or 0), 3),
        )
        for r in hotspot_rows
    ]

    cheapest  = min(providers, key=lambda p: p.total_cost_usd).provider if providers else None
    riskiest  = max(providers, key=lambda p: p.anomaly_rate_pct).provider if providers else None

    return ProviderComparison(
        period_days       = days,
        total_cost_usd    = _money(total_cost),
        providers         = providers,
        hotspots          = hotspots,
        cheapest_provider = cheapest,
        riskiest_provider = riskiest,
        recommendation    = _make_recommendation(providers),
    )


@router.get("/trend/{provider}")
async def get_provider_trend(
    provider: str,
    request: Request,
    days: int = Query(30, ge=7, le=365),
) -> dict:
    """
    Daily cost trend for a specific provider (AWS/GCP/Azure).
    Returns time-series suitable for Plotly line chart.
    """
    provider = provider.lower()
    if provider not in KNOWN_PROVIDERS:
        raise HTTPException(400, f"Unknown provider '{provider}'. Valid: {KNOWN_PROVIDERS}")

    pool = request.app.state.db
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                DATE(created_at AT TIME ZONE 'UTC') AS day,
                SUM(billed_cost) AS total_cost,
                AVG(crs_score)   AS avg_crs,
                COUNT(*) FILTER (WHERE pade_decision IN ('WARN','AUTO_OPTIMISE','BLOCK'))
                                 AS anomalies
            FROM cost_attribution
            WHERE LOWER(COALESCE(provider, 'unknown')) = $1
              AND created_at >= NOW() - ($2 || ' days')::INTERVAL
            GROUP BY DATE(created_at AT TIME ZONE 'UTC')
            ORDER BY day
        """, provider, str(days))

    display = PROVIDER_DISPLAY.get(provider, PROVIDER_DISPLAY["unknown"])
    return {
        "provider":     provider,
        "display_name": display["name"],
        "color":        display["color"],
        "period_days":  days,
        "series": [
            {
                "date":       str(r["day"]),
                "cost":       round(float(r["total_cost"] or 0), 6),
                "avg_crs":    round(float(r["avg_crs"] or 0), 3),
                "anomalies":  int(r["anomalies"] or 0),
            }
            for r in rows
        ],
    }


@router.get("/efficiency")
async def provider_efficiency(
    request: Request,
    days: int = Query(7, ge=1, le=90),
) -> dict:
    """
    Provider efficiency: cost-per-pipeline-run and anomaly density.
    Lower cost + lower anomaly rate = better provider for your workload.
    """
    pool = request.app.state.db
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                LOWER(COALESCE(provider, 'unknown'))  AS provider,
                COUNT(DISTINCT run_id)                 AS runs,
                SUM(billed_cost)                       AS total_cost,
                COUNT(*) FILTER (
                    WHERE pade_decision IN ('WARN','AUTO_OPTIMISE','BLOCK')
                )                                      AS anomalies
            FROM cost_attribution
            WHERE created_at >= NOW() - ($1 || ' days')::INTERVAL
            GROUP BY LOWER(COALESCE(provider, 'unknown'))
        """, str(days))

    result = []
    for row in rows:
        runs      = int(row["runs"] or 1)
        cost      = _to_decimal(row["total_cost"])
        anomalies = int(row["anomalies"] or 0)
        runs_decimal = Decimal(runs)
        cost_per_run = (cost / runs_decimal) if runs > 0 else _ZERO
        anomaly_density = (Decimal(anomalies) / runs_decimal) if runs > 0 else _ZERO
        efficiency = (Decimal("1.0") / (Decimal("1.0") + cost_per_run + anomaly_density)) if runs > 0 else _ZERO
        result.append({
            "provider":              row["provider"],
            "runs":                  runs,
            "total_cost_usd":        float(_money(cost)),
            "cost_per_run_usd":      float(cost_per_run.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)) if runs > 0 else 0,
            "anomalies":             anomalies,
            "anomaly_density":       float(anomaly_density.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)) if runs > 0 else 0,
            "efficiency_score":      float(efficiency.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)) if runs > 0 else 0,
        })

    result.sort(key=lambda x: x["efficiency_score"], reverse=True)
    return {"period_days": days, "providers": result}

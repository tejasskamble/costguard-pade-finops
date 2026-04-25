"""
backend/api/ingest.py - CostGuard v17.0

Ingestion and simulation endpoints for the canonical CostGuard runtime.
"""
import asyncio
import json
import logging
import os
import random
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from api.auth import get_current_user_id, require_admin, UserProfile
from cache import invalidate
from config import settings
from peg.policy_engine import extract_policy_bundle
from runtime_hardening import retry_async, safe_create_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ingest"])


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


_ZERO = Decimal("0")
_MONEY_QUANT = Decimal("0.000001")
_USAGE_QUANT = Decimal("0.0001")
_PCT_QUANT = Decimal("0.0001")


def _to_decimal(value: object) -> Decimal:
    if value is None:
        return _ZERO
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _quantize_money(value: object) -> Decimal:
    return _to_decimal(value).quantize(_MONEY_QUANT, rounding=ROUND_HALF_UP)


def _quantize_usage(value: object) -> Decimal:
    return _to_decimal(value).quantize(_USAGE_QUANT, rounding=ROUND_HALF_UP)


def _quantize_pct(value: object) -> Decimal:
    return _to_decimal(value).quantize(_PCT_QUANT, rounding=ROUND_HALF_UP)


def _json_number(value: Decimal) -> float:
    return float(value)

# ── Kafka guard (CONSTRAINT-1) ───────────────────────────────────────────────
KAFKA_ENABLED = os.getenv("KAFKA_ENABLED", "false").lower() == "true"
_kafka_producer = None


def _get_kafka_producer():
    global _kafka_producer
    if _kafka_producer is None:
        try:
            from confluent_kafka import Producer
            _kafka_producer = Producer(
                {"bootstrap.servers": os.getenv("KAFKA_BROKERS", "kafka:9092")}
            )
        except ImportError:
            logger.error("confluent_kafka not installed. Set KAFKA_ENABLED=false.")
            raise
    return _kafka_producer


# ── LocalEventBus (replaces Kafka for local dev) ─────────────────────────────

class LocalEventBus:
    """In-process asyncio queue — Kafka replacement (CONSTRAINT-1)."""
    _queue: asyncio.Queue = asyncio.Queue(maxsize=10_000)

    @classmethod
    async def publish(cls, topic: str, key: str, value: dict) -> None:
        await cls._queue.put({"topic": topic, "key": key, "value": value})

    @classmethod
    async def consume_forever(cls, handler):
        while True:
            item = await cls._queue.get()
            try:
                await handler(item)
            except Exception as exc:
                logger.exception(f"Event handler error: {exc}")
            cls._queue.task_done()


# ── PCAM asyncio attribution loop (replaces Flink — GAP-3) ──────────────────

async def pcam_attribution_loop(pool) -> None:
    window: List[dict] = []
    window_start = _utc_now()
    while True:
        try:
            item = await asyncio.wait_for(LocalEventBus._queue.get(), timeout=5.0)
            window.append(item.get("value", {}))
            LocalEventBus._queue.task_done()
        except asyncio.TimeoutError:
            logger.debug("PCAM attribution loop heartbeat timeout; awaiting new records")
        except Exception as exc:
            logger.exception(f"PCAM loop error: {exc}")

        if (_utc_now() - window_start).total_seconds() >= 60 and window:
            try:
                await _flush_window(window, window_start, pool)
            except Exception as exc:
                logger.exception(f"Window flush error: {exc}")
            window = []
            window_start = _utc_now()


async def _flush_window(records: List[dict], window_start: datetime, pool) -> None:
    window_end = _utc_now()
    async with pool.acquire() as conn:
        for rec in records:
            try:
                await conn.execute(
                    """
                    INSERT INTO cost_attribution (
                        run_id, stage_name, resource_type, billed_cost, effective_cost,
                        billing_currency, usage_quantity, usage_unit, provider, region,
                        cost_deviation_pct, historical_avg_cost, crs_score, pade_decision,
                        ai_recommendation, window_start, window_end, timestamp_start, timestamp_end
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19)
                    """,
                    rec.get("run_id"), rec.get("stage_name"), rec.get("resource_type"),
                    _quantize_money(rec.get("billed_cost")),
                    _quantize_money(rec.get("effective_cost")),
                    rec.get("billing_currency", "USD"),
                    _quantize_usage(rec.get("usage_quantity")),
                    rec.get("usage_unit"),
                    rec.get("provider"), rec.get("region"),
                    _quantize_pct(rec.get("cost_deviation_pct", 0.0)),
                    _quantize_money(rec.get("historical_avg_cost", 0.0)),
                    rec.get("crs_score"), rec.get("pade_decision"),
                    rec.get("ai_recommendation"),
                    window_start, window_end,
                    rec.get("timestamp_start"), rec.get("timestamp_end"),
                )
            except Exception as exc:
                logger.exception(f"DB write error for run {rec.get('run_id')}: {exc}")
    invalidate("alerts")   # FEATURE-4: clear stale cache after new data
    logger.info("Flushed %d records (%s -> %s)", len(records), window_start, window_end)


# ── Pydantic models ───────────────────────────────────────────────────────────

class CostRecordRequest(BaseModel):
    run_id: str
    stage_name: str
    branch_type: str
    executor_type: str
    resource_type: str
    billed_cost: Decimal
    effective_cost: Decimal
    billing_currency: str = "USD"
    usage_quantity: float
    usage_unit: str
    provider: str
    region: str
    timestamp_start: str
    timestamp_end: str


# ── Stage constants ───────────────────────────────────────────────────────────
STAGE_ORDER = [
    "checkout", "build", "unit_test", "integration_test",
    "security_scan", "docker_build", "deploy_staging", "deploy_prod",
]
EXECUTOR_TYPES = ["github_actions", "gitlab_ci", "jenkins"]
BRANCH_TYPES   = ["main", "feature", "hotfix", "release"]
PROVIDERS      = ["aws", "gcp", "azure"]
REGIONS        = ["ap-south-1", "us-east-1", "europe-west1", "eastus"]

STAGE_BASE_COSTS = {
    "checkout": 0.0012, "build": 0.0180, "unit_test": 0.0095,
    "integration_test": 0.0240, "security_scan": 0.0070,
    "docker_build": 0.0310, "deploy_staging": 0.0140, "deploy_prod": 0.0390,
}


def _build_synthetic_stage_data(anomaly_level: float, stage_name: str) -> dict:
    rng = random.Random()
    executor_enc = rng.randint(0, 2)
    branch_enc   = rng.randint(0, 3)
    stage_data   = {}
    for s in STAGE_ORDER:
        base = STAGE_BASE_COSTS.get(s, 0.01)
        mult = 1.0 + anomaly_level * rng.uniform(0.5, 3.0) if s == stage_name else 1.0
        cost = base * mult * rng.uniform(0.85, 1.15)
        hist = base * rng.uniform(0.9, 1.1)
        stage_data[s] = {
            "cost":          cost,
            "deviation":     (cost - hist) / hist if hist > 0 else 0.0,
            "duration":      rng.uniform(10, 300) * mult,
            "hist_avg_cost": hist,
            "hist_avg_dur":  rng.uniform(10, 300),
            "executor_enc":  executor_enc,
            "branch_enc":    branch_enc,
            "provider_enc":  rng.randint(0, 2),
        }
    return stage_data


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/ingest")
async def ingest_cost_record(
    record: CostRecordRequest,
    request: Request,
    user_id: int = Depends(get_current_user_id),
) -> dict:
    """Receive a PTA cost record and publish to the event bus."""
    pool = request.app.state.db
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO pipeline_runs (run_id, user_id, branch_type, executor_type, provider, region)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (run_id) DO NOTHING
            """,
            record.run_id, user_id,
            record.branch_type, record.executor_type,
            record.provider, record.region,
        )

    data = record.model_dump()
    data["billed_cost"] = _json_number(_quantize_money(record.billed_cost))
    data["effective_cost"] = _json_number(_quantize_money(record.effective_cost))
    data["usage_quantity"] = _json_number(_quantize_usage(record.usage_quantity))
    data["received_at"] = _utc_now().isoformat()

    if KAFKA_ENABLED:
        try:
            producer = _get_kafka_producer()
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: producer.produce(
                    "pipeline.cost.records",
                    key=record.run_id,
                    value=json.dumps(data),
                ),
            )
            await loop.run_in_executor(None, producer.poll, 0)
        except Exception:
            logger.debug("Kafka publish failed; falling back to LocalEventBus", exc_info=True)
            await LocalEventBus.publish("pipeline.cost.records", record.run_id, data)
    else:
        await LocalEventBus.publish("pipeline.cost.records", record.run_id, data)

    invalidate("alerts")
    logger.info(f"Ingested: run={record.run_id}, stage={record.stage_name}")
    return {"status": "ingested", "run_id": record.run_id, "timestamp": data["received_at"]}


@router.post("/ingest/simulate")
async def simulate_pipeline_run(
    request: Request,
    current_user: UserProfile = Depends(require_admin),
    anomaly_level: float = Query(0.5, ge=0.0, le=1.0),
    stage_name: str = Query("integration_test"),
) -> dict:
    """
    NEW-BUG-2 FIX: Simulate a synthetic pipeline run and DEFINITELY persist to DB.
    Inserts into BOTH pipeline_runs and cost_attribution.
    Uses ai_recommendation from the fixed score_pipeline() (NEW-BUG-1 fix).
    """
    pool = request.app.state.db
    if stage_name not in STAGE_ORDER:
        stage_name = "integration_test"

    # ── Resolve user from JWT (optional — anonymous simulation OK) ────────────
    resolved_user_id: Optional[int] = current_user.id
    user_email: Optional[str] = current_user.email

    run_id   = str(uuid.uuid4())
    provider = random.choice(PROVIDERS)
    region   = random.choice(REGIONS)
    now      = _utc_now()

    # ── Score through PADE (NEW-BUG-1 fix ensures non-empty ai_recommendation) ─
    stage_data  = _build_synthetic_stage_data(anomaly_level, stage_name)
    billed_cost = _quantize_money(stage_data[stage_name]["cost"])

    try:
        from pade.inference import score_pipeline, generate_anomaly_recommendation
        pade_result = await score_pipeline(
            run_id=run_id,
            stage_data=stage_data,
            stage_name=stage_name,
        )
        crs              = pade_result.crs
        decision_raw     = pade_result.decision
        ai_recommendation = pade_result.ai_recommendation  # Never "" on WARN+
    except Exception as exc:
        logger.exception(f"PADE scoring failed: {exc}")
        crs, decision_raw, ai_recommendation = 0.5 + anomaly_level * 0.4, "WARN", ""

    # ── Apply live policy thresholds ──────────────────────────────────────────
    try:
        async def _fetch_policy_row():
            async with pool.acquire() as conn:
                return await conn.fetchrow("SELECT * FROM policy_config ORDER BY id LIMIT 1")

        pcfg = await retry_async(
            _fetch_policy_row,
            attempts=2,
            delay=0.25,
            logger=logger,
            label="policy configuration lookup",
        )
        warn_t  = float(pcfg["warn_threshold"])
        auto_t  = float(pcfg["auto_optimise_threshold"])
        block_t = float(pcfg["block_threshold"])
    except Exception as exc:
        logger.warning("Policy threshold lookup failed; using defaults: %s", exc)
        warn_t, auto_t, block_t = 0.50, 0.75, 0.90

    from peg.opa_client import evaluate_policy
    policy_bundle = extract_policy_bundle(pcfg if 'pcfg' in locals() else None)
    decision_payload = await evaluate_policy(
        metrics={
            "crs": crs,
            "billed_cost": float(billed_cost),
            "duration_seconds": float(stage_data[stage_name]["duration"]),
            "latency_p95": float(stage_data[stage_name]["duration"]) * 1000.0,
        },
        context={
            "run_id": run_id,
            "stage_name": stage_name,
            "executor_type": "simulation",
            "branch": random.choice(BRANCH_TYPES),
            "domain": "synthetic",
            "gh_is_pr": False,
            "gh_by_core_team_member": True,
        },
        policy_bundle=policy_bundle,
    )
    decision = decision_payload["decision"]

    # ── Apply optimisation if warranted ──────────────────────────────────────
    action_taken: Optional[str] = None
    savings:      Optional[int] = None
    if decision in ("AUTO_OPTIMISE", "BLOCK"):
        try:
            from peg.optimiser import apply_optimisation
            opt          = apply_optimisation(stage_name)
            action_taken = opt.action_name
            savings      = opt.avg_savings_pct
        except Exception as exc:
            logger.debug("Optimiser unavailable for decision=%s: %s", decision, exc, exc_info=True)

    # ── NEW-BUG-2 FIX: Persist to database (pipeline_runs + cost_attribution) ─
    hist_avg = _quantize_money(STAGE_BASE_COSTS.get(stage_name, 0.01))
    deviation = (
        _quantize_pct(((billed_cost - hist_avg) / hist_avg) * Decimal("100"))
        if hist_avg
        else _ZERO
    )
    effective_cost = _quantize_money(billed_cost * Decimal("0.85"))
    usage_quantity = _quantize_usage(billed_cost * Decimal("1000"))
    window_end = now + timedelta(seconds=60)

    async with pool.acquire() as conn:
        # 1. Upsert pipeline run
        await conn.execute(
            """
            INSERT INTO pipeline_runs
                (run_id, user_id, branch_type, executor_type, provider,
                 region, total_cost_usd, stage_count, is_anomalous)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
            ON CONFLICT (run_id) DO NOTHING
            """,
            run_id, resolved_user_id,
            random.choice(BRANCH_TYPES), "simulation",
            provider, region,
            billed_cost, 1,
            decision not in ("ALLOW",),
        )
        # 2. Insert cost attribution row — this triggers the pg_notify (FEATURE-3)
        await conn.execute(
            """
            INSERT INTO cost_attribution (
                run_id, stage_name, resource_type,
                billed_cost, effective_cost, billing_currency,
                usage_quantity, usage_unit, provider, region,
                cost_deviation_pct, historical_avg_cost,
                crs_score, pade_decision, ai_recommendation,
                window_start, window_end, timestamp_start, timestamp_end
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19)
            """,
            run_id, stage_name, "compute",
            billed_cost, effective_cost, "USD",
            usage_quantity, "CPU-hours",
            provider, region,
            deviation, hist_avg,
            crs, decision, ai_recommendation,
            now, window_end, now, now + timedelta(minutes=5),
        )

    invalidate("alerts")   # FEATURE-4: invalidate cache after write

    # ── Fire notifications (fire-and-forget) ─────────────────────────────────
    if decision in ("WARN", "AUTO_OPTIMISE", "BLOCK"):
        from peg.notifier import send_slack_alert, send_email_alert
        if settings.SLACK_BOT_TOKEN and settings.SLACK_DEFAULT_CHANNEL:
            safe_create_task(
                send_slack_alert(
                    decision=decision, run_id=run_id, crs=crs,
                    stage_name=stage_name, cost=billed_cost, optimisation=action_taken,
                    ai_recommendation=ai_recommendation,
                ),
                logger=logger,
                label="slack notification",
            )
        if user_email and all([settings.SMTP_HOST, settings.SMTP_USER, settings.SMTP_PASSWORD]):
            safe_create_task(
                send_email_alert(
                    decision=decision, run_id=run_id, crs=crs,
                    stage_name=stage_name, cost=billed_cost,
                    recipient_email=user_email,
                    optimisation=action_taken,
                    ai_recommendation=ai_recommendation,
                ),
                logger=logger,
                label="email notification",
            )

    logger.info(
        f"Simulated: run={run_id}, stage={stage_name}, "
        f"anomaly={anomaly_level:.2f}, CRS={crs:.3f}, decision={decision}"
    )
    return {
        "run_id":            run_id,
        "stage_name":        stage_name,
        "crs":               round(crs, 4),
        "decision":          decision,
        "billed_cost":       _json_number(billed_cost),
        "action_taken":      action_taken,
        "projected_savings": savings,
        "ai_recommendation": ai_recommendation,
        "provider":          provider,
        "region":            region,
    }

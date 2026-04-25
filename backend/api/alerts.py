"""
backend/api/alerts.py - CostGuard v17.0

Real-time alert endpoints for the canonical 3-domain enterprise stack.
"""
import asyncio
import json
import logging
from typing import Optional

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from cache import cached, invalidate
from database import get_db_conn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["alerts"])

# Per-connection alert buffer (one queue per SSE client)
_alert_queues: list[asyncio.Queue] = []


def _pg_notify_callback(conn, pid, channel, payload):
    """Asyncpg LISTEN callback — broadcasts to all connected SSE clients."""
    try:
        event = json.loads(payload)
        for q in list(_alert_queues):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.debug("Dropping alert event for a saturated SSE queue")
    except Exception as exc:
        logger.warning(f"NOTIFY parse error: {exc}")


# ── REST endpoint (cached) ────────────────────────────────────────────────────

@router.get("/alerts/recent")
@cached("alerts", key_fn=lambda *a, **kw: f"recent:{kw.get('limit', 50)}")
async def get_recent_alerts(
    limit: int = Query(50, ge=1, le=500),
    conn: asyncpg.Connection = Depends(get_db_conn),
):
    """
    Return most recent WARN/AUTO_OPTIMISE/BLOCK decisions.
    FEATURE-4: cached 30s — VG-8: second call within TTL hits cache, not DB.
    """
    try:
        rows = await conn.fetch(
            """
            SELECT id, run_id, stage_name, resource_type,
                   billed_cost, effective_cost, provider, region,
                   cost_deviation_pct, historical_avg_cost,
                   crs_score, pade_decision,
                   ai_recommendation,
                   window_start, window_end, created_at
            FROM cost_attribution
            WHERE pade_decision IN ('WARN', 'AUTO_OPTIMISE', 'BLOCK')
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit,
        )
        return [dict(row) for row in rows]
    except Exception as exc:
        logger.exception("Error fetching recent alerts")
        raise HTTPException(500, "Failed to fetch recent alerts.")


@router.get("/alerts/summary")
@cached("daily_summary", key_fn=lambda *a, **kw: "summary:24h")
async def get_alerts_summary(conn: asyncpg.Connection = Depends(get_db_conn)):
    """Aggregated KPIs: total cost today, active pipelines, avg CRS, anomaly count."""
    try:
        row = await conn.fetchrow(
            """
            SELECT
                COALESCE(SUM(billed_cost), 0) AS total_cost_today,
                COUNT(DISTINCT run_id)         AS active_pipelines,
                COALESCE(AVG(crs_score), 0)   AS avg_crs,
                COUNT(*) FILTER (WHERE pade_decision IN ('WARN','AUTO_OPTIMISE','BLOCK'))
                                               AS anomaly_count
            FROM cost_attribution
            WHERE created_at >= NOW() - INTERVAL '24 hours'
            """
        )
        return dict(row) if row else {}
    except Exception as exc:
        logger.exception("Error fetching alerts summary")
        raise HTTPException(500, "Failed to fetch alert summary.")


# ── SSE stream endpoint (FEATURE-3) ──────────────────────────────────────────

@router.get("/alerts/stream")
async def alert_stream(request: Request) -> StreamingResponse:
    """
    FEATURE-3: Server-Sent Events stream for real-time alert notifications.
    Uses asyncpg LISTEN on 'pg_costguard_alerts' channel (triggered by
    migration 002's INSERT trigger on cost_attribution).
    CONSTRAINT-E: Cleans up LISTEN on client disconnect.
    """
    pool = request.app.state.db
    my_queue: asyncio.Queue = asyncio.Queue(maxsize=200)

    async def event_generator():
        # Register this client's queue
        _alert_queues.append(my_queue)

        # Acquire a dedicated connection for LISTEN (cannot share with pool)
        conn = await pool.acquire()
        try:
            await conn.add_listener("pg_costguard_alerts", _pg_notify_callback)
            logger.info("SSE client connected — listening on pg_costguard_alerts")

            # Send an initial heartbeat so the client knows we're live
            yield "data: {\"type\": \"connected\"}\n\n"

            while True:
                # CONSTRAINT-E: check disconnect before blocking
                if await request.is_disconnected():
                    logger.info("SSE client disconnected")
                    break

                try:
                    event = await asyncio.wait_for(my_queue.get(), timeout=15.0)
                    payload = json.dumps(event)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    # Keepalive comment to prevent proxy timeouts
                    yield ": keepalive\n\n"

        finally:
            # CONSTRAINT-E: always clean up LISTEN subscription
            try:
                await conn.remove_listener("pg_costguard_alerts", _pg_notify_callback)
            except Exception as exc:
                logger.debug("SSE listener cleanup warning: %s", exc)
            await pool.release(conn)
            if my_queue in _alert_queues:
                _alert_queues.remove(my_queue)
            logger.info("SSE cleanup complete")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

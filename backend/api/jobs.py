"""
backend/api/jobs.py — Enterprise Feature 2: Async Background Job Queue
========================================================================
Priority-based async job queue for heavy operations:
  - Full pipeline cost re-analysis (re-score all runs with new model weights)
  - Bulk anomaly backfill (apply new CRS thresholds to historical data)
  - Data export jobs (CSV/JSON export for BI tools)
  - Scheduled report generation

Uses asyncio.PriorityQueue internally (no Celery/Redis required).
Jobs are persistent across requests via an in-process queue with DB status tracking.
"""
import asyncio
import logging
import uuid
from datetime import datetime, timezone
from enum import IntEnum
from typing import Optional, List, Any
from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field

from api.auth import require_admin

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


# ── Job priority levels ───────────────────────────────────────────────────────

class Priority(IntEnum):
    CRITICAL = 1   # model reload, security scan
    HIGH     = 2   # anomaly backfill
    NORMAL   = 3   # scheduled reports
    LOW      = 4   # data exports, archiving


# ── In-process job queue (no Redis needed) ────────────────────────────────────

_job_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=500)
_job_registry: dict[str, dict] = {}   # job_id → status dict
_worker_running: bool = False


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ── Pydantic models ───────────────────────────────────────────────────────────

class JobSubmit(BaseModel):
    job_type: str = Field(..., description="reanalysis | backfill | export | report")
    priority: int = Field(Priority.NORMAL, ge=1, le=4)
    params: dict  = Field(default_factory=dict, description="Job-specific parameters")


class JobStatus(BaseModel):
    job_id:      str
    job_type:    str
    priority:    int
    status:      str  # QUEUED | RUNNING | DONE | FAILED
    queued_at:   str
    started_at:  Optional[str] = None
    finished_at: Optional[str] = None
    result:      Optional[Any] = None
    error:       Optional[str] = None
    progress_pct: int = 0


# ── Job handlers ──────────────────────────────────────────────────────────────

async def _handle_reanalysis(job_id: str, params: dict, pool) -> dict:
    """
    Re-score all pipeline runs using the current PADE model.
    Useful after uploading a new trained GAT checkpoint.
    """
    _update_job(job_id, status="RUNNING", progress_pct=5)
    from pade.inference import score_pipeline, bootstrap_gat_checkpoint
    bootstrap_gat_checkpoint()

    async with pool.acquire() as conn:
        runs = await conn.fetch("""
            SELECT DISTINCT run_id, stage_name,
                   billed_cost, crs_score
            FROM cost_attribution
            ORDER BY created_at DESC
            LIMIT $1
        """, params.get("limit", 100))

    total = len(runs)
    updated = 0

    if total == 0:
        _update_job(job_id, progress_pct=100)
        return {"runs_processed": 0, "runs_updated": 0}

    for i, run in enumerate(runs):
        pct = int((i + 1) / total * 90) + 5
        _update_job(job_id, progress_pct=pct)

        try:
            result = await score_pipeline(
                run_id=run["run_id"],
                stage_data={run["stage_name"]: {"cost": float(run["billed_cost"] or 0)}},
                stage_name=run["stage_name"],
            )
            async with pool.acquire() as conn:
                await conn.execute("""
                    UPDATE cost_attribution
                    SET crs_score = $1, pade_decision = $2
                    WHERE run_id = $3 AND stage_name = $4
                """, result.crs, result.decision, run["run_id"], run["stage_name"])
            updated += 1
        except Exception as exc:
            logger.warning(f"Re-analysis failed for {run['run_id']}: {exc}")
        await asyncio.sleep(0)   # yield to event loop

    return {"runs_processed": total, "runs_updated": updated}


async def _handle_backfill(job_id: str, params: dict, pool) -> dict:
    """
    Apply new CRS thresholds to historical cost_attribution records.
    Reclassifies decisions without re-running PADE inference.
    """
    _update_job(job_id, status="RUNNING", progress_pct=10)
    warn  = params.get("warn_threshold",  0.50)
    auto  = params.get("auto_threshold",  0.75)
    block = params.get("block_threshold", 0.90)

    async with pool.acquire() as conn:
        result = await conn.execute("""
            UPDATE cost_attribution SET pade_decision =
                CASE
                    WHEN crs_score >= $3 THEN 'BLOCK'
                    WHEN crs_score >= $2 THEN 'AUTO_OPTIMISE'
                    WHEN crs_score >= $1 THEN 'WARN'
                    ELSE 'ALLOW'
                END
            WHERE crs_score IS NOT NULL
        """, warn, auto, block)

    _update_job(job_id, progress_pct=100)
    rows_updated = int(result.split()[-1]) if result else 0
    return {"rows_reclassified": rows_updated, "thresholds": {"warn": warn, "auto": auto, "block": block}}


async def _handle_export(job_id: str, params: dict, pool) -> dict:
    """
    Export cost_attribution data to a JSON summary for BI tools.
    Stored in the job registry for retrieval via GET /api/jobs/{id}/result.
    HF-12: days is validated as a positive int before being interpolated into SQL.
    """
    _update_job(job_id, status="RUNNING", progress_pct=20)
    # HF-12: validate to int to prevent SQL injection via INTERVAL construction
    try:
        days = max(1, min(365, int(params.get("days", 30))))
    except (TypeError, ValueError):
        days = 30
    fmt  = params.get("format", "summary")

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT stage_name,
                   DATE(created_at) AS day,
                   SUM(billed_cost) AS total_cost,
                   AVG(crs_score)   AS avg_crs,
                   COUNT(*)         AS records,
                   COUNT(*) FILTER (WHERE pade_decision IN ('WARN','AUTO_OPTIMISE','BLOCK')) AS anomalies
            FROM cost_attribution
            WHERE created_at >= NOW() - ($1 * INTERVAL '1 day')
            GROUP BY stage_name, DATE(created_at)
            ORDER BY day DESC, total_cost DESC
        """,
            days,
        )

    _update_job(job_id, progress_pct=90)
    data = [dict(r) for r in rows]
    for d in data:
        for k, v in d.items():
            if hasattr(v, 'isoformat'):
                d[k] = v.isoformat()
            elif hasattr(v, '__float__'):
                d[k] = float(v)

    return {"format": fmt, "days": days, "record_count": len(data), "data": data[:500]}


async def _handle_report(job_id: str, params: dict, pool) -> dict:
    """Generate a structured cost summary report for the last N days.
    HF-12: days validated as int — same fix as _handle_export."""
    _update_job(job_id, status="RUNNING", progress_pct=15)
    try:
        days = max(1, min(365, int(params.get("days", 7))))
    except (TypeError, ValueError):
        days = 7

    async with pool.acquire() as conn:
        summary = await conn.fetchrow(
            """
            SELECT
                COALESCE(SUM(billed_cost), 0)   AS total_cost,
                COALESCE(AVG(crs_score), 0)     AS avg_crs,
                COUNT(DISTINCT run_id)           AS total_runs,
                COUNT(*) FILTER (WHERE pade_decision = 'BLOCK') AS blocks,
                COUNT(*) FILTER (WHERE pade_decision = 'AUTO_OPTIMISE') AS optimisations
            FROM cost_attribution
            WHERE created_at >= NOW() - ($1 * INTERVAL '1 day')
        """,
            days,
        )

        by_stage = await conn.fetch(
            """
            SELECT stage_name, SUM(billed_cost) AS cost, COUNT(*) AS runs
            FROM cost_attribution
            WHERE created_at >= NOW() - ($1 * INTERVAL '1 day')
            GROUP BY stage_name ORDER BY cost DESC LIMIT 10
        """,
            days,
        )

    _update_job(job_id, progress_pct=100)
    return {
        "period_days":  days,
        "generated_at": _utc_now().isoformat(),
        "totals":       {k: float(v or 0) for k, v in dict(summary).items()},
        "top_stages":   [dict(r) for r in by_stage],
    }


_HANDLERS = {
    "reanalysis": _handle_reanalysis,
    "backfill":   _handle_backfill,
    "export":     _handle_export,
    "report":     _handle_report,
}


# ── Worker loop ───────────────────────────────────────────────────────────────

def _update_job(job_id: str, **kwargs) -> None:
    if job_id in _job_registry:
        _job_registry[job_id].update(kwargs)


def get_worker_state() -> dict:
    return {
        "running": _worker_running,
        "queued_jobs": _job_queue.qsize(),
        "tracked_jobs": len(_job_registry),
    }


async def _worker_loop(pool) -> None:
    """
    Consume jobs from the priority queue.
    Runs as a background asyncio task launched from main.py lifespan.
    """
    global _worker_running
    _worker_running = True
    logger.info("Background job worker started.")

    while True:
        try:
            priority, job_id, job_type, params = await asyncio.wait_for(
                _job_queue.get(), timeout=10.0
            )
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break

        job = _job_registry.get(job_id)
        if job is None:
            _job_queue.task_done()
            continue

        if job.get("status") == "CANCELLED":
            logger.info("Skipping cancelled job %s before execution.", job_id)
            _job_queue.task_done()
            continue

        _update_job(job_id, status="RUNNING", started_at=_utc_now().isoformat())
        handler = _HANDLERS.get(job_type)

        if not handler:
            _update_job(job_id, status="FAILED", error=f"Unknown job type: {job_type}")
            _job_queue.task_done()
            continue

        try:
            result = await handler(job_id, params, pool)
            _update_job(
                job_id, status="DONE",
                result=result, progress_pct=100,
                finished_at=_utc_now().isoformat()
            )
            logger.info(f"Job {job_id} ({job_type}) completed.")
        except Exception as exc:
            logger.exception(f"Job {job_id} failed: {exc}")
            _update_job(job_id, status="FAILED", error=str(exc),
                        finished_at=_utc_now().isoformat())
        finally:
            _job_queue.task_done()

    _worker_running = False


def start_worker(pool) -> asyncio.Task:
    """Launch the worker loop as a background asyncio task."""
    return asyncio.create_task(_worker_loop(pool))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/submit", response_model=JobStatus)
async def submit_job(
    body: JobSubmit,
    request: Request,
    _admin=Depends(require_admin),
) -> JobStatus:
    """
    Submit a background job. Returns immediately with a job_id.
    Poll GET /api/jobs/{job_id} to track progress.
    """
    if body.job_type not in _HANDLERS:
        raise HTTPException(400, f"Unknown job type '{body.job_type}'. Valid: {list(_HANDLERS)}")

    if _job_queue.full():
        raise HTTPException(503, "Job queue is full. Try again later.")

    job_id    = str(uuid.uuid4())
    queued_at = _utc_now().isoformat()

    job_rec = {
        "job_id":      job_id,
        "job_type":    body.job_type,
        "priority":    body.priority,
        "status":      "QUEUED",
        "queued_at":   queued_at,
        "started_at":  None,
        "finished_at": None,
        "result":      None,
        "error":       None,
        "progress_pct": 0,
    }
    _job_registry[job_id] = job_rec
    await _job_queue.put((body.priority, job_id, body.job_type, body.params))

    logger.info(f"Job queued: {job_id} ({body.job_type}, priority={body.priority})")
    return JobStatus(**job_rec)


@router.get("/{job_id}", response_model=JobStatus)
async def get_job_status(
    job_id: str,
    _admin=Depends(require_admin),
) -> JobStatus:
    """Poll job status and progress."""
    job = _job_registry.get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    return JobStatus(**job)


@router.get("/{job_id}/result")
async def get_job_result(
    job_id: str,
    _admin=Depends(require_admin),
):
    """Retrieve the full result payload for a completed job."""
    job = _job_registry.get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    if job["status"] != "DONE":
        raise HTTPException(409, f"Job not done yet. Status: {job['status']}")
    return job["result"]


@router.get("/", response_model=List[JobStatus])
async def list_jobs(
    limit: int = 20,
    _admin=Depends(require_admin),
) -> List[JobStatus]:
    """List the most recent jobs, newest first."""
    jobs = sorted(_job_registry.values(), key=lambda j: j["queued_at"], reverse=True)
    return [JobStatus(**j) for j in jobs[:limit]]


@router.delete("/{job_id}")
async def cancel_job(
    job_id: str,
    _admin=Depends(require_admin),
) -> dict:
    """
    Mark a QUEUED job as cancelled.
    Running jobs cannot be cancelled (they complete or fail naturally).
    """
    job = _job_registry.get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    if job["status"] == "RUNNING":
        raise HTTPException(409, "Cannot cancel a running job.")
    if job["status"] in ("DONE", "FAILED"):
        raise HTTPException(409, f"Job already {job['status']}.")
    _update_job(job_id, status="CANCELLED", finished_at=_utc_now().isoformat())
    return {"status": "cancelled", "job_id": job_id}

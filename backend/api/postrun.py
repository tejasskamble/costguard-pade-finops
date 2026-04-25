"""Post-run APIs for IEEE result visibility and database import orchestration."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.auth import get_current_user, require_admin
from database import get_db_conn
from postrun.import_service import (
    CANONICAL_SEEDS,
    build_postrun_snapshot,
    import_snapshot_to_db,
    resolve_results_root,
)
from pade.model_registry import collect_model_registry

router = APIRouter(prefix="/api/postrun", tags=["postrun"])
logger = logging.getLogger(__name__)


class PostRunImportRequest(BaseModel):
    dry_run: bool = True
    results_root: Optional[str] = None
    chunk_size: int = Field(default=100_000, ge=1_000, le=1_000_000)
    min_ensemble_f1: float = Field(default=0.80, ge=0.0, le=1.0)


@router.get("/summary")
async def get_postrun_summary(
    results_root: Optional[str] = Query(default=None),
    chunk_size: int = Query(default=100_000, ge=1_000, le=1_000_000),
    min_ensemble_f1: float = Query(default=0.80, ge=0.0, le=1.0),
    _user=Depends(get_current_user),
):
    try:
        return await asyncio.to_thread(
            build_postrun_snapshot,
            results_root_override=results_root,
            chunk_size=chunk_size,
            min_ensemble_f1=min_ensemble_f1,
        )
    except Exception as exc:
        logger.exception("Post-run summary failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to build post-run summary.")


@router.get("/models")
async def get_postrun_model_registry(
    results_root: Optional[str] = Query(default=None),
    _user=Depends(get_current_user),
):
    try:
        resolved_root = resolve_results_root(results_root)
        registry = await asyncio.to_thread(
            collect_model_registry,
            results_root=resolved_root,
            seeds=CANONICAL_SEEDS,
        )
    except Exception as exc:
        logger.exception("Post-run model registry failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to collect model registry.")
    return {
        "results_root": registry.get("results_root"),
        "seed_models": registry.get("seeds", []),
    }


@router.post("/import")
async def import_postrun_results(
    request: PostRunImportRequest,
    _admin=Depends(require_admin),
    conn: asyncpg.Connection = Depends(get_db_conn),
):
    try:
        snapshot = await asyncio.to_thread(
            build_postrun_snapshot,
            results_root_override=request.results_root,
            chunk_size=request.chunk_size,
            min_ensemble_f1=request.min_ensemble_f1,
        )
        return await import_snapshot_to_db(conn, snapshot, dry_run=request.dry_run)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Post-run import failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to import post-run artifacts.")


@router.get("/import/history")
async def postrun_import_history(
    limit: int = Query(default=25, ge=1, le=250),
    _user=Depends(get_current_user),
    conn: asyncpg.Connection = Depends(get_db_conn),
):
    try:
        rows = await conn.fetch(
            """
            SELECT id, source_root, dry_run, status, summary, created_at
            FROM ieee_import_runs
            ORDER BY id DESC
            LIMIT $1
            """,
            limit,
        )
    except asyncpg.UndefinedTableError:
        return {
            "status": "migration_missing",
            "detail": "ieee_import_runs table is not available yet.",
            "rows": [],
        }
    except Exception as exc:
        logger.exception("Post-run history query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to read post-run import history.")
    return {
        "status": "ok",
        "rows": [dict(row) for row in rows],
    }


@router.get("/graphs/seed-metrics")
async def postrun_seed_metrics(
    metric_name: str = Query(default="f1_at_opt"),
    model_name: str = Query(default="ens"),
    metric_scope: str = Query(default="test"),
    _user=Depends(get_current_user),
    conn: asyncpg.Connection = Depends(get_db_conn),
):
    try:
        rows = await conn.fetch(
            """
            SELECT seed, domain, metric_value
            FROM ieee_seed_domain_metrics
            WHERE metric_name = $1
              AND model_name = $2
              AND metric_scope = $3
            ORDER BY seed ASC, domain ASC
            """,
            metric_name,
            model_name,
            metric_scope,
        )
    except asyncpg.UndefinedTableError:
        return {
            "status": "migration_missing",
            "rows": [],
            "detail": "ieee_seed_domain_metrics table is not available yet.",
        }
    except Exception as exc:
        logger.exception("Post-run seed metrics query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to read seed metrics.")
    return {
        "status": "ok",
        "metric_name": metric_name,
        "model_name": model_name,
        "metric_scope": metric_scope,
        "rows": [dict(row) for row in rows],
    }


@router.get("/graphs/domain-metrics")
async def postrun_domain_metrics(
    metric_name: str = Query(default="f1_at_opt"),
    model_name: Optional[str] = Query(default=None),
    metric_scope: str = Query(default="test"),
    _user=Depends(get_current_user),
    conn: asyncpg.Connection = Depends(get_db_conn),
):
    sql = """
        SELECT domain,
               model_name,
               metric_name,
               AVG(metric_value) AS mean_value,
               MIN(metric_value) AS min_value,
               MAX(metric_value) AS max_value,
               COUNT(*)::INT AS n
        FROM ieee_seed_domain_metrics
        WHERE metric_name = $1
          AND metric_scope = $2
          AND ($3::text IS NULL OR model_name = $3)
        GROUP BY domain, model_name, metric_name
        ORDER BY domain ASC, model_name ASC
    """
    try:
        rows = await conn.fetch(sql, metric_name, metric_scope, model_name)
    except asyncpg.UndefinedTableError:
        return {
            "status": "migration_missing",
            "rows": [],
            "detail": "ieee_seed_domain_metrics table is not available yet.",
        }
    except Exception as exc:
        logger.exception("Post-run domain metrics query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to read domain metrics.")
    return {
        "status": "ok",
        "metric_name": metric_name,
        "model_name": model_name,
        "metric_scope": metric_scope,
        "rows": [dict(row) for row in rows],
    }


@router.get("/graphs/anomaly-counts")
async def postrun_anomaly_counts(
    split_name: str = Query(default="test"),
    model_name: Optional[str] = Query(default=None),
    _user=Depends(get_current_user),
    conn: asyncpg.Connection = Depends(get_db_conn),
):
    try:
        rows = await conn.fetch(
            """
            SELECT seed,
                   domain,
                   split_name,
                   model_name,
                   total_samples,
                   anomaly_count,
                   anomaly_rate,
                   threshold,
                   mean_score
            FROM ieee_prediction_summaries
            WHERE split_name = $1
              AND ($2::text IS NULL OR model_name = $2)
            ORDER BY seed ASC, domain ASC, model_name ASC
            """,
            split_name,
            model_name,
        )
    except asyncpg.UndefinedTableError:
        return {
            "status": "migration_missing",
            "rows": [],
            "detail": "ieee_prediction_summaries table is not available yet.",
        }
    except Exception as exc:
        logger.exception("Post-run anomaly count query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to read anomaly counts.")
    return {
        "status": "ok",
        "split_name": split_name,
        "model_name": model_name,
        "rows": [dict(row) for row in rows],
    }


@router.get("/graphs/dataset-summaries")
async def postrun_dataset_summaries(
    _user=Depends(get_current_user),
    conn: asyncpg.Connection = Depends(get_db_conn),
):
    try:
        rows = await conn.fetch(
            """
            SELECT seed,
                   domain,
                   dataset_name,
                   COUNT(*)::INT AS file_count,
                   COALESCE(SUM(size_bytes), 0)::BIGINT AS total_size_bytes,
                   COALESCE(SUM(row_count), 0)::BIGINT AS total_rows,
                   COALESCE(MAX(column_count), 0)::INT AS max_columns
            FROM ieee_prepared_dataset_summaries
            GROUP BY seed, domain, dataset_name
            ORDER BY seed ASC, domain ASC, dataset_name ASC
            """
        )
    except asyncpg.UndefinedTableError:
        return {
            "status": "migration_missing",
            "rows": [],
            "detail": "ieee_prepared_dataset_summaries table is not available yet.",
        }
    except Exception as exc:
        logger.exception("Post-run dataset summary query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to read dataset summaries.")
    return {
        "status": "ok",
        "rows": [dict(row) for row in rows],
    }

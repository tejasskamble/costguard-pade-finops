"""Safe continual-learning readiness APIs (capture only, no online training)."""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, Optional

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.auth import UserProfile, get_current_user
from database import get_db_conn
from pade.inference import get_pade_runtime_status, score_pipeline
from peg.opa_client import evaluate_policy
from peg.policy_engine import extract_policy_bundle

router = APIRouter(prefix="/api/continual", tags=["continual-learning"])
logger = logging.getLogger(__name__)


class ObservationIn(BaseModel):
    run_id: Optional[str] = None
    source: str = "api"
    stage_name: str = Field(default="integration_test", min_length=1, max_length=100)
    provider: Optional[str] = Field(default=None, max_length=30)
    region: Optional[str] = Field(default=None, max_length=50)
    billed_cost: Optional[float] = None
    effective_cost: Optional[float] = None
    usage_quantity: Optional[float] = None
    usage_unit: Optional[str] = Field(default=None, max_length=40)
    branch_type: Optional[str] = Field(default="main", max_length=50)
    executor_type: Optional[str] = Field(default="github_actions", max_length=50)
    payload: Dict[str, Any] = Field(default_factory=dict)
    feedback_label: Optional[str] = Field(default=None, max_length=60)
    feedback_notes: Optional[str] = None


class FeedbackIn(BaseModel):
    observation_id: int
    label: str = Field(min_length=1, max_length=60)
    notes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


def _risk_level(crs_score: float) -> str:
    if crs_score >= 0.90:
        return "critical"
    if crs_score >= 0.75:
        return "high"
    if crs_score >= 0.50:
        return "medium"
    return "low"


def _build_stage_payload(observation: ObservationIn) -> Dict[str, Any]:
    payload = observation.payload or {}
    billed_cost = float(observation.billed_cost or payload.get("cost") or 0.0)
    return {
        "cost": billed_cost,
        "deviation": float(payload.get("cost_deviation_pct") or payload.get("deviation") or 0.0),
        "duration": float(payload.get("duration") or 0.0),
        "hist_avg_cost": float(payload.get("historical_avg_cost") or payload.get("hist_avg_cost") or 0.0),
        "hist_avg_dur": float(payload.get("historical_avg_duration") or payload.get("hist_avg_dur") or 0.0),
        "executor_enc": float(payload.get("executor_enc") or 0.0),
        "branch_enc": float(payload.get("branch_enc") or 0.0),
        "provider_enc": float(payload.get("provider_enc") or 0.0),
    }


@router.post("/observe")
async def capture_observation(
    body: ObservationIn,
    current_user: UserProfile = Depends(get_current_user),
    conn: asyncpg.Connection = Depends(get_db_conn),
):
    run_id = body.run_id or f"obs-{uuid.uuid4().hex[:12]}"
    stage_data = {body.stage_name: _build_stage_payload(body)}
    billed_cost = float(body.billed_cost or body.payload.get("cost") or 0.0)

    try:
        observation_row = await conn.fetchrow(
            """
            INSERT INTO user_uploaded_observations (
                user_id, source, run_id, stage_name, provider, region,
                billed_cost, effective_cost, usage_quantity, usage_unit,
                branch_type, executor_type, payload, pending_retraining, created_at
            )
            VALUES (
                $1, $2, $3, $4, $5, $6,
                $7, $8, $9, $10,
                $11, $12, $13::jsonb, TRUE, NOW()
            )
            RETURNING id, created_at
            """,
            current_user.id,
            body.source,
            run_id,
            body.stage_name,
            body.provider,
            body.region,
            body.billed_cost,
            body.effective_cost,
            body.usage_quantity,
            body.usage_unit,
            body.branch_type,
            body.executor_type,
            json.dumps(body.payload),
        )
    except Exception as exc:
        logger.exception("Failed to persist uploaded observation: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to persist observation payload.")

    observation_id = int(observation_row["id"])

    pade_result = await score_pipeline(
        run_id=run_id,
        stage_data=stage_data,
        stage_name=body.stage_name,
        executor_type=body.executor_type,
        branch_type=body.branch_type,
    )
    crs_score = float(pade_result.crs)
    risk_level = _risk_level(crs_score)

    try:
        policy_row = await conn.fetchrow("SELECT * FROM policy_config ORDER BY id LIMIT 1")
    except Exception as exc:
        logger.warning("Policy lookup failed for continual capture (%s); falling back to defaults.", exc)
        policy_row = None

    policy_bundle = extract_policy_bundle(dict(policy_row) if policy_row else None)
    policy_eval = await evaluate_policy(
        metrics={"crs": crs_score, "billed_cost": billed_cost},
        context={
            "stage_name": body.stage_name,
            "branch": body.branch_type or "",
            "gh_is_pr": False,
            "gh_by_core_team_member": current_user.role == "admin",
        },
        policy_bundle=policy_bundle,
    )
    opa_decision = str(policy_eval.get("decision", "ALLOW"))
    policy_source = str(policy_eval.get("policy_source") or policy_eval.get("source") or "inline")

    pade_status = get_pade_runtime_status()
    model_version = str(pade_status.get("model_type") or "canonical-gatv2-backend-adapter")
    checkpoint_path = pade_status.get("checkpoint_path")

    event_payload = {
        "input": {
            "run_id": run_id,
            "stage_name": body.stage_name,
            "provider": body.provider,
            "region": body.region,
            "billed_cost": billed_cost,
            "payload": body.payload,
        },
        "pade": {
            "crs_score": crs_score,
            "decision": pade_result.decision,
            "gat_prob": pade_result.gat_prob,
        },
        "opa": policy_eval,
    }

    try:
        event_row = await conn.fetchrow(
            """
            INSERT INTO inference_events (
                observation_id, model_version, model_checkpoint_path,
                crs_score, anomaly_score, risk_level, pade_decision,
                opa_decision, policy_source, ai_recommendation, decision_payload, created_at
            )
            VALUES (
                $1, $2, $3,
                $4, $5, $6, $7,
                $8, $9, $10, $11::jsonb, NOW()
            )
            RETURNING id, created_at
            """,
            observation_id,
            model_version,
            checkpoint_path,
            crs_score,
            crs_score,
            risk_level,
            pade_result.decision,
            opa_decision,
            policy_source,
            pade_result.ai_recommendation,
            json.dumps(event_payload),
        )
        inference_event_id = int(event_row["id"])
    except Exception as exc:
        logger.exception("Failed to persist inference event: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to persist inference event.")

    try:
        await conn.execute(
            """
            INSERT INTO retraining_queue (observation_id, status, queued_at, export_metadata)
            VALUES ($1, 'pending', NOW(), '{}'::jsonb)
            ON CONFLICT (observation_id) DO UPDATE SET
                status = 'pending',
                queued_at = COALESCE(retraining_queue.queued_at, NOW())
            """,
            observation_id,
        )
    except Exception as exc:
        logger.exception("Failed to queue observation for offline retraining: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to queue observation for retraining.")

    if body.feedback_label:
        try:
            await conn.execute(
                """
                INSERT INTO feedback_labels (
                    observation_id, inference_event_id, user_id, label, notes, metadata, created_at
                )
                VALUES ($1, $2, $3, $4, $5, '{}'::jsonb, NOW())
                """,
                observation_id,
                inference_event_id,
                current_user.id,
                body.feedback_label,
                body.feedback_notes,
            )
        except Exception as exc:
            logger.warning("Feedback write skipped for observation %s: %s", observation_id, exc)

    return {
        "status": "captured",
        "observation_id": observation_id,
        "inference_event_id": inference_event_id,
        "run_id": run_id,
        "pade_decision": pade_result.decision,
        "opa_decision": opa_decision,
        "crs_score": crs_score,
        "anomaly_score": crs_score,
        "risk_level": risk_level,
        "pending_retraining": True,
        "mode": "capture_only_no_online_training",
    }


@router.post("/feedback")
async def capture_feedback(
    body: FeedbackIn,
    current_user: UserProfile = Depends(get_current_user),
    conn: asyncpg.Connection = Depends(get_db_conn),
):
    try:
        observation_exists = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM user_uploaded_observations WHERE id = $1)",
            body.observation_id,
        )
    except Exception as exc:
        logger.exception("Feedback observation lookup failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to check observation.")
    if not observation_exists:
        raise HTTPException(status_code=404, detail="Observation not found.")

    latest_inference_id = await conn.fetchval(
        """
        SELECT id
        FROM inference_events
        WHERE observation_id = $1
        ORDER BY created_at DESC
        LIMIT 1
        """,
        body.observation_id,
    )

    try:
        row = await conn.fetchrow(
            """
            INSERT INTO feedback_labels (
                observation_id, inference_event_id, user_id, label, notes, metadata, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, NOW())
            RETURNING id, created_at
            """,
            body.observation_id,
            latest_inference_id,
            current_user.id,
            body.label,
            body.notes,
            json.dumps(body.metadata),
        )
        await conn.execute(
            """
            INSERT INTO retraining_queue (observation_id, status, queued_at, export_metadata)
            VALUES ($1, 'pending', NOW(), '{}'::jsonb)
            ON CONFLICT (observation_id) DO UPDATE SET status = 'pending'
            """,
            body.observation_id,
        )
    except Exception as exc:
        logger.exception("Feedback persistence failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to persist feedback.")

    return {
        "status": "recorded",
        "feedback_id": int(row["id"]),
        "observation_id": body.observation_id,
        "pending_retraining": True,
    }


@router.get("/retraining-readiness")
async def retraining_readiness(
    limit: int = Query(default=100, ge=1, le=1000),
    _user=Depends(get_current_user),
    conn: asyncpg.Connection = Depends(get_db_conn),
):
    try:
        counts = await conn.fetchrow(
            """
            SELECT
                COUNT(*)::INT AS total,
                COUNT(*) FILTER (WHERE status = 'pending')::INT AS pending,
                COUNT(*) FILTER (WHERE status = 'exported')::INT AS exported,
                COUNT(*) FILTER (WHERE status = 'consumed')::INT AS consumed
            FROM retraining_queue
            """
        )
        rows = await conn.fetch(
            """
            SELECT
                q.id,
                q.observation_id,
                q.status,
                q.queued_at,
                q.last_exported_at,
                o.stage_name,
                o.provider,
                o.region,
                o.billed_cost,
                o.pending_retraining,
                e.crs_score,
                e.pade_decision,
                e.opa_decision,
                e.policy_source,
                e.risk_level
            FROM retraining_queue q
            JOIN user_uploaded_observations o ON o.id = q.observation_id
            LEFT JOIN LATERAL (
                SELECT crs_score, pade_decision, opa_decision, policy_source, risk_level
                FROM inference_events ie
                WHERE ie.observation_id = q.observation_id
                ORDER BY ie.created_at DESC
                LIMIT 1
            ) e ON TRUE
            ORDER BY q.queued_at DESC
            LIMIT $1
            """,
            limit,
        )
    except asyncpg.UndefinedTableError:
        return {
            "status": "migration_missing",
            "detail": "continual-learning tables are not available yet.",
            "counts": {"total": 0, "pending": 0, "exported": 0, "consumed": 0},
            "rows": [],
        }
    except Exception as exc:
        logger.exception("Retraining readiness query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to read retraining readiness.")

    return {
        "status": "ok",
        "counts": dict(counts) if counts else {"total": 0, "pending": 0, "exported": 0, "consumed": 0},
        "rows": [dict(row) for row in rows],
        "mode": "offline_manual_retraining_only",
    }


@router.get("/retraining-export")
async def retraining_export(
    status: str = Query(default="pending"),
    limit: int = Query(default=200, ge=1, le=2000),
    _user=Depends(get_current_user),
    conn: asyncpg.Connection = Depends(get_db_conn),
):
    try:
        rows = await conn.fetch(
            """
            SELECT
                q.id AS queue_id,
                q.observation_id,
                q.status,
                q.queued_at,
                o.source,
                o.run_id,
                o.stage_name,
                o.provider,
                o.region,
                o.billed_cost,
                o.effective_cost,
                o.usage_quantity,
                o.usage_unit,
                o.branch_type,
                o.executor_type,
                o.payload,
                e.crs_score,
                e.anomaly_score,
                e.risk_level,
                e.pade_decision,
                e.opa_decision,
                e.policy_source,
                e.ai_recommendation,
                f.label AS latest_feedback_label,
                f.notes AS latest_feedback_notes,
                f.metadata AS latest_feedback_metadata
            FROM retraining_queue q
            JOIN user_uploaded_observations o ON o.id = q.observation_id
            LEFT JOIN LATERAL (
                SELECT crs_score, anomaly_score, risk_level, pade_decision, opa_decision, policy_source, ai_recommendation
                FROM inference_events ie
                WHERE ie.observation_id = q.observation_id
                ORDER BY ie.created_at DESC
                LIMIT 1
            ) e ON TRUE
            LEFT JOIN LATERAL (
                SELECT label, notes, metadata
                FROM feedback_labels fl
                WHERE fl.observation_id = q.observation_id
                ORDER BY fl.created_at DESC
                LIMIT 1
            ) f ON TRUE
            WHERE q.status = $1
            ORDER BY q.queued_at ASC
            LIMIT $2
            """,
            status,
            limit,
        )
    except asyncpg.UndefinedTableError:
        return {
            "status": "migration_missing",
            "detail": "continual-learning tables are not available yet.",
            "rows": [],
        }
    except Exception as exc:
        logger.exception("Retraining export query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to export retraining queue.")

    return {
        "status": "ok",
        "queue_status": status,
        "rows": [dict(row) for row in rows],
        "mode": "export_only_manual_offline_retraining",
    }

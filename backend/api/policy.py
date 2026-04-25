import json
import logging

import asyncpg
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth import require_admin
from database import get_db_conn
from peg.policy_engine import make_policy_response, normalize_policy_bundle

router = APIRouter(prefix="/api", tags=["policy"])
logger = logging.getLogger(__name__)

class PolicyThresholds(BaseModel):
    warn_threshold: float = Field(..., ge=0.0, le=1.0)
    auto_optimise_threshold: float = Field(..., ge=0.0, le=1.0)
    block_threshold: float = Field(..., ge=0.0, le=1.0)
    policy_bundle: dict = Field(default_factory=dict)

@router.get("/policy")
async def get_policy(
    _admin=Depends(require_admin),
    conn: asyncpg.Connection = Depends(get_db_conn),
):
    try:
        row = await conn.fetchrow("SELECT * FROM policy_config ORDER BY id LIMIT 1")
        if not row:
            row = await conn.fetchrow("""
                INSERT INTO policy_config (warn_threshold, auto_optimise_threshold, block_threshold, policy_bundle)
                VALUES (0.50, 0.75, 0.90, '{}'::jsonb)
                RETURNING *
            """)
        return make_policy_response(dict(row))
    except Exception:
        logger.exception("Error fetching policy")
        raise HTTPException(status_code=500, detail="Failed to fetch policy configuration.")

@router.put("/policy")
async def update_policy(
    thresholds: PolicyThresholds,
    _admin=Depends(require_admin),
    conn: asyncpg.Connection = Depends(get_db_conn)
):
    policy_bundle = normalize_policy_bundle(
        thresholds.policy_bundle,
        warn_threshold=thresholds.warn_threshold,
        auto_optimise_threshold=thresholds.auto_optimise_threshold,
        block_threshold=thresholds.block_threshold,
    )
    try:
        result = await conn.execute("""
            UPDATE policy_config
            SET warn_threshold = $1,
                auto_optimise_threshold = $2,
                block_threshold = $3,
                policy_bundle = $4::jsonb,
                updated_at = NOW()
            WHERE id = (SELECT MIN(id) FROM policy_config)
        """,
        thresholds.warn_threshold,
        thresholds.auto_optimise_threshold,
        thresholds.block_threshold,
        json.dumps(policy_bundle),
        )
        if result == "UPDATE 0":
            raise HTTPException(
                status_code=500,
                detail="Policy table is empty; run the database migration first."
            )
        return {
            "status": "updated",
            "thresholds": {
                "warn_threshold": thresholds.warn_threshold,
                "auto_optimise_threshold": thresholds.auto_optimise_threshold,
                "block_threshold": thresholds.block_threshold,
            },
            "policy_bundle": policy_bundle,
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error updating policy")
        raise HTTPException(status_code=500, detail="Failed to update policy configuration.")

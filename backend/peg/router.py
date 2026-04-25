"""Backend policy enforcement and remediation routes."""
import asyncio
import logging
from decimal import Decimal
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from .opa_client import evaluate_policy
from .policy_engine import extract_policy_bundle
from .optimiser import apply_optimisation, OptimisationResult, CATALOGUE, select_optimisation
from .notifier import send_slack_alert, send_email_alert
from api.auth import get_current_user, require_admin
from cache import invalidate
from config import settings
from runtime_hardening import retry_async, safe_create_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["peg"])


# ── Pydantic models ───────────────────────────────────────────────────────────

class PEGRequest(BaseModel):
    run_id: str
    stage_name: str
    crs: float
    billed_cost: Decimal = Field(..., gt=0)
    duration_seconds: float = 0.0
    latency_p95: float = 0.0
    executor_type: str = "github_actions"
    branch: str = "main"
    domain: str = "synthetic"
    gh_is_pr: bool = False
    gh_by_core_team_member: bool = False
    attribution_snapshot: Dict[str, Any] = Field(default_factory=dict)


class PEGResponse(BaseModel):
    decision: str
    action_taken: Optional[str] = None
    projected_savings_pct: Optional[int] = None
    crs: float
    run_id: str
    ai_recommendation: str = ""
    reasons: list[str] = []
    matched_rules: list[str] = []
    actions: list[str] = []
    policy_source: str = "inline"


# ── Decision endpoint ─────────────────────────────────────────────────────────

@router.post("/decision", response_model=PEGResponse)
async def get_decision(
    request: PEGRequest,
    http_request: Request,
    _admin=Depends(require_admin),
) -> PEGResponse:
    """Evaluate pipeline cost risk — returns decision, action, AI recommendation."""
    logger.info(f"Evaluating: run={request.run_id}, stage={request.stage_name}, CRS={request.crs:.3f}")
    pool = http_request.app.state.db

    # Live thresholds from DB
    try:
        async def _fetch_policy_row():
            async with pool.acquire() as conn:
                return await conn.fetchrow("SELECT * FROM policy_config ORDER BY id LIMIT 1")

        pcfg = await retry_async(
            _fetch_policy_row,
            attempts=2,
            delay=0.25,
            logger=logger,
            label="PEG policy lookup",
        )
        warn_t  = float(pcfg["warn_threshold"])
        auto_t  = float(pcfg["auto_optimise_threshold"])
        block_t = float(pcfg["block_threshold"])
    except Exception as exc:
        logger.warning("PEG policy lookup failed; using defaults: %s", exc)
        warn_t, auto_t, block_t = 0.50, 0.75, 0.90

    policy_bundle = extract_policy_bundle(pcfg if 'pcfg' in locals() else None)
    billed_cost = float(request.billed_cost)
    decision_result = await evaluate_policy(
        metrics={
            "crs": request.crs,
            "billed_cost": billed_cost,
            "duration_seconds": request.duration_seconds,
            "latency_p95": request.latency_p95,
        },
        context={
            "run_id": request.run_id,
            "stage_name": request.stage_name,
            "executor_type": request.executor_type,
            "branch": request.branch,
            "domain": request.domain,
            "gh_is_pr": request.gh_is_pr,
            "gh_by_core_team_member": request.gh_by_core_team_member,
        },
        policy_bundle=policy_bundle,
    )
    decision = decision_result["decision"]

    action_taken:     Optional[str] = None
    savings:          Optional[int] = None
    ai_recommendation: str          = ""

    if decision == "AUTO_OPTIMISE":
        try:
            opt          = apply_optimisation(request.stage_name)
            action_taken = opt.action_name
            savings      = opt.avg_savings_pct
        except Exception as exc:
            logger.warning(f"Optimisation selection failed: {exc}")

    if decision in ("WARN", "AUTO_OPTIMISE", "BLOCK"):
        try:
            from pade.inference import generate_anomaly_recommendation
            ai_recommendation = await generate_anomaly_recommendation(
                decision=decision,
                stage_name=request.stage_name,
                crs=request.crs,
                cost=billed_cost,
                action=action_taken,
            )
        except Exception as exc:
            logger.warning(f"AI recommendation failed: {exc}")
            ai_recommendation = (
                f"CRS={request.crs:.3f} on '{request.stage_name}'. "
                f"Recommended: {action_taken or 'review resource allocation'}."
            )

    # Persist decision to DB
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE cost_attribution
                SET pade_decision = $1, crs_score = $2, ai_recommendation = $3
                WHERE run_id = $4 AND stage_name = $5
                """,
                decision, request.crs, ai_recommendation,
                request.run_id, request.stage_name,
            )
    except Exception as exc:
        logger.warning(f"DB decision update skipped: {exc}")

    invalidate("alerts")   # FEATURE-4: invalidate cache after write

    # Look up owner email (GAP-5)
    owner_email: Optional[str] = None
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT u.email FROM pipeline_runs pr
                JOIN users u ON pr.user_id = u.id
                WHERE pr.run_id = $1
                """,
                request.run_id,
            )
        owner_email = row["email"] if row else settings.SMTP_USER
    except Exception as exc:
        logger.debug("Owner email lookup failed; using configured fallback: %s", exc, exc_info=True)
        owner_email = settings.SMTP_USER

    # Fire notifications
    if decision in ("WARN", "AUTO_OPTIMISE", "BLOCK"):
        if settings.SLACK_BOT_TOKEN and settings.SLACK_DEFAULT_CHANNEL:
            safe_create_task(
                send_slack_alert(
                    decision=decision, run_id=request.run_id,
                    crs=request.crs, stage_name=request.stage_name,
                    cost=request.billed_cost, optimisation=action_taken,
                    ai_recommendation=ai_recommendation,
                ),
                logger=logger,
                label="slack notification",
            )
        if owner_email and all([settings.SMTP_HOST, settings.SMTP_USER, settings.SMTP_PASSWORD]):
            safe_create_task(
                send_email_alert(
                    decision=decision, run_id=request.run_id,
                    crs=request.crs, stage_name=request.stage_name,
                    cost=request.billed_cost,
                    recipient_email=owner_email,
                    optimisation=action_taken,
                    ai_recommendation=ai_recommendation,
                ),
                logger=logger,
                label="email notification",
            )

    return PEGResponse(
        decision=decision,
        action_taken=action_taken,
        projected_savings_pct=savings,
        crs=request.crs,
        run_id=request.run_id,
        ai_recommendation=ai_recommendation,
        reasons=list(decision_result.get("reasons", [])),
        matched_rules=list(decision_result.get("matched_rules", [])),
        actions=list(decision_result.get("actions", [])),
        policy_source=str(decision_result.get("policy_source", "inline")),
    )


# ── FEATURE-5: Remediation YAML download ─────────────────────────────────────

def _build_remediation_yaml(
    stage_name: str,
    run_id: str,
    ai_recommendation: str,
) -> str:
    """
    Generate a GitHub Actions workflow YAML patch tailored to the stage_name.
    Uses the CATALOGUE from peg/optimiser.py to select the right action.
    """
    action = select_optimisation(stage_name)
    action_name = action["name"] if action else "scope_limited_test_suite"
    description = action["description"] if action else "Run only changed-file tests"
    savings     = action["avg_savings_pct"] if action else 55

    # Build stage-specific patch
    stage_steps = {
        "build": """
      - name: Cache dependencies (CostGuard Fix)
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
          restore-keys: ${{ runner.os }}-pip-""",
        "unit_test": """
      - name: Scope-limited test suite (CostGuard Fix)
        run: |
          CHANGED=$(git diff --name-only HEAD~1 HEAD | grep '.py' | tr '\\n' ' ')
          pytest --co -q --rootdir=. -k "$(echo $CHANGED | xargs -I{} basename {} .py | tr ' ' ' or ')" || pytest -x""",
        "integration_test": """
      - name: Reduced parallelism integration test (CostGuard Fix)
        run: pytest tests/integration/ -n 2 --dist=loadscope  # was -n auto""",
        "deploy_prod": """
      - name: Switch to spot instance runner (CostGuard Fix)
        # Add to your workflow's 'runs-on' config:
        # runs-on: [self-hosted, spot]
        # Or for GitHub-hosted: use ubuntu-latest (already spot-based)
        run: echo "Runner type optimised — check self-hosted runner config" """,
        "deploy_staging": """
      - name: Switch to preemptible runner (CostGuard Fix)
        run: echo "Configure GCP Preemptible or AWS Spot in runner labels" """,
    }
    stage_step = stage_steps.get(stage_name, f"""
      - name: Cost optimisation (CostGuard Fix — {description})
        run: echo "Apply: {description}" """)

    return f"""# CostGuard Auto-Remediation Patch
# Generated for run: {run_id[:12]}...
# Stage: {stage_name}
# Decision: BLOCK
# Recommended action: {action_name} (est. {savings}% cost reduction)
#
# AI Recommendation:
# {ai_recommendation.replace(chr(10), chr(10) + '# ')}
#
# Apply this patch to your .github/workflows/*.yml file.

name: CostGuard Optimised Pipeline
on: [push, pull_request]

jobs:
  {stage_name.replace('_', '-')}:
    name: {stage_name.replace('_', ' ').title()} (Optimised)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 2
{stage_step}

# ─── CostGuard Metadata ───────────────────────────────────────────
# run_id:     {run_id}
# stage:      {stage_name}
# action:     {action_name}
# savings:    ~{savings}%
# generated:  costguard-pade-v17.0
"""


@router.get("/remediate/{run_id}")
async def generate_remediation(
    run_id: str,
    http_request: Request,
    current_user=Depends(get_current_user),
) -> Response:
    """
    FEATURE-5: Returns a downloadable GitHub Actions YAML patch for a BLOCK decision.
    VG-7: Dashboard shows [⬇ Download Fix YAML] button for BLOCK alerts.
    """
    pool = http_request.app.state.db
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT stage_name, pade_decision, ai_recommendation
                FROM cost_attribution
                WHERE run_id = $1 AND pade_decision = 'BLOCK'
                ORDER BY created_at DESC LIMIT 1
                """,
                run_id,
            )
    except Exception as exc:
        logger.exception("Remediation lookup failed for run %s: %s", run_id, exc)
        raise HTTPException(500, "Failed to generate remediation artifact.")

    if not row:
        raise HTTPException(404, "No BLOCK decision found for this run_id.")

    yaml_content = _build_remediation_yaml(
        stage_name=row["stage_name"],
        run_id=run_id,
        ai_recommendation=row["ai_recommendation"] or "",
    )
    return Response(
        content=yaml_content,
        media_type="application/x-yaml",
        headers={
            "Content-Disposition": f"attachment; filename=costguard-fix-{run_id[:8]}.yml"
        },
    )

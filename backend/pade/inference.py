"""Backend PADE inference endpoints and helpers."""
from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel

from api.auth import require_admin
from config import settings

from .checkpoint_resolver import CheckpointResolver, get_resolver
from .ensemble import CRSResult, classify_crs, compute_crs
from .feature_builder import STAGE_ORDER, build_pipeline_graph
from .gat_model import PipelineGAT

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/pade", tags=["pade"])

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
ACTIVE_CHECKPOINT_PATH = CHECKPOINT_DIR / "gat_best.pt"
MIN_CHECKPOINT_BYTES = 1024 * 1024


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp_path), str(path))
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

if _HAS_TORCH:
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    _device = "cpu"

_gat_model: Optional[PipelineGAT] = None
_model_loaded_at: Optional[datetime] = None
_resolved_checkpoint_path: Optional[Path] = None
_checkpoint_source: str = "bootstrap"
_checkpoint_mode: Optional[str] = None
_last_checkpoint_error: Optional[str] = None
_checkpoint_resolver: CheckpointResolver = get_resolver()

NODE_POSITIONS: Dict[str, tuple] = {
    "checkout": (0.50, 0.90),
    "build": (0.30, 0.70),
    "unit_test": (0.70, 0.70),
    "integration_test": (0.50, 0.50),
    "security_scan": (0.30, 0.30),
    "docker_build": (0.70, 0.30),
    "deploy_staging": (0.40, 0.10),
    "deploy_prod": (0.60, 0.10),
}
DAG_EDGES = [
    ("checkout", "build"),
    ("checkout", "unit_test"),
    ("build", "integration_test"),
    ("unit_test", "integration_test"),
    ("integration_test", "security_scan"),
    ("integration_test", "docker_build"),
    ("security_scan", "deploy_staging"),
    ("docker_build", "deploy_staging"),
    ("deploy_staging", "deploy_prod"),
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _candidate_modes() -> List[str]:
    modes: List[str] = []
    for value in (settings.PADE_DATA_MODE, "synthetic", "real", "bitbrains"):
        cleaned = (value or "").strip().lower()
        if cleaned and cleaned not in modes:
            modes.append(cleaned)
    return modes or ["synthetic"]


def _active_seed() -> Optional[int]:
    raw = os.getenv("COSTGUARD_ACTIVE_SEED", "").strip()
    if not raw:
        return None
    try:
        parsed = int(raw)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _load_checkpoint_state(checkpoint_path: Path) -> dict:
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is not installed.")
    try:
        return torch.load(checkpoint_path, map_location=_device, weights_only=True)
    except TypeError:  # pragma: no cover - compatibility for older torch
        return torch.load(checkpoint_path, map_location=_device)


def _build_model_from_checkpoint(checkpoint_path: Path) -> Optional[PipelineGAT]:
    if not _HAS_TORCH:
        return None
    try:
        model = PipelineGAT(in_channels=8)
        state_dict = _load_checkpoint_state(checkpoint_path)
        model.load_state_dict(state_dict, strict=False)
        model.to(_device)
        model.eval()
        return model
    except Exception as exc:
        logger.exception("Checkpoint load failed for %s: %s", checkpoint_path, exc)
        return None


def _build_bootstrap_model() -> Optional[PipelineGAT]:
    if not _HAS_TORCH:
        return None
    model = PipelineGAT(in_channels=8)
    model.to(_device)
    model.eval()
    return model


def _detect_checkpoint_source(path: Path, mode: str, seed: Optional[int] = None) -> str:
    try:
        latest_run = _checkpoint_resolver.get_latest_run_dir(mode, seed=seed)
        if latest_run is not None and path.resolve() == (latest_run / "checkpoints" / "gat_best.pt").resolve():
            return "run-scan"
    except Exception:
        pass

    manifest = _checkpoint_resolver.get_manifest(mode, seed=seed)
    manifest_path = manifest.get("gat_checkpoint") if manifest else None
    if manifest_path:
        candidate = Path(str(manifest_path))
        if not candidate.is_absolute():
            candidate = Path(__file__).resolve().parent.parent.parent / candidate
        try:
            if candidate.resolve() == path.resolve():
                return "manifest"
        except Exception:
            pass

    if path.parent == CHECKPOINT_DIR:
        return "bundled"
    return "resolved"


def _resolve_gat_checkpoint() -> tuple[Optional[Path], str, Optional[str]]:
    seed = _active_seed()
    for mode in _candidate_modes():
        path = _checkpoint_resolver.get_gat_checkpoint(mode, seed=seed)
        if path is not None:
            return path, _detect_checkpoint_source(path, mode, seed=seed), mode
    return None, "bootstrap", None


def _set_runtime_model(
    model: Optional[PipelineGAT],
    *,
    checkpoint_path: Optional[Path],
    source: str,
    mode: Optional[str],
    error: Optional[str] = None,
) -> None:
    global _gat_model, _model_loaded_at, _resolved_checkpoint_path, _checkpoint_source, _checkpoint_mode, _last_checkpoint_error
    _gat_model = model
    _model_loaded_at = _utc_now() if model is not None else None
    _resolved_checkpoint_path = checkpoint_path
    _checkpoint_source = source
    _checkpoint_mode = mode
    _last_checkpoint_error = error


def bootstrap_gat_checkpoint(checkpoint_dir: Path = CHECKPOINT_DIR) -> None:
    """Best-effort GAT bootstrap; never raises during API startup."""
    del checkpoint_dir  # resolver is the authority now
    if not _HAS_TORCH:
        logger.warning("PyTorch not installed; PADE inference will use neutral fallback scores.")
        _set_runtime_model(
            None,
            checkpoint_path=None,
            source="torch-missing",
            mode=None,
            error="PyTorch is not installed.",
        )
        return

    resolved_path, source, mode = _resolve_gat_checkpoint()
    if resolved_path is not None:
        model = _build_model_from_checkpoint(resolved_path)
        if model is not None:
            logger.info("PADE GAT checkpoint loaded from %s (%s, mode=%s)", resolved_path, source, mode)
            _set_runtime_model(
                model,
                checkpoint_path=resolved_path,
                source=source,
                mode=mode,
            )
            return
        logger.warning("Resolved checkpoint could not be loaded; falling back to bootstrap weights.")

    bootstrap_model = _build_bootstrap_model()
    _set_runtime_model(
        bootstrap_model,
        checkpoint_path=None,
        source="bootstrap",
        mode=mode,
        error="No valid trained checkpoint could be loaded." if resolved_path is None else "Resolved checkpoint load failed.",
    )


def load_gat_model(in_channels: int = 8) -> Optional[PipelineGAT]:
    """Return the cached GAT model when available."""
    del in_channels
    global _gat_model
    if _gat_model is not None:
        return _gat_model
    bootstrap_gat_checkpoint()
    return _gat_model


def reload_gat_model(
    checkpoint_path: Path,
    *,
    source: str = "uploaded",
    mode: Optional[str] = "manual-upload",
) -> bool:
    """Hot-reload the backend inference checkpoint."""
    if not _HAS_TORCH or not checkpoint_path.exists():
        return False
    if checkpoint_path.stat().st_size < MIN_CHECKPOINT_BYTES:
        logger.warning("Rejected checkpoint %s because it is smaller than %s bytes.", checkpoint_path, MIN_CHECKPOINT_BYTES)
        return False

    model = _build_model_from_checkpoint(checkpoint_path)
    if model is None:
        return False

    _set_runtime_model(
        model,
        checkpoint_path=checkpoint_path,
        source=source,
        mode=mode,
    )
    return True


def get_pade_runtime_status() -> dict:
    checkpoint_exists = bool(_resolved_checkpoint_path and _resolved_checkpoint_path.exists())
    bootstrap_mode = _checkpoint_source in {"bootstrap", "torch-missing"}
    status = "ok" if (_gat_model is not None and not bootstrap_mode) else "degraded"
    return {
        "status": status,
        "model_loaded": _gat_model is not None,
        "device": str(_device),
        "checkpoint_exists": checkpoint_exists,
        "checkpoint_path": str(_resolved_checkpoint_path) if _resolved_checkpoint_path else None,
        "checkpoint_source": _checkpoint_source,
        "checkpoint_mode": _checkpoint_mode,
        "bootstrap_mode": bootstrap_mode,
        "loaded_at": _model_loaded_at.isoformat() if _model_loaded_at else None,
        "inference_ready": _gat_model is not None or not _HAS_TORCH,
        "model_type": "canonical-gatv2-backend-adapter",
        "crs_ready": True,
        "last_error": _last_checkpoint_error,
    }


async def generate_anomaly_recommendation(
    decision: str,
    stage_name: str,
    crs: float,
    cost: float,
    action: Optional[str] = None,
) -> str:
    """Generate an actionable recommendation with an LLM fallback to static text."""
    fallback = (
        f"Stage '{stage_name}' recorded CRS={crs:.3f} with billed cost ${cost:.4f}. "
        f"Recommended action: {action or 'review resource allocation and pipeline concurrency'}. "
        "Expected cost reduction on the next run is typically 30% to 65%."
    )
    if not settings.OPENAI_API_KEY:
        return fallback
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        prompt = (
            f"You are a FinOps reviewer. Stage={stage_name}. Decision={decision}. "
            f"CRS={crs:.3f}. Cost=${cost:.4f}. Suggested action={action or 'review resource allocation'}. "
            "Write exactly three sentences covering impact, remediation, and expected savings."
        )
        resp = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=180,
            temperature=0.3,
        )
        text = resp.choices[0].message.content.strip()
        return text or fallback
    except Exception as exc:
        logger.warning("OpenAI recommendation failed; using fallback (%s)", exc)
        return fallback


async def score_pipeline(
    run_id: str,
    stage_data: Dict[str, Any],
    deviation_sequence: Optional[List[float]] = None,
    stage_name: Optional[str] = None,
    executor_type: Optional[str] = None,
    branch_type: Optional[str] = None,
) -> CRSResult:
    """Compute a backend CRS score with canonical-model fallback semantics."""
    del deviation_sequence, executor_type, branch_type
    gat_prob = 0.5
    model = load_gat_model(in_channels=8)
    if _HAS_TORCH and model is not None:
        try:
            graph = build_pipeline_graph(stage_data).to(_device)
            batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=_device)
            with torch.no_grad():
                logit = model(graph.x, graph.edge_index, batch)
                gat_prob = float(torch.sigmoid(logit).item())
        except Exception as exc:
            logger.warning("GAT inference failed for run %s: %s", run_id, exc)
            gat_prob = 0.5

    crs = compute_crs(gat_prob)
    decision = classify_crs(crs)
    ai_rec = ""
    if decision in {"WARN", "AUTO_OPTIMISE", "BLOCK"}:
        resolved_stage = stage_name or "integration_test"
        resolved_cost = float(stage_data.get(resolved_stage, {}).get("cost", 0.0))
        ai_rec = await generate_anomaly_recommendation(decision, resolved_stage, crs, resolved_cost)

    return CRSResult(
        crs=crs,
        decision=decision,
        gat_prob=gat_prob,
        lstm_prob=0.0,
        stage_data={"run_id": run_id},
        ai_recommendation=ai_rec,
    )


class CheckpointUploadResponse(BaseModel):
    status: str
    message: str
    checkpoint_path: Optional[str] = None


@router.post("/load-checkpoint", response_model=CheckpointUploadResponse)
async def load_checkpoint(
    file: UploadFile = File(...),
    _admin=Depends(require_admin),
):
    if not _HAS_TORCH:
        raise HTTPException(503, "Checkpoint loading is unavailable because PyTorch is not installed.")
    if not file.filename or not file.filename.endswith(".pt"):
        raise HTTPException(400, "Only .pt PyTorch checkpoint files are accepted.")

    file_bytes = await file.read()
    if len(file_bytes) < MIN_CHECKPOINT_BYTES:
        raise HTTPException(
            400,
            f"Checkpoint is too small. Expected at least {MIN_CHECKPOINT_BYTES} bytes.",
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = CHECKPOINT_DIR / f".upload-{uuid.uuid4().hex}.pt"
    try:
        _atomic_write_bytes(temp_path, file_bytes)
        model = _build_model_from_checkpoint(temp_path)
        if model is None:
            raise HTTPException(422, "Uploaded checkpoint could not be loaded into the backend GAT model.")

        os.replace(str(temp_path), str(ACTIVE_CHECKPOINT_PATH))
        _set_runtime_model(
            model,
            checkpoint_path=ACTIVE_CHECKPOINT_PATH,
            source="uploaded",
            mode="manual-upload",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Checkpoint upload failed: %s", exc)
        raise HTTPException(500, "Failed to store the uploaded checkpoint.")
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    return CheckpointUploadResponse(
        status="loaded",
        message="Model loaded successfully.",
        checkpoint_path=str(ACTIVE_CHECKPOINT_PATH),
    )


@router.get("/status")
async def pade_status():
    return get_pade_runtime_status()


@router.get("/dag")
async def get_dag_visualization(
    run_id: Optional[str] = Query(None, description="Pipeline run ID; omit for latest run"),
    http_request: Request = None,
) -> dict:
    pool = http_request.app.state.db
    try:
        async with pool.acquire() as conn:
            if run_id:
                rows = await conn.fetch(
                    """
                    SELECT DISTINCT ON (stage_name)
                           stage_name,
                           COALESCE(crs_score, 0.5)         AS crs_score,
                           COALESCE(billed_cost, 0.0)       AS billed_cost,
                           COALESCE(pade_decision, 'ALLOW') AS pade_decision,
                           COALESCE(ai_recommendation, '')  AS ai_recommendation
                    FROM cost_attribution
                    WHERE run_id = $1
                    ORDER BY stage_name, created_at DESC
                    """,
                    run_id,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT DISTINCT ON (stage_name)
                           stage_name,
                           COALESCE(crs_score, 0.5)         AS crs_score,
                           COALESCE(billed_cost, 0.0)       AS billed_cost,
                           COALESCE(pade_decision, 'ALLOW') AS pade_decision,
                           COALESCE(ai_recommendation, '')  AS ai_recommendation
                    FROM cost_attribution
                    ORDER BY stage_name, created_at DESC
                    """
                )
    except Exception as exc:
        logger.warning("DAG DB fetch failed: %s", exc)
        rows = []

    stage_metrics = {row["stage_name"]: dict(row) for row in rows}
    nodes = []
    for stage in STAGE_ORDER:
        pos = NODE_POSITIONS.get(stage, (0.5, 0.5))
        metrics = stage_metrics.get(stage, {})
        crs = float(metrics.get("crs_score", 0.5))
        billed_cost = float(metrics.get("billed_cost", 0.0))
        decision = metrics.get("pade_decision", "ALLOW")
        ai_rec = metrics.get("ai_recommendation", "")
        nodes.append(
            {
                "id": stage,
                "label": stage.replace("_", "\n"),
                "crs": crs,
                "x": pos[0],
                "y": pos[1],
                "billed_cost": billed_cost,
                "decision": decision,
                "ai_rec": ai_rec,
            }
        )
    edges = [{"src": src, "dst": dst} for src, dst in DAG_EDGES]
    return {"nodes": nodes, "edges": edges, "run_id": run_id}

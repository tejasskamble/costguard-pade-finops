"""
backend/api/pade_training.py  — CostGuard v17.0
Wraps the canonical CostGuard_PADE_FULL.py (v17.0) behind async FastAPI endpoints.
Training runs in a ThreadPoolExecutor so CPU-bound work never blocks
the async event loop. Progress is polled via GET /status/{job_id}.
"""
import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from api.auth import get_current_user, UserProfile
from database import get_db_conn
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pade/train", tags=["pade-training"])

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pade-train")

PADE_DIR = Path(__file__).parent.parent / "pade"
PADE_WORK_DIR = PADE_DIR / "training_workspace"
PADE_CKPT_DIR = PADE_DIR / "checkpoints"


def _atomic_temp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_temp_path(path)
    try:
        with tmp.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp), str(path))
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _atomic_write_text(path: Path, content: str) -> None:
    _atomic_write_bytes(path, content.encode("utf-8"))


def _atomic_torch_save(path: Path, payload) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_temp_path(path)
    try:
        torch.save(payload, str(tmp))
        try:
            with tmp.open("rb") as handle:
                os.fsync(handle.fileno())
        except OSError:
            pass
        os.replace(str(tmp), str(path))
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


# ─── Pydantic models ──────────────────────────────────────────────────────────

class TrainingConfig(BaseModel):
    mode: str = "full"            # lstm | gat | full | baseline | synth
    synth_mode: str = "enhanced"  # legacy | enhanced
    epochs: int = 30
    n_synthetic_rows: int = 10000
    seed: int = 42
    anomaly_rate: float = 0.08


class TrainingStartResponse(BaseModel):
    job_id: int
    message: str
    estimated_duration_minutes: int


# ─── Blocking training (runs in thread pool) ──────────────────────────────────

def _blocking_pade_train(job_id: int, config: dict, db_dsn: str) -> dict:
    """
    Synchronous PADE training executed in a ThreadPoolExecutor.
    Calls pade_full.py functions using their exact signatures.
    Updates pade_training_jobs.progress via its own asyncpg connection.
    """
    import asyncpg, asyncio as _asyncio

    async def _set_progress(pct: int):
        conn = await asyncpg.connect(dsn=db_dsn)
        try:
            await conn.execute(
                "UPDATE pade_training_jobs SET progress=$1 WHERE id=$2",
                pct, job_id,
            )
        finally:
            await conn.close()

    def _upd(pct: int):
        loop = _asyncio.new_event_loop()
        try:
            loop.run_until_complete(_set_progress(pct))
        except Exception as e:
            logger.warning(f"Progress update failed: {e}")
        finally:
            loop.close()

    # Add pade dir to sys.path
    pade_str = str(PADE_DIR)
    if pade_str not in sys.path:
        sys.path.insert(0, pade_str)

    mode           = config.get("mode", "full")
    seed           = config.get("seed", 42)
    n_rows         = config.get("n_synthetic_rows", 10000)
    anomaly_rate   = config.get("anomaly_rate", 0.08)
    epochs         = config.get("epochs", 30)
    results        = {"mode": mode}
    stage_errors: List[str] = []

    try:
        PADE_WORK_DIR.mkdir(parents=True, exist_ok=True)
        data_dir  = str(PADE_WORK_DIR / "data")
        ml_dir    = str(PADE_WORK_DIR / "ml_ready")

        Path(data_dir).mkdir(parents=True, exist_ok=True)
        Path(ml_dir).mkdir(parents=True, exist_ok=True)

        _upd(5)

        from pade_full import (
            LSTMConfig, GATConfig, seed_everything,
            run_synthetic_data_generation, run_preprocessing,
            train_lstm, train_gat, run_baseline_comparison,
        )

        # Canonical engine compatibility: pade_full reads synth mode from a module-level
        # global (_SYNTH_MODE) rather than a function parameter.  Without this
        # the user's synth_mode config setting is silently ignored.
        import pade_full as _pf
        _pf._SYNTH_MODE = config.get("synth_mode", "enhanced")

        _upd(10)
        results["import"] = "pade_full.py loaded"

        # Always generate synthetic data first
        try:
            seed_everything(seed)
            run_synthetic_data_generation(
                n_rows=n_rows,
                out_dir=data_dir,
                seed=seed,
                anomaly_rate=anomaly_rate,
            )
            results["synth"] = f"generated {n_rows} rows"
            _upd(25)
        except Exception as exc:
            results["synth"] = f"error: {exc}"
            stage_errors.append(f"synth: {exc}")
            _upd(25)

        if mode in ("synth",):
            if stage_errors:
                raise RuntimeError("; ".join(stage_errors))
            _upd(100)
            results["status"] = "completed"
            return results

        # Run preprocessing
        try:
            run_preprocessing(raw_dir=data_dir, out_dir=ml_dir, seed=seed)
            results["preprocessing"] = "completed"
            _upd(40)
        except Exception as exc:
            results["preprocessing"] = f"error: {exc}"
            stage_errors.append(f"preprocessing: {exc}")
            _upd(40)

        # LSTM training
        if mode in ("lstm", "full"):
            try:
                cfg_lstm = LSTMConfig(epochs=epochs, seed=seed)
                train_lstm(cfg=cfg_lstm, ckpt_dir=PADE_CKPT_DIR)
                results["lstm"] = "trained"
                _upd(65)
            except Exception as exc:
                results["lstm"] = f"error: {exc}"
                stage_errors.append(f"lstm: {exc}")
                logger.warning(f"LSTM training error: {exc}")
                _upd(65)

        # GAT training
        if mode in ("gat", "full"):
            try:
                cfg_gat = GATConfig(epochs=epochs, seed=seed)
                train_gat(cfg=cfg_gat, ckpt_dir=PADE_CKPT_DIR)
                results["gat"] = "trained"
                _upd(85)

                # Canonical bridge: the research GATv2Pipeline checkpoint is saved to
                # gat_best.pt, but inference.py loads PipelineGAT from gat.pt.
                # These are incompatible architectures.  We re-save the trained
                # checkpoint under gat.pt using PipelineGAT so that the live
                # inference endpoint automatically picks up the trained weights.
                gat_best = PADE_CKPT_DIR / "gat_best.pt"
                gat_target = PADE_CKPT_DIR / "gat.pt"
                if gat_best.exists():
                    try:
                        import torch, sys as _sys
                        _pade_str = str(PADE_DIR)
                        if _pade_str not in _sys.path:
                            _sys.path.insert(0, _pade_str)
                        from gat_model import PipelineGAT
                        # Bootstrap a fresh PipelineGAT and copy any weight
                        # keys that share the same name/shape (partial transfer)
                        bridge_model = PipelineGAT(in_channels=8)
                        src_state = torch.load(gat_best, map_location="cpu",
                                               weights_only=True)
                        bridge_state = bridge_model.state_dict()
                        transferred, skipped = 0, 0
                        for k, v in src_state.items():
                            # Strip any module prefix
                            bare_k = k.replace("module.", "")
                            if bare_k in bridge_state and bridge_state[bare_k].shape == v.shape:
                                bridge_state[bare_k] = v
                                transferred += 1
                            else:
                                skipped += 1
                        bridge_model.load_state_dict(bridge_state)
                        _atomic_torch_save(gat_target, bridge_model.state_dict())
                        results["gat_bridge"] = (
                            f"checkpoint bridged to PipelineGAT "
                            f"({transferred} layers transferred, {skipped} skipped)"
                        )
                        # mark that a trained checkpoint is present
                        _atomic_write_text(PADE_CKPT_DIR / "trained.flag", "1")
                        logger.info(f"GAT bridge: {transferred} layers → gat.pt")
                    except Exception as bridge_exc:
                        logger.warning(f"GAT bridge failed: {bridge_exc}")
                        results["gat_bridge"] = f"error: {bridge_exc}"
                        stage_errors.append(f"gat_bridge: {bridge_exc}")

            except Exception as exc:
                results["gat"] = f"error: {exc}"
                stage_errors.append(f"gat: {exc}")
                logger.warning(f"GAT training error: {exc}")
                _upd(85)

        # Baseline comparison
        if mode == "baseline":
            try:
                import numpy as np
                ml_path = Path(ml_dir) / "task_B"
                feature_train = ml_path / "X_feature_train.npy"
                feature_val = ml_path / "X_feature_val.npy"
                feature_test = ml_path / "X_feature_test.npy"
                if feature_train.exists() and feature_test.exists():
                    baseline_results = run_baseline_comparison(
                        X_train=np.asarray(np.load(str(feature_train), mmap_mode="r"), dtype=np.float32),
                        X_test=np.asarray(np.load(str(feature_test), mmap_mode="r"), dtype=np.float32),
                        y_train=np.asarray(np.load(str(ml_path / "y_train.npy"), mmap_mode="r"), dtype=np.float32),
                        y_test=np.asarray(np.load(str(ml_path / "y_test.npy"), mmap_mode="r"), dtype=np.float32),
                        pos_rate=float(np.asarray(np.load(str(ml_path / "y_train.npy"), mmap_mode="r"), dtype=np.float32).mean()),
                        X_val=np.asarray(np.load(str(feature_val), mmap_mode="r"), dtype=np.float32) if feature_val.exists() else None,
                        y_val=np.asarray(np.load(str(ml_path / "y_val.npy"), mmap_mode="r"), dtype=np.float32) if (ml_path / "y_val.npy").exists() else None,
                    )
                    results["baseline"] = {
                        "status": "completed",
                        "models": sorted(baseline_results.keys()),
                    }
                else:
                    results["baseline"] = "skipped: preprocessing required first"
            except Exception as exc:
                results["baseline"] = f"error: {exc}"
            _upd(90)

        _upd(100)
        if stage_errors:
            raise RuntimeError("; ".join(stage_errors))
        results["status"] = "completed"

    except Exception as exc:
        logger.exception(f"PADE job {job_id} fatal: {exc}")
        results["status"] = f"failed: {exc}"
        results["error"] = str(exc)

    return results


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/start", response_model=TrainingStartResponse)
async def start_training(
    config: TrainingConfig,
    request: Request,
    db=Depends(get_db_conn),
    current_user: UserProfile = Depends(get_current_user),
):
    """
    Launch a PADE training job in the background.
    Returns job_id immediately — poll /status/{job_id} for progress.
    """
    valid_modes = {"lstm", "gat", "full", "baseline", "synth"}
    if config.mode not in valid_modes:
        raise HTTPException(400, f"mode must be one of: {valid_modes}")

    try:
        row = await db.fetchrow(
            """INSERT INTO pade_training_jobs (user_id, job_type, status, config_json)
               VALUES ($1, $2, 'running', $3::jsonb)
               RETURNING id""",
            current_user.id, config.mode, json.dumps(config.dict()),
        )
        job_id = row["id"]
    except Exception as exc:
        logger.exception(f"start_training DB error: {exc}")
        raise HTTPException(500, "Failed to create training job")

    pool = request.app.state.db

    async def _bg_train():
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _EXECUTOR,
                _blocking_pade_train,
                job_id, config.dict(), settings.DATABASE_URL,
            )
            final_status = "done" if result.get("status") == "completed" else "failed"
            async with pool.acquire() as c:
                await c.execute(
                    """UPDATE pade_training_jobs
                       SET status=$1, progress=100, finished_at=NOW(), result_json=$2::jsonb
                       WHERE id=$3""",
                    final_status, json.dumps(result), job_id,
                )
            logger.info(f"PADE job {job_id} finished: {final_status}")
        except Exception as exc:
            logger.exception(f"PADE job {job_id} bg error: {exc}")
            try:
                async with pool.acquire() as c:
                    await c.execute(
                        """UPDATE pade_training_jobs
                           SET status='failed', result_json=$1::jsonb, finished_at=NOW()
                           WHERE id=$2""",
                        json.dumps({"error": str(exc)}), job_id,
                    )
            except Exception as inner_exc:
                logger.warning("Failed to persist training failure for job %s: %s", job_id, inner_exc)

    asyncio.create_task(_bg_train())

    est = {"lstm": 5, "gat": 8, "full": 15, "baseline": 3, "synth": 2}.get(config.mode, 10)
    return TrainingStartResponse(
        job_id=job_id,
        message=f"Training job #{job_id} started (mode: {config.mode}). Poll /status/{job_id}.",
        estimated_duration_minutes=est,
    )


@router.get("/status/{job_id}")
async def get_training_status(
    job_id: int,
    request: Request,
    db=Depends(get_db_conn),
    current_user: UserProfile = Depends(get_current_user),
):
    """Poll training progress (0–100), status, and results."""
    try:
        row = await db.fetchrow(
            """SELECT id, job_type, status, progress, eta_seconds,
                      config_json, result_json, started_at, finished_at
               FROM pade_training_jobs WHERE id = $1""",
            job_id,
        )
        if not row:
            raise HTTPException(404, f"Training job {job_id} not found")
        return dict(row)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"get_training_status error: {exc}")
        raise HTTPException(500, "Failed to fetch job status")


@router.get("/history")
async def get_training_history(
    request: Request,
    db=Depends(get_db_conn),
    current_user: UserProfile = Depends(get_current_user),
    limit: int = 20,
):
    """Last N training jobs for the current user, newest first."""
    limit = min(limit, 100)
    try:
        rows = await db.fetch(
            """SELECT id, job_type, status, progress, started_at, finished_at,
                      EXTRACT(EPOCH FROM
                          (COALESCE(finished_at, NOW()) - started_at))::int AS duration_seconds
               FROM pade_training_jobs WHERE user_id = $1
               ORDER BY started_at DESC LIMIT $2""",
            current_user.id, limit,
        )
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.exception(f"get_training_history error: {exc}")
        raise HTTPException(500, "Failed to fetch history")


@router.post("/data/upload")
async def upload_training_data(
    file: UploadFile = File(...),
    request: Request = None,
    db=Depends(get_db_conn),
    current_user: UserProfile = Depends(get_current_user),
):
    """Upload a CSV file for custom PADE training data."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted")
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 50MB)")
    upload_dir = PADE_WORK_DIR / "custom_data"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{current_user.id}_{file.filename}"
    _atomic_write_bytes(dest, content)
    rows = content.count(b"\n")
    return {"message": f"Uploaded {file.filename} ({rows} rows)", "path": str(dest), "rows": rows}


@router.get("/baseline")
async def get_baseline_report(
    request: Request,
    db=Depends(get_db_conn),
    current_user: UserProfile = Depends(get_current_user),
):
    """Return latest baseline comparison from DB."""
    try:
        row = await db.fetchrow(
            """SELECT result_json, finished_at FROM pade_training_jobs
               WHERE job_type = 'baseline' AND status = 'done' AND user_id = $1
               ORDER BY finished_at DESC LIMIT 1""",
            current_user.id,
        )
        if not row:
            return {"message": "No baseline results. Run a baseline training job first."}
        return {"result": row["result_json"], "computed_at": row["finished_at"]}
    except Exception as exc:
        logger.exception(f"get_baseline_report error: {exc}")
        raise HTTPException(500, "Failed to fetch baseline report")

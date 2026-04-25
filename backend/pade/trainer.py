"""
backend/pade/trainer.py
HF-13: The original stub tried to call a non-existent costguard_trainer.py via
subprocess, which always failed and logged errors.  Replaced with a thin wrapper
that delegates to pade_full.py directly (same engine used by the ML Training
Lab UI and the /api/pade/train/* endpoints).
"""
import logging
import os
import time
from pathlib import Path
from datetime import date, datetime

logger = logging.getLogger(__name__)

CHECKPOINT_DIR   = Path(__file__).parent / "checkpoints"
LAST_TRAINED_FILE = CHECKPOINT_DIR / "last_trained.txt"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp), str(path))
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def should_retrain() -> bool:
    """Return True if no checkpoint exists or last trained > 7 days ago."""
    if not CHECKPOINT_DIR.exists():
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        return True
    if not LAST_TRAINED_FILE.exists():
        return True
    try:
        last_date_str = LAST_TRAINED_FILE.read_text().strip()
        last_date = datetime.strptime(last_date_str, "%Y-%m-%d %H:%M:%S").date()
        return (date.today() - last_date).days >= 7
    except Exception:
        logger.exception("Error reading last_trained.txt, forcing retrain.")
        return True


def run_training(epochs: int = 30, seed: int = 42, n_rows: int = 10_000) -> bool:
    """
    HF-13: Invoke pade_full.py training pipeline directly instead of via a
    subprocess call to the non-existent costguard_trainer.py.
    Returns True on success, False on failure.
    """
    import sys, os
    pade_dir = str(Path(__file__).parent)
    if pade_dir not in sys.path:
        sys.path.insert(0, pade_dir)

    work_dir  = Path(__file__).parent / "training_workspace"
    data_dir  = str(work_dir / "data")
    ml_dir    = str(work_dir / "ml_ready")
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(ml_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Starting PADE training via pade_full.py…")
    try:
        from pade_full import (
            LSTMConfig, GATConfig, seed_everything,
            run_synthetic_data_generation, run_preprocessing,
            train_lstm, train_gat,
        )
        import pade_full as _pf
        _pf._SYNTH_MODE = "enhanced"

        seed_everything(seed)
        run_synthetic_data_generation(
            n_rows=n_rows, out_dir=data_dir, seed=seed, anomaly_rate=0.08
        )
        run_preprocessing(raw_dir=data_dir, out_dir=ml_dir, seed=seed)
        train_lstm(cfg=LSTMConfig(epochs=epochs, seed=seed), ckpt_dir=CHECKPOINT_DIR)
        train_gat(cfg=GATConfig(epochs=epochs, seed=seed),   ckpt_dir=CHECKPOINT_DIR)

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(LAST_TRAINED_FILE, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("PADE training completed successfully.")
        return True
    except Exception as exc:
        logger.exception(f"PADE training failed: {exc}")
        return False


def ensure_models():
    """Check if retraining is needed and run training if so."""
    if should_retrain():
        logger.info("Retraining condition met. Starting training…")
        run_training()
    else:
        logger.info("Models are up to date (last trained < 7 days ago).")

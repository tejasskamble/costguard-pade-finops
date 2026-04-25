"""
backend/pade/memory_guard.py - CostGuard v17.0 Production
=========================================================
VRAM / RAM monitoring utility.

Every component touching GPU memory must respect the RTX 3050-oriented VRAM
budget. This module provides logging, probing, and guardrails before any ML
operation runs.
"""
import gc
import logging
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

# Warn when a tracked block consumes more than this (MB)
_VRAM_DELTA_WARN_MB = 500.0


class MemoryGuard:
    """
    Context manager that logs VRAM/RAM before and after a block.

    Usage::

        mg = MemoryGuard()
        with mg.track("LSTM training epoch 1"):
            train_one_epoch(model, loader)

        MemoryGuard.log_vram("Before GAT DataLoader build")
        MemoryGuard.safe_cuda_clear()
        MemoryGuard.assert_vram_available(required_gb=1.5, label="GAT forward pass")
    """

    # ── Static utilities ──────────────────────────────────────────────────────

    @staticmethod
    def log_vram(label: str = "") -> None:
        """Log free and total VRAM (if CUDA available) plus system RAM."""
        try:
            import torch
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                free_gb = free_bytes / 1e9
                total_gb = total_bytes / 1e9
                used_gb = total_gb - free_gb
                logger.info(
                    "[MemoryGuard%s] VRAM: %.2f GB used / %.2f GB total (%.2f GB free)",
                    f" | {label}" if label else "",
                    used_gb,
                    total_gb,
                    free_gb,
                )
            else:
                logger.info(
                    "[MemoryGuard%s] CUDA unavailable — CPU-only mode",
                    f" | {label}" if label else "",
                )
        except Exception as exc:
            logger.warning("[MemoryGuard] log_vram failed: %s", exc)

        # Also log system RAM
        try:
            import psutil  # optional but common in ML environments
            vm = psutil.virtual_memory()
            logger.info(
                "[MemoryGuard%s] RAM: %.2f GB used / %.2f GB total (%.1f%% used)",
                f" | {label}" if label else "",
                vm.used / 1e9,
                vm.total / 1e9,
                vm.percent,
            )
        except ImportError:
            logger.debug("[MemoryGuard] psutil not installed; RAM metrics skipped")
        except Exception as exc:
            logger.debug("[MemoryGuard] RAM log failed (psutil): %s", exc)

    @staticmethod
    def safe_cuda_clear() -> None:
        """
        torch.cuda.empty_cache() + gc.collect() with full error handling.
        Safe to call even when CUDA is unavailable.
        """
        try:
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.debug("[MemoryGuard] CUDA cache cleared.")
        except Exception as exc:
            logger.warning("[MemoryGuard] safe_cuda_clear failed: %s", exc)

    @staticmethod
    def assert_vram_available(
        required_gb: float,
        label: str = "",
    ) -> None:
        """
        Raise RuntimeError with an actionable message if free VRAM is below
        required_gb. No-op when CUDA is unavailable.

        Args:
            required_gb: Minimum free VRAM in GB required to proceed.
            label: Human-readable description of the operation (for the error message).
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return
            free_bytes, _ = torch.cuda.mem_get_info()
            free_gb = free_bytes / 1e9
            if free_gb < required_gb:
                raise RuntimeError(
                    f"[MemoryGuard] Insufficient VRAM for '{label}': "
                    f"need {required_gb:.2f} GB, have {free_gb:.2f} GB free. "
                    f"Try: (1) reduce batch_size, (2) call safe_cuda_clear() first, "
                    f"(3) use float32 instead of float64."
                )
            logger.debug(
                "[MemoryGuard] VRAM check OK for '%s': %.2f GB free (need %.2f GB)",
                label, free_gb, required_gb,
            )
        except RuntimeError:
            raise  # re-raise our own RuntimeError
        except Exception as exc:
            logger.warning("[MemoryGuard] assert_vram_available failed: %s", exc)

    @staticmethod
    def get_free_vram_gb() -> Optional[float]:
        """Return free VRAM in GB, or None if CUDA unavailable."""
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            free_bytes, _ = torch.cuda.mem_get_info()
            return free_bytes / 1e9
        except Exception:
            return None

    @staticmethod
    def probe_safe_batch_size(
        default: int = 256,
        min_vram_gb: float = 0.5,
    ) -> int:
        """
        BUG-MEDIUM-8 / Patch 4-D: Probe free VRAM and return a safe batch size.
        Reduces batch size if VRAM is tight.

        Args:
            default: The batch size to use when VRAM is plentiful.
            min_vram_gb: Minimum free VRAM (GB) headroom required for safe operation.

        Returns:
            A safe batch size: default, default//2, or default//4.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return default
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            if free_gb < min_vram_gb + 0.5:
                safe = max(32, default // 4)
                logger.warning(
                    "[MemoryGuard] Low VRAM (%.2f GB free) — reducing batch_size "
                    "%d → %d",
                    free_gb, default, safe,
                )
                return safe
            if free_gb < min_vram_gb + 1.0:
                safe = max(64, default // 2)
                logger.warning(
                    "[MemoryGuard] Moderate VRAM (%.2f GB free) — reducing batch_size "
                    "%d → %d",
                    free_gb, default, safe,
                )
                return safe
            return default
        except Exception as exc:
            logger.debug("[MemoryGuard] Falling back to default batch size: %s", exc)
            return default

    # ── Context manager ───────────────────────────────────────────────────────

    @contextmanager
    def track(self, label: str):
        """
        Context manager. Logs VRAM before/after the block and warns if the
        delta exceeds _VRAM_DELTA_WARN_MB (500 MB by default).

        Usage::
            with MemoryGuard().track("GAT epoch 5"):
                train_one_epoch(gat_model, train_loader)
        """
        before_free: Optional[float] = None

        try:
            import torch
            if torch.cuda.is_available():
                before_free = torch.cuda.mem_get_info()[0] / 1e6  # MB
                logger.info(
                    "[MemoryGuard | %s] START — %.0f MB VRAM free", label, before_free
                )
        except Exception as exc:
            logger.debug("[MemoryGuard] START probe skipped for %s: %s", label, exc)

        try:
            yield
        finally:
            try:
                import torch
                if torch.cuda.is_available():
                    after_free = torch.cuda.mem_get_info()[0] / 1e6  # MB
                    delta_mb = (before_free or 0.0) - after_free
                    level = logging.WARNING if delta_mb > _VRAM_DELTA_WARN_MB else logging.INFO
                    logger.log(
                        level,
                        "[MemoryGuard | %s] END — %.0f MB VRAM free (Δ=%.0f MB consumed%s)",
                        label,
                        after_free,
                        delta_mb,
                        " ⚠ HIGH" if delta_mb > _VRAM_DELTA_WARN_MB else "",
                    )
            except Exception as exc:
                logger.debug("[MemoryGuard] END probe skipped for %s: %s", label, exc)

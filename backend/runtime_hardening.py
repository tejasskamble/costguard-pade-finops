"""Shared backend hardening helpers for retries, warnings, and task safety."""
from __future__ import annotations

import asyncio
import gc
import logging
import random
import sys
import threading
import warnings
from typing import Any, Awaitable, Callable, Optional, Sequence, TypeVar

T = TypeVar("T")


def configure_backend_runtime() -> None:
    """Enable warning capture and silence narrowly scoped third-party noise."""
    warnings.filterwarnings(
        "ignore",
        message=r"The usage of `scatter\(reduce='max'\)` can be accelerated via the 'torch-scatter' package",
        category=UserWarning,
        module=r"torch_geometric\.utils\._scatter",
    )
    logging.captureWarnings(True)


def install_backend_exception_hooks(logger: logging.Logger) -> None:
    """Log uncaught main-thread and worker-thread exceptions."""

    def _log_exception(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logger.critical("Unhandled backend exception", exc_info=(exc_type, exc, tb))

    def _thread_hook(args: threading.ExceptHookArgs) -> None:
        logger.critical(
            "Unhandled backend thread exception in %s",
            getattr(args.thread, "name", "unknown"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = _log_exception
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_hook


def install_asyncio_exception_handler(loop: asyncio.AbstractEventLoop, logger: logging.Logger) -> None:
    """Log background task failures consistently."""

    def _handler(_loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        message = context.get("message", "Background asyncio task failed")
        if exc is not None:
            logger.error("%s", message, exc_info=(type(exc), exc, exc.__traceback__))
        else:
            logger.error("%s", message)

    loop.set_exception_handler(_handler)


def safe_create_task(
    coro: Awaitable[Any],
    *,
    logger: logging.Logger,
    label: str,
) -> asyncio.Task[Any]:
    """Create a task and guarantee late exceptions are logged."""
    task = asyncio.create_task(coro)

    def _done_callback(done_task: asyncio.Task[Any]) -> None:
        try:
            done_task.result()
        except asyncio.CancelledError:
            logger.debug("%s task cancelled", label)
        except Exception as exc:
            logger.error("%s task failed: %s", label, exc, exc_info=exc)

    task.add_done_callback(_done_callback)
    return task


async def retry_async(
    func: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    delay: float = 0.5,
    backoff: float = 2.0,
    max_delay: float = 4.0,
    jitter: float = 0.05,
    exceptions: Sequence[type[BaseException]] = (Exception,),
    logger: Optional[logging.Logger] = None,
    label: str = "operation",
) -> T:
    """Retry an async operation with bounded exponential backoff."""
    current_delay = max(0.0, delay)
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return await func()
        except tuple(exceptions) as exc:  # type: ignore[arg-type]
            last_exc = exc
            if attempt >= attempts:
                break
            if logger is not None:
                logger.warning(
                    "Retrying %s after %s (%d/%d)",
                    label,
                    type(exc).__name__,
                    attempt,
                    attempts,
                )
            await asyncio.sleep(min(max_delay, current_delay) + random.uniform(0.0, jitter))
            current_delay = min(max_delay, max(current_delay * backoff, current_delay + 0.1))
    assert last_exc is not None
    raise last_exc


def is_torch_oom(exc: BaseException) -> bool:
    """Detect common torch OOM failure modes from exception text."""
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def clear_torch_memory(logger: Optional[logging.Logger] = None, *, label: str = "") -> None:
    """Release Python and CUDA caches where available."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as exc:  # pragma: no cover - optional dependency guard
        if logger is not None:
            logger.debug("Torch memory clear skipped for %s: %s", label or "backend", exc)

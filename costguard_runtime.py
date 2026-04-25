"""Shared runtime hardening helpers for CostGuard scripts and analytics."""
from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import random
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")

KNOWN_WARNING_FILTERS: Tuple[Tuple[str, str, type[Warning], str], ...] = (
    (
        "ignore",
        r"The usage of `scatter\(reduce='max'\)` can be accelerated via the 'torch-scatter' package",
        UserWarning,
        r"torch_geometric\.utils\._scatter",
    ),
    (
        "ignore",
        r".*The PostScript backend does not support transparency.*",
        UserWarning,
        r"matplotlib\..*",
    ),
)


def configure_warning_filters() -> None:
    """Apply narrowly scoped warning filters for known third-party noise."""
    for action, message, category, module in KNOWN_WARNING_FILTERS:
        warnings.filterwarnings(action, message=message, category=category, module=module)
    logging.captureWarnings(True)


def _utf8_console_stream(stream: Any) -> Any:
    try:
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    return stream


def configure_console_logger(name: str, level: int = logging.INFO, stream: Any = None) -> logging.Logger:
    """Configure a UTF-8-safe console logger without duplicate handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not getattr(logger, "_costguard_console_configured", False):
        handler = logging.StreamHandler(_utf8_console_stream(stream or sys.stdout))
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.propagate = False
        setattr(logger, "_costguard_console_configured", True)
    return logger


def _render_log_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (int, bool)):
        return str(value)
    if value is None:
        return "null"
    text = str(value)
    if not text:
        return '""'
    if any(ch.isspace() for ch in text) or any(ch in text for ch in '"='):
        return json.dumps(text, ensure_ascii=True)
    return text


def format_log_event(*tags: Any, message: str | None = None, **fields: Any) -> str:
    tag_text = "".join(f"[{tag}]" for tag in tags if tag not in (None, ""))
    parts = [tag_text] if tag_text else []
    if message:
        parts.append(str(message))
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={_render_log_value(value)}")
    return " ".join(part for part in parts if part)


def log_event(
    logger: logging.Logger,
    *tags: Any,
    level: int = logging.INFO,
    message: str | None = None,
    **fields: Any,
) -> None:
    logger.log(level, format_log_event(*tags, message=message, **fields))


class StageTimer:
    def __init__(self) -> None:
        self.started_at = time.perf_counter()

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self.started_at


def format_duration_s(duration_s: float) -> str:
    return f"{float(duration_s):.2f}s"


def _atomic_temp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")


def _fsync_parent_dir(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
    except (AttributeError, OSError):
        return
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def atomic_write_file(path: str | Path, writer: Callable[[Path], None]) -> None:
    """Write through a sibling temp file, fsync it, then atomically replace path."""
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _atomic_temp_path(final_path)
    try:
        tmp_path.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        writer(tmp_path)
        try:
            with tmp_path.open("rb") as handle:
                os.fsync(handle.fileno())
        except OSError:
            pass
        os.replace(str(tmp_path), str(final_path))
        _fsync_parent_dir(final_path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    def _writer(tmp_path: Path) -> None:
        with tmp_path.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

    atomic_write_file(path, _writer)


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    def _writer(tmp_path: Path) -> None:
        with tmp_path.open("w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

    atomic_write_file(path, _writer)


def atomic_write_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(payload, indent=indent, default=str), encoding="utf-8")


def install_global_exception_hooks(logger: logging.Logger) -> None:
    """Log uncaught process-wide exceptions instead of losing them to stderr."""

    def _log_exception(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logger.critical("Unhandled exception reached the process boundary", exc_info=(exc_type, exc, tb))

    def _thread_hook(args: threading.ExceptHookArgs) -> None:
        logger.critical(
            "Unhandled exception in thread '%s'",
            getattr(args.thread, "name", "unknown"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = _log_exception
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_hook


def install_asyncio_exception_handler(loop: asyncio.AbstractEventLoop, logger: logging.Logger) -> None:
    """Ensure detached task failures are logged with context."""

    def _handler(_loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        message = context.get("message", "Asyncio background task failed")
        if exc is not None:
            logger.error("%s", message, exc_info=(type(exc), exc, exc.__traceback__))
        else:
            logger.error("%s", message)

    loop.set_exception_handler(_handler)


def is_torch_oom(exc: BaseException) -> bool:
    """Return True when an exception represents a CUDA or torch OOM condition."""
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def clear_torch_memory(logger: Optional[logging.Logger] = None, *, label: str = "") -> None:
    """Clear Python and CUDA caches when torch is available."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as exc:  # pragma: no cover - defensive optional-dep guard
        if logger is not None:
            logger.debug("Torch memory clear skipped for %s: %s", label or "runtime", exc)


def retry_sync(
    func: Callable[[], T],
    *,
    attempts: int = 3,
    delay: float = 0.5,
    backoff: float = 2.0,
    max_delay: float = 4.0,
    exceptions: Sequence[type[BaseException]] = (Exception,),
    logger: Optional[logging.Logger] = None,
    label: str = "operation",
) -> T:
    """Retry a synchronous operation with bounded exponential backoff."""
    current_delay = max(0.0, delay)
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return func()
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
            time.sleep(current_delay)
            current_delay = min(max_delay, max(current_delay * backoff, current_delay + 0.1))
    assert last_exc is not None
    raise last_exc


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
            sleep_for = min(max_delay, current_delay) + random.uniform(0.0, jitter)
            await asyncio.sleep(sleep_for)
            current_delay = min(max_delay, max(current_delay * backoff, current_delay + 0.1))
    assert last_exc is not None
    raise last_exc

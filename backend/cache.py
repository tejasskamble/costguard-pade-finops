"""
backend/cache.py — FEATURE-4: In-Memory TTL Cache Layer
CONSTRAINT-A: Uses cachetools.TTLCache — no Redis, no Docker service needed.
Thread-safe for asyncio (single event loop — no mutex required).
"""
import logging
from functools import wraps
from typing import Any, Callable, Optional

from cachetools import TTLCache

logger = logging.getLogger(__name__)

# ─── Named cache pools with tuned TTLs ───────────────────────────────────────
_caches: dict[str, TTLCache] = {
    "alerts":        TTLCache(maxsize=100, ttl=30),    # 30s  — fast-changing live feed
    "daily_summary": TTLCache(maxsize=50,  ttl=300),   # 5min — moderate churn
    "policy":        TTLCache(maxsize=10,  ttl=600),   # 10min — slow-changing thresholds
    "forecast":      TTLCache(maxsize=20,  ttl=900),   # 15min — expensive ETS computation
    "dag":           TTLCache(maxsize=200, ttl=60),    # 1min  — per-run_id graph
    "pade_status":   TTLCache(maxsize=5,   ttl=15),    # 15s  — model health probe
}


def cached(cache_name: str, key_fn: Optional[Callable] = None):
    """
    Decorator for async FastAPI route handlers.
    Caches the return value in the named TTLCache.

    key_fn: optional callable(request, *args, **kwargs) → str
            Falls back to str(args)+str(kwargs) if None.

    CONSTRAINT-A: fully in-process — no network, no serialisation cost.
    VG-8: First call hits DB; subsequent calls within TTL return from cache.
    """
    def decorator(fn: Callable):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            cache = _caches.get(cache_name)
            if cache is None:
                logger.warning(f"Unknown cache name '{cache_name}' — bypassing cache")
                return await fn(*args, **kwargs)

            # Build cache key
            try:
                key = key_fn(*args, **kwargs) if key_fn else (str(args) + str(sorted(kwargs.items())))
            except Exception:
                key = str(args) + str(kwargs)

            if key in cache:
                logger.debug(f"Cache HIT  [{cache_name}] key={key[:60]}")
                return cache[key]

            logger.debug(f"Cache MISS [{cache_name}] key={key[:60]} — querying DB")
            result = await fn(*args, **kwargs)
            cache[key] = result
            return result

        return wrapper
    return decorator


def invalidate(cache_name: str, key: Optional[str] = None) -> None:
    """
    Invalidate a specific key or the entire named cache.
    Called after writes (policy update, new alert insert) to keep data fresh.
    """
    cache = _caches.get(cache_name)
    if cache is None:
        return
    if key is not None:
        cache.pop(key, None)
        logger.debug(f"Cache INVALIDATED [{cache_name}] key={key[:60]}")
    else:
        cache.clear()
        logger.debug(f"Cache CLEARED [{cache_name}]")


def invalidate_all() -> None:
    """Clear every cache pool — called on server startup to prevent stale data."""
    for name, cache in _caches.items():
        cache.clear()
    logger.info("All caches cleared on startup")

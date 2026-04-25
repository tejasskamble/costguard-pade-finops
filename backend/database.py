"""
backend/database.py
Unchanged from original except path resolution for migration file.
Works with the new idempotent 001_init_focus_schema.sql (no DROP TABLE).
"""
import asyncpg
import logging
from pathlib import Path

from fastapi import HTTPException, Request

from config import settings
from runtime_hardening import retry_async

logger = logging.getLogger(__name__)


async def create_pool() -> asyncpg.Pool:
    """Create and return a connection pool using settings from config.py."""

    async def _open_pool() -> asyncpg.Pool:
        return await asyncpg.create_pool(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            database=settings.DB_NAME,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            min_size=settings.DB_MIN_CONN,
            max_size=settings.DB_MAX_CONN,
            command_timeout=60,
        )

    return await retry_async(
        _open_pool,
        attempts=3,
        delay=1.0,
        logger=logger,
        label="asyncpg pool creation",
    )


async def run_migration(pool: asyncpg.Pool) -> None:
    """
    Execute the initial SQL migration to create/update tables.
    GAP-8 fix: migration is now idempotent (IF NOT EXISTS, no DROP TABLE).
    Searches two locations: project root/migrations and parent dir.
    """
    # Try relative to backend dir first, then parent
    candidates = [
        Path(__file__).parent.parent / "migrations" / "001_init_focus_schema.sql",
        Path(__file__).parent / ".." / ".." / "migrations" / "001_init_focus_schema.sql",
    ]
    migration_file = None
    for c in candidates:
        if c.exists():
            migration_file = c.resolve()
            break

    if migration_file is None:
        logger.warning("Migration file not found — database schema may be incomplete.")
        return

    sql = migration_file.read_text(encoding="utf-8")
    async with pool.acquire() as conn:
        try:
            await conn.execute(sql)
            logger.info("Database migration applied successfully: %s", migration_file.name)
        except Exception as exc:
            logger.exception("Migration failed: %s", exc)
            raise


async def close_pool(pool: asyncpg.Pool | None) -> None:
    if pool is None:
        logger.info("Database connection pool was never created; shutdown is a no-op.")
        return
    await pool.close()
    logger.info("Database connection pool closed.")


async def get_db_conn(request: Request):
    """FastAPI dependency: yield a single connection from the pool."""
    pool = request.app.state.db
    if pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable; CostGuard is running in degraded mode.")
    async with pool.acquire() as conn:
        yield conn

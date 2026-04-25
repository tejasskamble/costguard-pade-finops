"""
backend/lcqi/query_executor.py
CONSTRAINT-4: generate_nl_summary() has static fallback when OPENAI_API_KEY absent.
"""
import time
import logging
from dataclasses import dataclass
from typing import List, Any

import asyncpg

from config import settings
from .text_to_sql import sanitise_sql

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    columns: List[str]
    rows: List[List[Any]]
    row_count: int
    execution_time_ms: float


async def execute_cost_query(sql: str, pool: asyncpg.Pool) -> QueryResult:
    """Execute a read-only SQL query and return results."""
    start = time.perf_counter()
    sql = sanitise_sql(sql)
    if pool is None:
        raise ValueError("Cost query service is unavailable while the database is degraded.")
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SET LOCAL statement_timeout = '30s'")
                await conn.execute("SET TRANSACTION READ ONLY")
                records = await conn.fetch(sql)
    except Exception as exc:
        logger.exception("Query execution failed")
        raise ValueError("Query execution failed against the read-only cost dataset.")

    elapsed_ms = (time.perf_counter() - start) * 1000

    if not records:
        return QueryResult(columns=[], rows=[], row_count=0, execution_time_ms=elapsed_ms)

    columns = list(records[0].keys())
    rows    = [list(r.values()) for r in records]
    return QueryResult(
        columns=columns,
        rows=rows,
        row_count=len(rows),
        execution_time_ms=elapsed_ms,
    )


async def generate_nl_summary(question: str, result: QueryResult) -> str:
    """
    Produce a natural language summary of the query result.
    CONSTRAINT-4: Falls back to a static summary if OPENAI_API_KEY is absent.
    Never raises — always returns a non-empty string.
    """
    if result.row_count == 0:
        return "No data found for your question. Try simulating a pipeline run first."

    # Static fallback (no OpenAI needed)
    if not settings.OPENAI_API_KEY:
        preview = result.rows[:3]
        row_text = "; ".join(
            ", ".join(f"{col}={val}" for col, val in zip(result.columns, row))
            for row in preview
        )
        return (
            f"Query returned {result.row_count} row(s). "
            f"Sample: {row_text[:300]}."
        )

    preview = result.rows[:5]
    context = (
        f"Question: {question}\n"
        f"Rows returned: {result.row_count}\n"
        f"Columns: {result.columns}\n"
        f"First few rows: {preview}"
    )

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            max_tokens=settings.OPENAI_MAX_TOKENS,
            messages=[
                {
                    "role":    "system",
                    "content": "Summarise the SQL query result in 2–3 clear sentences for a DevOps engineer. Be specific about numbers and stage names.",
                },
                {"role": "user", "content": context},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning(f"OpenAI summary failed: {exc}")
        return f"Query returned {result.row_count} row(s) in {result.execution_time_ms:.1f} ms."

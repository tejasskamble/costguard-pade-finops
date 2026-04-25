"""
backend/lcqi/router.py - CostGuard v17.0

Natural-language cost query routing for the enterprise dashboard.
"""
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.auth import get_current_user
from database import get_db_conn
from .query_executor import QueryResult, execute_cost_query, generate_nl_summary
from .text_to_sql import translate_to_sql
from config import settings

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
router  = APIRouter(prefix="/api", tags=["lcqi"])


# ── Pydantic models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    sql_generated: str
    columns: list[str]
    rows: list[list]
    natural_language_answer: str
    execution_time_ms: float
    row_count: int


# ── Standard (non-streaming) query ───────────────────────────────────────────

@router.post("/query")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def cost_query(
    request: Request,
    query_req: QueryRequest,
    conn=Depends(get_db_conn),
) -> QueryResponse:
    """Translate NL → SQL, execute, return full response (non-streaming)."""
    try:
        sql    = await translate_to_sql(query_req.question)
        pool   = request.app.state.db
        result = await execute_cost_query(sql, pool)
        answer = await generate_nl_summary(query_req.question, result)
        return QueryResponse(
            sql_generated=sql,
            columns=result.columns,
            rows=result.rows,
            natural_language_answer=answer,
            execution_time_ms=result.execution_time_ms,
            row_count=result.row_count,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        logger.exception("LCQI query failed")
        raise HTTPException(500, "Internal error processing query.")


# ── FEATURE-6: Streaming query endpoint ───────────────────────────────────────

def _static_summary_tokens(question: str, result: QueryResult) -> list[str]:
    """Tokenize a static fallback summary into word chunks for streaming."""
    if result.row_count == 0:
        text = "No data found for your question. Try simulating a pipeline run first."
    else:
        preview = result.rows[:3]
        text = (
            f"Your query returned {result.row_count} row(s) "
            f"in {result.execution_time_ms:.1f} ms. "
            f"Sample data: {preview[:200]}."
        )
    return text.split(" ")


@router.get("/query/stream")
async def stream_query(
    question: str = Query(..., description="Natural language question about pipeline costs"),
    request: Request = None,
) -> StreamingResponse:
    """
    FEATURE-6: SSE streaming LCQI endpoint — answers appear word-by-word.
    CONSTRAINT-C: Falls back to static tokenised summary without OpenAI key.
    CONSTRAINT-E: Sends 'data: [DONE]\\n\\n' to terminate the stream.
    """
    # Execute SQL first (synchronous side — always fast)
    try:
        sql    = await translate_to_sql(question)
        pool   = request.app.state.db
        result = await execute_cost_query(sql, pool)
    except Exception as exc:
        logger.exception("LCQI stream pre-processing failed")
        sql    = "-- query error"
        result = QueryResult(columns=[], rows=[], row_count=0, execution_time_ms=0)

    async def token_generator():
        # First event: send the SQL so the dashboard can display it immediately
        yield f"data: {json.dumps({'sql': sql, 'row_count': result.row_count, 'columns': result.columns, 'rows': result.rows[:100]})}\n\n"

        if not settings.OPENAI_API_KEY:
            # CONSTRAINT-C: static fallback streamed word-by-word
            for word in _static_summary_tokens(question, result):
                if await request.is_disconnected():
                    return
                yield f"data: {json.dumps({'token': word + ' '})}\n\n"
            yield "data: [DONE]\n\n"
            return

        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

            # Build summary prompt
            preview = result.rows[:5]
            context = (
                f"Question: {question}\n"
                f"SQL returned {result.row_count} rows.\n"
                f"Columns: {result.columns}\n"
                f"Sample: {preview}"
            )
            stream = await client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {
                        "role":    "system",
                        "content": (
                            "You are a FinOps analyst. Summarise the SQL result in "
                            "2-3 sentences for a DevOps engineer. Be specific about numbers."
                        ),
                    },
                    {"role": "user", "content": context},
                ],
                max_tokens=300,
                stream=True,
            )

            async for chunk in stream:
                if await request.is_disconnected():
                    break
                token = chunk.choices[0].delta.content or ""
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"

            yield "data: [DONE]\n\n"   # CONSTRAINT-E: required terminator

        except Exception as exc:
            logger.warning(f"OpenAI stream failed: {exc} — using static fallback")
            for word in _static_summary_tokens(question, result):
                if await request.is_disconnected():
                    return
                yield f"data: {json.dumps({'token': word + ' '})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── NEW-BUG-4: Query history endpoints ───────────────────────────────────────

@router.get("/query/history")
async def get_query_history(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    current_user=Depends(get_current_user),
):
    """
    NEW-BUG-4: Return the last N queries for the authenticated user.
    Allows Page 03 to restore chat history across page navigations.
    """
    pool = request.app.state.db
    try:
        async with pool.acquire() as conn:
            row_u = await conn.fetchrow(
                "SELECT id FROM users WHERE email = $1", current_user.email
            )
            if not row_u:
                return []
            rows = await conn.fetch(
                """
                SELECT question, sql_generated, nl_answer, row_count, created_at
                FROM query_history
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                row_u["id"], limit,
            )
            return [dict(r) for r in rows]
    except Exception as exc:
        logger.exception("Error fetching query history")
        raise HTTPException(500, "Failed to fetch query history.")


class SaveQueryRequest(BaseModel):
    question:      str
    sql_generated: Optional[str] = None
    nl_answer:     Optional[str] = None
    row_count:     int = 0


@router.post("/query/save")
async def save_query(
    request: Request,
    body: SaveQueryRequest,
    current_user=Depends(get_current_user),
):
    """NEW-BUG-4: Persist a query result to the history table."""
    pool = request.app.state.db
    try:
        async with pool.acquire() as conn:
            row_u = await conn.fetchrow(
                "SELECT id FROM users WHERE email = $1", current_user.email
            )
            if not row_u:
                raise HTTPException(404, "User not found")
            await conn.execute(
                """
                INSERT INTO query_history
                    (user_id, question, sql_generated, nl_answer, row_count)
                VALUES ($1, $2, $3, $4, $5)
                """,
                row_u["id"], body.question,
                body.sql_generated, body.nl_answer, body.row_count,
            )
        return {"status": "saved"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error saving query")
        raise HTTPException(500, "Failed to save query history.")

"""
backend/lcqi/text_to_sql.py
CONSTRAINT-4: Added try/except fallback when OPENAI_API_KEY is not set.
GAP-8 fix: includes ai_recommendation column in schema hint.
"""
import logging
import re
from config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a read-only PostgreSQL SQL assistant for CostGuard.
You ONLY write SELECT queries against the table: cost_attribution
Columns available: id, run_id, stage_name, resource_type, billed_cost,
  effective_cost, billing_currency, usage_quantity, usage_unit, provider,
  region, cost_deviation_pct, historical_avg_cost, crs_score, pade_decision,
  ai_recommendation, window_start, window_end, timestamp_start, timestamp_end,
  created_at
Return ONLY the SQL query. No explanation. No markdown. No code blocks.
Always include LIMIT 100 unless the user specifies otherwise."""

BANNED_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
    "TRUNCATE", "EXEC", "CREATE", "GRANT", "REVOKE",
    "COPY", "CALL", "VACUUM", "ANALYZE", "REFRESH",
    "SET ", "SHOW ", "LISTEN", "NOTIFY", "PG_SLEEP",
    "DBLINK", "COPY_PROGRAM",
]
ALLOWED_TABLES = {"cost_attribution"}
_TABLE_REF_RE = re.compile(r"\b(?:FROM|JOIN)\s+([a-z_][a-z0-9_\.]*)", re.IGNORECASE)
_CTE_RE = re.compile(r"\bWITH\s+([a-z_][a-z0-9_]*)\s+AS\b", re.IGNORECASE)
_CTE_CONTINUATION_RE = re.compile(r",\s*([a-z_][a-z0-9_]*)\s+AS\b", re.IGNORECASE)

# Simple keyword-to-SQL fallbacks for common questions (no OpenAI needed)
FALLBACK_SQL_MAP = [
    (["most expensive", "highest cost", "cost the most"],
     "SELECT stage_name, SUM(billed_cost) AS total_cost FROM cost_attribution "
     "GROUP BY stage_name ORDER BY total_cost DESC LIMIT 10"),
    (["block", "blocked"],
     "SELECT * FROM cost_attribution WHERE pade_decision = 'BLOCK' "
     "ORDER BY created_at DESC LIMIT 20"),
    (["warn"],
     "SELECT * FROM cost_attribution WHERE pade_decision = 'WARN' "
     "ORDER BY created_at DESC LIMIT 20"),
    (["average cost", "avg cost"],
     "SELECT stage_name, AVG(billed_cost) AS avg_cost FROM cost_attribution "
     "GROUP BY stage_name ORDER BY avg_cost DESC LIMIT 10"),
    (["anomal"],
     "SELECT * FROM cost_attribution WHERE pade_decision IN ('WARN','AUTO_OPTIMISE','BLOCK') "
     "ORDER BY created_at DESC LIMIT 20"),
    (["provider"],
     "SELECT provider, SUM(billed_cost) AS total, COUNT(*) AS runs "
     "FROM cost_attribution GROUP BY provider ORDER BY total DESC LIMIT 10"),
    (["recent", "latest", "last"],
     "SELECT * FROM cost_attribution ORDER BY created_at DESC LIMIT 20"),
]


def _fallback_sql(question: str) -> str:
    """Return a best-guess SQL for common questions when OpenAI is unavailable."""
    q_lower = question.lower()
    for keywords, sql in FALLBACK_SQL_MAP:
        if any(k in q_lower for k in keywords):
            return sql
    return (
        "SELECT stage_name, SUM(billed_cost) AS total_cost, "
        "AVG(crs_score) AS avg_crs, COUNT(*) AS records "
        "FROM cost_attribution "
        "GROUP BY stage_name ORDER BY total_cost DESC LIMIT 20"
    )


def sanitise_sql(sql: str) -> str:
    """Raise ValueError when SQL is not a single read-only query over approved tables."""
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise ValueError("SQL query is empty.")
    if ";" in cleaned:
        raise ValueError("Only single-statement queries are allowed.")

    sql_upper = cleaned.upper()
    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
        raise ValueError("Only read-only SELECT queries are allowed.")
    for keyword in BANNED_KEYWORDS:
        if keyword in sql_upper:
            raise ValueError(f"SQL contains banned keyword: {keyword}")

    cte_names = {name.lower() for name in _CTE_RE.findall(cleaned)}
    cte_names.update(name.lower() for name in _CTE_CONTINUATION_RE.findall(cleaned))

    invalid_refs = []
    for raw_ref in _TABLE_REF_RE.findall(cleaned):
        ref = raw_ref.split(".")[-1].lower()
        if ref in cte_names or ref in ALLOWED_TABLES:
            continue
        invalid_refs.append(ref)

    if invalid_refs:
        raise ValueError(
            f"SQL references disallowed relation(s): {', '.join(sorted(set(invalid_refs)))}"
        )

    return cleaned


async def translate_to_sql(question: str) -> str:
    """
    Convert a natural language question to SQL.
    CONSTRAINT-4: Falls back to keyword-based SQL if OPENAI_API_KEY is absent.
    """
    if not settings.OPENAI_API_KEY:
        logger.info("OPENAI_API_KEY not set — using keyword-based SQL fallback.")
        return sanitise_sql(_fallback_sql(question))

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            max_tokens=500,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ],
        )
        raw_sql = response.choices[0].message.content.strip()
        # Strip possible markdown fences
        if raw_sql.startswith("```sql"):
            raw_sql = raw_sql[6:]
        if raw_sql.startswith("```"):
            raw_sql = raw_sql[3:]
        if raw_sql.endswith("```"):
            raw_sql = raw_sql[:-3]
        return sanitise_sql(raw_sql.strip())
    except Exception as exc:
        logger.warning(f"OpenAI SQL generation failed: {exc} — using fallback SQL")
        return sanitise_sql(_fallback_sql(question))

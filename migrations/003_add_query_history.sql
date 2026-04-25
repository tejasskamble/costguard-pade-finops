-- ============================================================
-- CostGuard Migration 003 — LCQI Query History
-- NEW-BUG-4: Persists chat queries across page navigations
-- Idempotent — safe to run multiple times
-- ============================================================

CREATE TABLE IF NOT EXISTS query_history (
    id            BIGSERIAL PRIMARY KEY,
    user_id       INTEGER REFERENCES users(id) ON DELETE CASCADE,
    question      TEXT NOT NULL,
    sql_generated TEXT,
    nl_answer     TEXT,
    row_count     INTEGER DEFAULT 0,
    created_at    TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_query_history_user   ON query_history(user_id);
CREATE INDEX IF NOT EXISTS idx_query_history_created ON query_history(created_at DESC);

-- Record migration
INSERT INTO schema_migrations (version) VALUES ('003_add_query_history')
ON CONFLICT (version) DO NOTHING;

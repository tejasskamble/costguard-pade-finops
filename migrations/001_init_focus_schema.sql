-- ============================================================
-- CostGuard — Initial Schema Migration (Idempotent)
-- Fixes: GAP-7 (no DROP), GAP-8 (IF NOT EXISTS throughout)
--        GAP-1 (users table), GAP-6 (ai_recommendation column)
-- ============================================================

-- Migration tracking table
CREATE TABLE IF NOT EXISTS schema_migrations (
    version      VARCHAR(50) PRIMARY KEY,
    applied_at   TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Users table (GAP-1 fix: real auth, email-based)
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    email         VARCHAR(255) UNIQUE NOT NULL,
    full_name     VARCHAR(255) NOT NULL,
    password_hash TEXT     NOT NULL,
    role          VARCHAR(20) DEFAULT 'viewer'
                  CHECK (role IN ('admin', 'analyst', 'viewer')),
    is_active     BOOLEAN  DEFAULT TRUE,
    phone         VARCHAR(20),
    avatar_url    TEXT,
    last_login_at TIMESTAMP WITH TIME ZONE,
    created_at    TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- Pipeline runs (with user_id FK for per-user email routing — CONSTRAINT-3)
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id        VARCHAR(36) PRIMARY KEY,
    user_id       INTEGER REFERENCES users(id) ON DELETE SET NULL,
    branch_type   VARCHAR(50),
    executor_type VARCHAR(50),
    provider      VARCHAR(20),
    region        VARCHAR(50),
    total_cost_usd DECIMAL(10,6),
    stage_count   INTEGER,
    is_anomalous  BOOLEAN DEFAULT FALSE,
    created_at    TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_user ON pipeline_runs(user_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_created ON pipeline_runs(created_at DESC);

-- Detailed cost attribution per stage per window
-- GAP-6 fix: added ai_recommendation TEXT column
CREATE TABLE IF NOT EXISTS cost_attribution (
    id                  BIGSERIAL PRIMARY KEY,
    run_id              VARCHAR(36) NOT NULL,
    stage_name          VARCHAR(100) NOT NULL,
    resource_type       VARCHAR(50),
    billed_cost         DECIMAL(10,6),
    effective_cost      DECIMAL(10,6),
    billing_currency    VARCHAR(10) DEFAULT 'USD',
    usage_quantity      DECIMAL(10,4),
    usage_unit          VARCHAR(50),
    provider            VARCHAR(20),
    region              VARCHAR(50),
    cost_deviation_pct  DECIMAL(8,4),
    historical_avg_cost DECIMAL(10,6),
    crs_score           DECIMAL(5,4),
    pade_decision       VARCHAR(20),
    ai_recommendation   TEXT,
    window_start        TIMESTAMP WITH TIME ZONE,
    window_end          TIMESTAMP WITH TIME ZONE,
    timestamp_start     TIMESTAMP WITH TIME ZONE,
    timestamp_end       TIMESTAMP WITH TIME ZONE,
    created_at          TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_cost_attribution_run_id ON cost_attribution(run_id);
CREATE INDEX IF NOT EXISTS idx_cost_attribution_stage  ON cost_attribution(run_id, stage_name);
CREATE INDEX IF NOT EXISTS idx_cost_attribution_created ON cost_attribution(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cost_attribution_decision ON cost_attribution(pade_decision);

-- Policy configuration (singleton row)
CREATE TABLE IF NOT EXISTS policy_config (
    id                       SERIAL PRIMARY KEY,
    warn_threshold           DECIMAL(4,3) DEFAULT 0.50,
    auto_optimise_threshold  DECIMAL(4,3) DEFAULT 0.75,
    block_threshold          DECIMAL(4,3) DEFAULT 0.90,
    updated_at               TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert default thresholds only if table is empty (idempotent)
INSERT INTO policy_config (warn_threshold, auto_optimise_threshold, block_threshold)
SELECT 0.50, 0.75, 0.90
WHERE NOT EXISTS (SELECT 1 FROM policy_config LIMIT 1);

-- Add ai_recommendation column to existing deployments (idempotent ALTER)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'cost_attribution'
          AND column_name = 'ai_recommendation'
    ) THEN
        ALTER TABLE cost_attribution ADD COLUMN ai_recommendation TEXT;
    END IF;
END $$;

-- Add user_id column to existing pipeline_runs deployments (idempotent)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'pipeline_runs'
          AND column_name = 'user_id'
    ) THEN
        ALTER TABLE pipeline_runs ADD COLUMN user_id INTEGER REFERENCES users(id) ON DELETE SET NULL;
    END IF;
END $$;

-- Record migration version
INSERT INTO schema_migrations (version) VALUES ('001_init_focus_schema')
ON CONFLICT (version) DO NOTHING;

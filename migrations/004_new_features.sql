-- ═══════════════════════════════════════════════════════════════
-- CostGuard v9.0  —  Migration 004: New Features
-- Idempotent: safe to run on fresh DB and on existing populated DB
-- ═══════════════════════════════════════════════════════════════

-- ── OTP password reset table (Objective B) ──────────────────────────────────
CREATE TABLE IF NOT EXISTS password_reset_otps (
    id          SERIAL PRIMARY KEY,
    email       VARCHAR(255) NOT NULL,
    otp_hash    TEXT NOT NULL,
    expires_at  TIMESTAMPTZ NOT NULL,
    used        BOOLEAN DEFAULT FALSE,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_otp_email_active
    ON password_reset_otps(email) WHERE used = FALSE;

-- Fix 3A: fail_count for brute-force OTP lockout
ALTER TABLE password_reset_otps
    ADD COLUMN IF NOT EXISTS fail_count INTEGER DEFAULT 0;

-- ── PADE training jobs (Objective A) ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pade_training_jobs (
    id           SERIAL PRIMARY KEY,
    user_id      INTEGER REFERENCES users(id) ON DELETE SET NULL,
    job_type     VARCHAR(50) NOT NULL DEFAULT 'full',
    status       VARCHAR(20) NOT NULL DEFAULT 'queued',
    progress     INTEGER DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
    eta_seconds  INTEGER,
    config_json  JSONB DEFAULT '{}',
    result_json  JSONB DEFAULT '{}',
    started_at   TIMESTAMPTZ DEFAULT NOW(),
    finished_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_pade_jobs_user
    ON pade_training_jobs(user_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_pade_jobs_status
    ON pade_training_jobs(status);

-- ── Support enquiries (Objective C - Page 08) ───────────────────────────────
CREATE TABLE IF NOT EXISTS support_enquiries (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER REFERENCES users(id) ON DELETE SET NULL,
    name            VARCHAR(255) NOT NULL,
    email           VARCHAR(255) NOT NULL,
    phone           VARCHAR(20),
    subject         VARCHAR(255) NOT NULL,
    category        VARCHAR(50) NOT NULL DEFAULT 'other',
    priority        VARCHAR(20) NOT NULL DEFAULT 'medium',
    message         TEXT NOT NULL,
    attachment_path TEXT,
    status          VARCHAR(20) NOT NULL DEFAULT 'open',
    admin_notes     TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_enquiries_user
    ON support_enquiries(user_id);
CREATE INDEX IF NOT EXISTS idx_enquiries_status
    ON support_enquiries(status);
CREATE INDEX IF NOT EXISTS idx_enquiries_created
    ON support_enquiries(created_at DESC);

-- ── Extend users table (Objective C - Page 07) ──────────────────────────────
ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(20) DEFAULT 'viewer'
    CHECK (role IN ('admin', 'analyst', 'viewer'));
ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_url TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMPTZ;
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;

-- Fix 7: backfill NULL values so legacy users can still log in
UPDATE users SET is_active = TRUE  WHERE is_active IS NULL;
UPDATE users SET role      = 'viewer' WHERE role IS NULL;
ALTER TABLE users ADD COLUMN IF NOT EXISTS phone VARCHAR(20);

-- ── User activity log ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_activity_log (
    id          BIGSERIAL PRIMARY KEY,
    user_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
    action      VARCHAR(100) NOT NULL,
    page        VARCHAR(100),
    metadata    JSONB DEFAULT '{}',
    ip_address  INET,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_activity_user
    ON user_activity_log(user_id, created_at DESC);

-- ── Notification preferences ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS notification_preferences (
    user_id             INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    email_enabled       BOOLEAN DEFAULT TRUE,
    slack_enabled       BOOLEAN DEFAULT FALSE,
    slack_webhook       TEXT,
    budget_threshold    DECIMAL(10,2),
    anomaly_sensitivity VARCHAR(20) DEFAULT 'medium',
    digest_frequency    VARCHAR(20) DEFAULT 'daily',
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── Pipeline run tags (Page 10) ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pipeline_run_tags (
    run_id  VARCHAR(36) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    tag     VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (run_id, tag)
);

-- ── Auto-update updated_at trigger for support_enquiries ─────────────────────
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_enquiries_updated_at ON support_enquiries;
CREATE TRIGGER update_enquiries_updated_at
    BEFORE UPDATE ON support_enquiries
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

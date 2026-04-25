-- CostGuard v17.0 - Post-run IEEE artifact import tables

CREATE TABLE IF NOT EXISTS ieee_import_runs (
    id          BIGSERIAL PRIMARY KEY,
    source_root TEXT NOT NULL,
    dry_run     BOOLEAN NOT NULL DEFAULT TRUE,
    status      VARCHAR(20) NOT NULL DEFAULT 'running',
    summary     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ieee_seed_runs (
    seed            INTEGER PRIMARY KEY,
    status          VARCHAR(20) NOT NULL,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    manifest_path   TEXT NOT NULL,
    metrics         JSONB NOT NULL DEFAULT '{}'::jsonb,
    model_artifacts JSONB NOT NULL DEFAULT '{}'::jsonb,
    import_batch_id BIGINT REFERENCES ieee_import_runs(id) ON DELETE SET NULL,
    imported_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ieee_aggregate_metrics (
    id              BIGSERIAL PRIMARY KEY,
    scope           VARCHAR(160) NOT NULL,
    metric_name     VARCHAR(120) NOT NULL,
    mean_value      DOUBLE PRECISION,
    std_value       DOUBLE PRECISION,
    sample_size     INTEGER,
    source_file     TEXT NOT NULL,
    import_batch_id BIGINT REFERENCES ieee_import_runs(id) ON DELETE SET NULL,
    imported_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (scope, metric_name, source_file)
);

CREATE INDEX IF NOT EXISTS idx_ieee_seed_runs_status
    ON ieee_seed_runs(status);

CREATE INDEX IF NOT EXISTS idx_ieee_seed_runs_import_batch
    ON ieee_seed_runs(import_batch_id);

CREATE INDEX IF NOT EXISTS idx_ieee_aggregate_metrics_scope
    ON ieee_aggregate_metrics(scope, metric_name);

INSERT INTO schema_migrations (version)
VALUES ('006_ieee_postrun_artifacts')
ON CONFLICT (version) DO NOTHING;


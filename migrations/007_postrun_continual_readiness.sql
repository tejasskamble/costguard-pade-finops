-- CostGuard v17.0 - Post-run analytics expansion + continual-learning readiness

CREATE TABLE IF NOT EXISTS ieee_seed_domain_metrics (
    id              BIGSERIAL PRIMARY KEY,
    seed            INTEGER NOT NULL,
    domain          VARCHAR(40) NOT NULL,
    metric_scope    VARCHAR(40) NOT NULL,
    model_name      VARCHAR(60) NOT NULL,
    metric_name     VARCHAR(120) NOT NULL,
    metric_value    DOUBLE PRECISION,
    source_path     TEXT,
    import_batch_id BIGINT REFERENCES ieee_import_runs(id) ON DELETE SET NULL,
    imported_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (seed, domain, metric_scope, model_name, metric_name)
);

CREATE INDEX IF NOT EXISTS idx_ieee_seed_domain_metrics_seed_domain
    ON ieee_seed_domain_metrics(seed, domain);

CREATE INDEX IF NOT EXISTS idx_ieee_seed_domain_metrics_metric
    ON ieee_seed_domain_metrics(metric_scope, model_name, metric_name);

CREATE TABLE IF NOT EXISTS ieee_training_runs (
    id                  BIGSERIAL PRIMARY KEY,
    seed                INTEGER NOT NULL,
    domain              VARCHAR(40) NOT NULL,
    run_number          INTEGER NOT NULL,
    run_dir             TEXT NOT NULL,
    started_at          TIMESTAMPTZ,
    elapsed_s           DOUBLE PRECISION,
    note                TEXT,
    runtime_controls    JSONB NOT NULL DEFAULT '{}'::jsonb,
    hardware_profile    JSONB NOT NULL DEFAULT '{}'::jsonb,
    training_state      JSONB NOT NULL DEFAULT '{}'::jsonb,
    config              JSONB NOT NULL DEFAULT '{}'::jsonb,
    import_batch_id     BIGINT REFERENCES ieee_import_runs(id) ON DELETE SET NULL,
    imported_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (seed, domain, run_number)
);

CREATE INDEX IF NOT EXISTS idx_ieee_training_runs_seed_domain
    ON ieee_training_runs(seed, domain);

CREATE TABLE IF NOT EXISTS ieee_prediction_summaries (
    id                  BIGSERIAL PRIMARY KEY,
    seed                INTEGER NOT NULL,
    domain              VARCHAR(40) NOT NULL,
    split_name          VARCHAR(20) NOT NULL,
    model_name          VARCHAR(40) NOT NULL,
    total_samples       INTEGER NOT NULL DEFAULT 0,
    anomaly_count       INTEGER NOT NULL DEFAULT 0,
    anomaly_rate        DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    threshold           DOUBLE PRECISION,
    mean_score          DOUBLE PRECISION,
    source_path         TEXT,
    import_batch_id     BIGINT REFERENCES ieee_import_runs(id) ON DELETE SET NULL,
    imported_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (seed, domain, split_name, model_name)
);

CREATE INDEX IF NOT EXISTS idx_ieee_prediction_summaries_seed_domain
    ON ieee_prediction_summaries(seed, domain, split_name);

CREATE TABLE IF NOT EXISTS ieee_prepared_dataset_summaries (
    id                  BIGSERIAL PRIMARY KEY,
    seed                INTEGER NOT NULL,
    domain              VARCHAR(40) NOT NULL,
    dataset_name        VARCHAR(120) NOT NULL,
    file_path           TEXT NOT NULL,
    file_name           VARCHAR(255) NOT NULL,
    file_ext            VARCHAR(20) NOT NULL,
    size_bytes          BIGINT NOT NULL DEFAULT 0,
    row_count           BIGINT,
    column_count        INTEGER,
    parse_status        VARCHAR(40) NOT NULL DEFAULT 'unknown',
    schema_preview      JSONB NOT NULL DEFAULT '[]'::jsonb,
    import_batch_id     BIGINT REFERENCES ieee_import_runs(id) ON DELETE SET NULL,
    imported_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (seed, domain, file_path)
);

CREATE INDEX IF NOT EXISTS idx_ieee_prepared_dataset_seed_domain
    ON ieee_prepared_dataset_summaries(seed, domain);

CREATE INDEX IF NOT EXISTS idx_ieee_prepared_dataset_dataset_name
    ON ieee_prepared_dataset_summaries(dataset_name);

CREATE TABLE IF NOT EXISTS user_uploaded_observations (
    id                  BIGSERIAL PRIMARY KEY,
    user_id             INTEGER REFERENCES users(id) ON DELETE SET NULL,
    source              VARCHAR(40) NOT NULL DEFAULT 'api',
    run_id              VARCHAR(64),
    stage_name          VARCHAR(100) NOT NULL,
    provider            VARCHAR(30),
    region              VARCHAR(50),
    billed_cost         DOUBLE PRECISION,
    effective_cost      DOUBLE PRECISION,
    usage_quantity      DOUBLE PRECISION,
    usage_unit          VARCHAR(40),
    branch_type         VARCHAR(50),
    executor_type       VARCHAR(50),
    payload             JSONB NOT NULL DEFAULT '{}'::jsonb,
    pending_retraining  BOOLEAN NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_uploaded_observations_pending
    ON user_uploaded_observations(pending_retraining, created_at DESC);

CREATE TABLE IF NOT EXISTS inference_events (
    id                      BIGSERIAL PRIMARY KEY,
    observation_id          BIGINT NOT NULL REFERENCES user_uploaded_observations(id) ON DELETE CASCADE,
    model_version           VARCHAR(120),
    model_checkpoint_path   TEXT,
    crs_score               DOUBLE PRECISION NOT NULL,
    anomaly_score           DOUBLE PRECISION NOT NULL,
    risk_level              VARCHAR(20) NOT NULL,
    pade_decision           VARCHAR(20) NOT NULL,
    opa_decision            VARCHAR(20),
    policy_source           VARCHAR(30),
    ai_recommendation       TEXT,
    decision_payload        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_inference_events_observation
    ON inference_events(observation_id, created_at DESC);

CREATE TABLE IF NOT EXISTS feedback_labels (
    id                  BIGSERIAL PRIMARY KEY,
    observation_id      BIGINT NOT NULL REFERENCES user_uploaded_observations(id) ON DELETE CASCADE,
    inference_event_id  BIGINT REFERENCES inference_events(id) ON DELETE SET NULL,
    user_id             INTEGER REFERENCES users(id) ON DELETE SET NULL,
    label               VARCHAR(60) NOT NULL,
    notes               TEXT,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_labels_observation
    ON feedback_labels(observation_id, created_at DESC);

CREATE TABLE IF NOT EXISTS retraining_queue (
    id                  BIGSERIAL PRIMARY KEY,
    observation_id      BIGINT NOT NULL UNIQUE REFERENCES user_uploaded_observations(id) ON DELETE CASCADE,
    status              VARCHAR(30) NOT NULL DEFAULT 'pending',
    queued_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_exported_at    TIMESTAMPTZ,
    export_metadata     JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_retraining_queue_status
    ON retraining_queue(status, queued_at DESC);

INSERT INTO schema_migrations (version)
VALUES ('007_postrun_continual_readiness')
ON CONFLICT (version) DO NOTHING;

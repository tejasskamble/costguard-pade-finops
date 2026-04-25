-- ============================================================
-- CostGuard Migration 002 — Real-Time Alert Notifications
-- FEATURE-3: PostgreSQL LISTEN/NOTIFY trigger for SSE stream
-- Idempotent — safe to run multiple times
-- ============================================================

-- Notify function: fires pg_notify on every WARN/AUTO_OPTIMISE/BLOCK insert
CREATE OR REPLACE FUNCTION notify_new_alert()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.pade_decision IN ('WARN', 'AUTO_OPTIMISE', 'BLOCK') THEN
        PERFORM pg_notify(
            'pg_costguard_alerts',
            json_build_object(
                'run_id',      NEW.run_id,
                'stage_name',  NEW.stage_name,
                'crs_score',   NEW.crs_score,
                'decision',    NEW.pade_decision,
                'cost',        NEW.billed_cost,
                'ai_rec',      COALESCE(NEW.ai_recommendation, ''),
                'created_at',  to_char(NEW.created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"')
            )::text
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop and recreate trigger (idempotent)
DROP TRIGGER IF EXISTS costguard_alert_notify ON cost_attribution;

CREATE TRIGGER costguard_alert_notify
AFTER INSERT ON cost_attribution
FOR EACH ROW EXECUTE FUNCTION notify_new_alert();

-- Record migration
INSERT INTO schema_migrations (version) VALUES ('002_add_notify_trigger')
ON CONFLICT (version) DO NOTHING;

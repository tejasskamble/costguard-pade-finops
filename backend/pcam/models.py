from sqlalchemy import (
    MetaData,
    Table,
    Column,
    BigInteger,
    String,
    Float,
    Boolean,
    Integer,
    Index,
    Numeric,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import TIMESTAMP

metadata = MetaData()

pipeline_runs = Table(
    "pipeline_runs",
    metadata,
    Column("run_id", String(36), primary_key=True),
    Column("branch_type", String(50)),
    Column("executor_type", String(50)),
    Column("provider", String(20)),
    Column("region", String(50)),
    Column("total_cost_usd", Numeric(10, 6, asdecimal=True)),
    Column("stage_count", Integer),
    Column("is_anomalous", Boolean, server_default=text("false")),
    Column("created_at", TIMESTAMP(timezone=True), server_default=text("now()")),
)

cost_attribution = Table(
    "cost_attribution",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("run_id", String(36), nullable=False),
    Column("stage_name", String(100), nullable=False),
    Column("resource_type", String(50)),
    Column("billed_cost", Numeric(10, 6, asdecimal=True)),
    Column("effective_cost", Numeric(10, 6, asdecimal=True)),
    Column("billing_currency", String(10), server_default=text("'USD'")),
    Column("usage_quantity", Numeric(10, 4, asdecimal=True)),
    Column("usage_unit", String(50)),
    Column("provider", String(20)),
    Column("region", String(50)),
    Column("cost_deviation_pct", Numeric(8, 4, asdecimal=True)),
    Column("historical_avg_cost", Numeric(10, 6, asdecimal=True)),
    Column("crs_score", Numeric(5, 4, asdecimal=True)),
    Column("pade_decision", String(20)),
    Column("window_start", TIMESTAMP(timezone=True)),
    Column("window_end", TIMESTAMP(timezone=True)),
    Column("timestamp_start", TIMESTAMP(timezone=True)),
    Column("timestamp_end", TIMESTAMP(timezone=True)),
    Column("created_at", TIMESTAMP(timezone=True), server_default=text("now()")),
)

Index("idx_cost_attribution_run_id", cost_attribution.c.run_id)
Index("idx_cost_attribution_stage", cost_attribution.c.run_id, cost_attribution.c.stage_name)
Index("idx_cost_attribution_created", cost_attribution.c.created_at.desc())

policy_config = Table(
    "policy_config",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("warn_threshold", Numeric(4, 3, asdecimal=True), server_default=text("0.50")),
    Column("auto_optimise_threshold", Numeric(4, 3, asdecimal=True), server_default=text("0.75")),
    Column("block_threshold", Numeric(4, 3, asdecimal=True), server_default=text("0.90")),
    Column("policy_bundle", JSONB, server_default=text("'{}'::jsonb")),
    Column("updated_at", TIMESTAMP(timezone=True), server_default=text("now()")),
)

"""Post-run IEEE artifacts dashboard backed by imported database tables."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from utils.api_client import (
    get_continual_export,
    get_continual_readiness,
    get_postrun_anomaly_counts,
    get_postrun_dataset_summaries,
    get_postrun_domain_metrics,
    get_postrun_import_history,
    get_postrun_model_registry,
    get_postrun_seed_metrics,
    is_authenticated,
    log_page_visit,
    run_postrun_import,
)

logger = logging.getLogger(__name__)


def _safe_rows(payload: Dict[str, Any] | None, key: str = "rows") -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get(key)
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _safe_chart_line(data: Any) -> None:
    try:
        st.line_chart(data)
    except Exception as exc:
        logger.warning("Line chart rendering failed: %s", exc)
        st.warning("Seed chart unavailable for the current dataset shape.")


def _safe_chart_bar(data: Any) -> None:
    try:
        st.bar_chart(data)
    except Exception as exc:
        logger.warning("Bar chart rendering failed: %s", exc)
        st.warning("Bar chart unavailable for the current dataset shape.")


def _domain_label(value: str) -> str:
    mapping = {
        "real": "TravisTorrent",
        "synthetic": "Synthetic",
        "bitbrains": "BitBrains",
    }
    return mapping.get(str(value).lower(), str(value))


def _registry_table(model_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for row in model_rows:
        domains = row.get("domains", []) if isinstance(row.get("domains"), list) else []
        domain_count = len(domains)
        complete_domains = 0
        artifact_total = 0
        for domain in domains:
            artifact_map = domain.get("artifacts", {}) if isinstance(domain, dict) else {}
            exists_flags = [
                bool(artifact.get("exists"))
                for artifact in artifact_map.values()
                if isinstance(artifact, dict)
            ]
            if any(exists_flags):
                complete_domains += 1
            artifact_total += sum(1 for flag in exists_flags if flag)
        records.append(
            {
                "seed": row.get("seed"),
                "status": row.get("status"),
                "domains_ready": f"{complete_domains}/{domain_count}",
                "checkpoint_artifacts": artifact_total,
                "seed_root": row.get("seed_root"),
            }
        )
    return pd.DataFrame(records)


if not is_authenticated():
    st.error("Please login to access this page.")
    st.stop()

log_page_visit("13_post_run_artifacts")

profile = st.session_state.get("user_profile", {})
user_role = str(profile.get("role", "viewer")).lower()
is_admin = user_role == "admin"

st.markdown(
    """
<div style="background:linear-gradient(135deg,rgba(255,107,53,.08) 0%,rgba(44,62,122,.06) 100%);
  border:1px solid rgba(255,107,53,.15);border-radius:16px;padding:24px 28px;margin-bottom:24px;">
  <h1 style="margin:0;color:#E8F0FE;">Post-Run IEEE Artifacts</h1>
  <p style="margin:6px 0 0;color:#6B7A99;">Database-backed overview for seeds, domains, model artifacts, OPA decisions, and continual-learning readiness.</p>
</div>
""",
    unsafe_allow_html=True,
)

if is_admin:
    with st.expander("Import Controls (Admin)", expanded=False):
        min_f1 = st.slider("Minimum ensemble F1 quality gate", 0.0, 1.0, 0.80, 0.01)
        col_a, col_b = st.columns(2)
        with col_a:
            dry_run_clicked = st.button("Run Dry-Run Import", use_container_width=True)
        with col_b:
            apply_clicked = st.button("Import Into Database", use_container_width=True)

        if dry_run_clicked:
            result = run_postrun_import(dry_run=True, min_ensemble_f1=min_f1)
            if result:
                st.success("Dry-run completed.")
                st.json(result)
        if apply_clicked:
            result = run_postrun_import(dry_run=False, min_ensemble_f1=min_f1)
            if result:
                st.success("Import request completed.")
                st.json(result)
else:
    st.info("Read-only mode: import controls are available to admin users.")

with st.spinner("Loading post-run data..."):
    model_payload = get_postrun_model_registry() or {}
    seed_metric_payload = get_postrun_seed_metrics(metric_name="f1_at_opt", model_name="ens", metric_scope="test") or {}
    domain_metric_payload = get_postrun_domain_metrics(metric_name="f1_at_opt", model_name="ens", metric_scope="test") or {}
    anomaly_payload = get_postrun_anomaly_counts(split_name="test", model_name="ens") or {}
    dataset_payload = get_postrun_dataset_summaries() or {}
    history_payload = get_postrun_import_history(limit=20) or {}
    continual_readiness_payload = get_continual_readiness(limit=200) or {}
    continual_export_payload = get_continual_export(status="pending", limit=100) or {}

model_rows = _safe_rows(model_payload, "seed_models")
seed_metric_rows = _safe_rows(seed_metric_payload)
domain_metric_rows = _safe_rows(domain_metric_payload)
anomaly_rows = _safe_rows(anomaly_payload)
dataset_rows = _safe_rows(dataset_payload)
history_rows = _safe_rows(history_payload)
continual_rows = _safe_rows(continual_readiness_payload)
continual_export_rows = _safe_rows(continual_export_payload)

registry_df = _registry_table(model_rows) if model_rows else pd.DataFrame()
complete_seed_count = int((registry_df["status"] == "complete").sum()) if not registry_df.empty else 0
domains_seen = sorted(
    {
        _domain_label(row.get("domain", ""))
        for row in seed_metric_rows
        if row.get("domain")
    }
)
latest_import = history_rows[0] if history_rows else {}
latest_import_summary = latest_import.get("summary", {}) if isinstance(latest_import, dict) else {}
if isinstance(latest_import_summary, str):
    latest_import_summary = {}
elif not isinstance(latest_import_summary, dict):
    latest_import_summary = {}
aggregate_metric_count = int(latest_import_summary.get("aggregate_metric_rows", 0) or 0)

st.markdown("### IEEE Overview")
ov1, ov2, ov3, ov4 = st.columns(4)
with ov1:
    st.metric("Total Seeds", f"{complete_seed_count}/10")
with ov2:
    st.metric("Domains", ", ".join(domains_seen) if domains_seen else "N/A")
with ov3:
    st.metric("Aggregate Metrics", aggregate_metric_count)
with ov4:
    readiness_counts = continual_readiness_payload.get("counts", {}) if isinstance(continual_readiness_payload, dict) else {}
    st.metric("Queue Pending", int(readiness_counts.get("pending", 0) or 0))

st.markdown("### Model Registry")
if not registry_df.empty:
    st.dataframe(registry_df, use_container_width=True, hide_index=True)
else:
    st.warning("Model registry is empty.")

st.markdown("### Seed Metrics")
if seed_metric_rows:
    seed_metric_df = pd.DataFrame(seed_metric_rows)
    seed_metric_df["domain"] = seed_metric_df["domain"].map(_domain_label)
    st.dataframe(seed_metric_df, use_container_width=True, hide_index=True)
    pivot_df = seed_metric_df.pivot(index="seed", columns="domain", values="metric_value")
    _safe_chart_line(pivot_df)
else:
    st.warning("Seed metric rows are missing.")

st.markdown("### Domain Comparison")
if domain_metric_rows:
    domain_metric_df = pd.DataFrame(domain_metric_rows)
    domain_metric_df["domain"] = domain_metric_df["domain"].map(_domain_label)
    st.dataframe(domain_metric_df, use_container_width=True, hide_index=True)
    _safe_chart_bar(domain_metric_df.set_index("domain")["mean_value"])
else:
    st.warning("Domain metric aggregates are missing.")

st.markdown("### Dataset Summary")
if dataset_rows:
    dataset_df = pd.DataFrame(dataset_rows)
    dataset_df["domain"] = dataset_df["domain"].map(_domain_label)
    st.dataframe(dataset_df, use_container_width=True, hide_index=True)
    dataset_chart_df = (
        dataset_df.groupby(["domain", "dataset_name"], as_index=False)["total_rows"].sum()
        .sort_values(["domain", "dataset_name"])
    )
    _safe_chart_bar(dataset_chart_df.set_index("dataset_name")["total_rows"])
else:
    st.warning("Prepared dataset summaries are missing.")

st.markdown("### Prediction and Anomaly Summary")
if anomaly_rows:
    anomaly_df = pd.DataFrame(anomaly_rows)
    anomaly_df["domain"] = anomaly_df["domain"].map(_domain_label)
    st.dataframe(anomaly_df, use_container_width=True, hide_index=True)
else:
    st.warning("Prediction anomaly summaries are missing.")

st.markdown("### OPA Panel")
latest_event = continual_rows[0] if continual_rows else {}
latest_policy_source = str(latest_event.get("policy_source", "n/a")).lower()
latest_opa_decision = str(latest_event.get("opa_decision", "n/a"))
latest_pade_decision = str(latest_event.get("pade_decision", "n/a"))

op1, op2, op3 = st.columns(3)
with op1:
    st.metric("Decision Source", latest_policy_source.upper())
with op2:
    st.metric("OPA Decision", latest_opa_decision)
with op3:
    st.metric("PADE Decision", latest_pade_decision)

if continual_rows:
    opa_df = pd.DataFrame(continual_rows)
    st.dataframe(
        opa_df[
            [
                "queued_at",
                "stage_name",
                "provider",
                "crs_score",
                "pade_decision",
                "opa_decision",
                "policy_source",
                "risk_level",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No continual events available yet.")

st.markdown("### Continual Learning")
if isinstance(continual_readiness_payload, dict):
    counts = continual_readiness_payload.get("counts", {})
    cl1, cl2, cl3, cl4 = st.columns(4)
    cl1.metric("Total", int(counts.get("total", 0) or 0))
    cl2.metric("Pending", int(counts.get("pending", 0) or 0))
    cl3.metric("Exported", int(counts.get("exported", 0) or 0))
    cl4.metric("Consumed", int(counts.get("consumed", 0) or 0))

if continual_export_rows:
    export_df = pd.DataFrame(continual_export_rows)
    st.dataframe(
        export_df[
            [
                "queue_id",
                "run_id",
                "stage_name",
                "provider",
                "billed_cost",
                "crs_score",
                "risk_level",
                "policy_source",
                "opa_decision",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No pending retraining export rows available.")

st.markdown("### Import History")
if history_rows:
    history_df = pd.DataFrame(history_rows)
    st.dataframe(history_df, use_container_width=True, hide_index=True)
else:
    detail = history_payload.get("detail") if isinstance(history_payload, dict) else None
    if detail:
        st.warning(detail)
    else:
        st.info("No import history entries found.")

"""
dashboard/pages/10_run_history.py - CostGuard v17.0

Historical run review for the canonical CostGuard platform.
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone
from components.cinematic_ui import apply_cinematic_ui, cinematic_header
from utils.api_client import (
    is_authenticated, get_pipeline_runs, get_recent_alerts, log_page_visit,
)

if not is_authenticated():
    st.error("🔒 Please login to access this page.")
    st.stop()

log_page_visit("10_run_history")
apply_cinematic_ui("10_run_history")

st.markdown(
    cinematic_header(
        "Run History Archive",
        "Searchable pipeline run history with drill-down, comparisons, and exports.",
        icon="ARCHIVE",
        status="Timeline Synced",
    ),
    unsafe_allow_html=True,
)

# ── Filters ───────────────────────────────────────────────────────────────────
with st.expander("🔍 Filters", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        date_from = st.date_input("From", value=datetime.now(timezone.utc).date() - timedelta(days=30))
    with col2:
        date_to = st.date_input("To", value=datetime.now(timezone.utc).date())
    with col3:
        provider_filter = st.multiselect("Provider", ["AWS", "GCP", "Azure", "Self-Hosted"],
                                          default=[])
    with col4:
        status_filter = st.multiselect("Anomaly Status",
                                        ["ALLOW", "WARN", "AUTO_OPTIMISE", "BLOCK"],
                                        default=[])
    col5, col6 = st.columns(2)
    with col5:
        cost_min = st.number_input("Min Cost ($)", value=0.0, step=1.0)
    with col6:
        cost_max = st.number_input("Max Cost ($)", value=10000.0, step=100.0)
    search_text = st.text_input("🔎 Search run ID or branch name", placeholder="e.g. main, feature/*, abc123")

st.markdown("---")

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading pipeline runs…"):
    runs = get_recent_alerts(limit=200) or []

# Convert to DataFrame
if runs:
    df = pd.DataFrame(runs)

    # Normalize columns
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Apply filters
    if search_text:
        mask = df.astype(str).apply(lambda row: row.str.contains(search_text, case=False).any(), axis=1)
        df = df[mask]
    if status_filter and "decision" in df.columns:
        df = df[df["decision"].isin(status_filter)]
    if provider_filter and "provider" in df.columns:
        df = df[df["provider"].isin(provider_filter)]
    if "total_cost" in df.columns:
        df = df[(df["total_cost"] >= cost_min) & (df["total_cost"] <= cost_max)]

    total_count = len(df)
    st.markdown(f"**{total_count} runs** matching current filters")

    # Paginate
    PAGE_SIZE = 50
    total_pages = max(1, (total_count + PAGE_SIZE - 1) // PAGE_SIZE)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    page_df = df.iloc[(page-1)*PAGE_SIZE : page*PAGE_SIZE].copy()

    # Decision color column
    def _decision_badge(d):
        colors = {"ALLOW": "🟢", "WARN": "🟡", "AUTO_OPTIMISE": "🟣", "BLOCK": "🔴"}
        return f"{colors.get(d, '⚪')} {d}"

    if "decision" in page_df.columns:
        page_df["Status"] = page_df["decision"].apply(_decision_badge)

    # Column selection
    show_cols = [c for c in [
        "run_id", "Status", "total_cost", "provider", "branch_type",
        "crs_score", "timestamp", "stage_count",
    ] if c in page_df.columns]

    st.dataframe(page_df[show_cols] if show_cols else page_df,
                 use_container_width=True, hide_index=True)

    # ── Drill-down ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔎 Run Drill-Down")
    available_runs = df["run_id"].tolist() if "run_id" in df.columns else []
    if available_runs:
        selected_run = st.selectbox("Select Run ID", available_runs[:100])
        if selected_run:
            run_row = df[df["run_id"] == selected_run].iloc[0] if len(df[df["run_id"] == selected_run]) > 0 else None
            if run_row is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Cost", f"${float(run_row.get('total_cost', 0)):.2f}")
                with col2:
                    st.metric("CRS Score", f"{float(run_row.get('crs_score', 0)):.4f}")
                with col3:
                    st.metric("Decision", str(run_row.get("decision", "—")))
                st.json({k: str(v) for k, v in run_row.items()})

    # ── Compare two runs ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ⚖️ Compare Two Runs")
    c1, c2 = st.columns(2)
    with c1:
        run_a = st.selectbox("Run A", available_runs[:100], key="compare_a")
    with c2:
        run_b = st.selectbox("Run B", available_runs[:100], key="compare_b",
                              index=min(1, len(available_runs)-1))

    if run_a != run_b and available_runs:
        row_a = df[df["run_id"] == run_a].iloc[0] if len(df[df["run_id"] == run_a]) > 0 else None
        row_b = df[df["run_id"] == run_b].iloc[0] if len(df[df["run_id"] == run_b]) > 0 else None
        if row_a is not None and row_b is not None:
            numeric_cols = ["total_cost", "crs_score", "stage_count"]
            compare_data = []
            for col in numeric_cols:
                if col in row_a and col in row_b:
                    va, vb = float(row_a.get(col, 0) or 0), float(row_b.get(col, 0) or 0)
                    diff = vb - va
                    pct = f"{diff/va*100:+.1f}%" if va != 0 else "—"
                    compare_data.append({"Metric": col, "Run A": f"{va:.4f}", "Run B": f"{vb:.4f}", "Diff": pct})
            if compare_data:
                st.dataframe(pd.DataFrame(compare_data), use_container_width=True, hide_index=True)

    # ── Export ─────────────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button("📥 Export All (CSV)", csv_data,
                            "pipeline_runs.csv", "text/csv", use_container_width=True)
    with col2:
        import json as _json
        json_data = df.to_json(orient="records", default_handler=str)
        st.download_button("📥 Export All (JSON)", json_data,
                            "pipeline_runs.json", "application/json", use_container_width=True)
else:
    st.info("No pipeline run data found. Ingest pipeline runs via the API to see history here.")
    st.code("""
# Example: ingest a pipeline run
POST /api/ingest
{
  "run_id": "run-abc123",
  "provider": "AWS",
  "total_cost": 42.50,
  "stages": [...]
}""", language="json")

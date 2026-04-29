"""
dashboard/pages/01_pipeline_costs.py - CostGuard v17.0

Pipeline cost overview and forecasting for the active enterprise dashboard.
"""
import random
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.cinematic_ui import apply_cinematic_ui, cinematic_header
from utils.api_client import get_recent_alerts, get_forecast, get_daily_summary

apply_cinematic_ui("01_pipeline_costs")

# ── Shared Plotly dark layout (CONSTRAINT-B) ─────────────────────────────────
PLOTLY_LAYOUT = dict(
    plot_bgcolor  = "rgba(8,12,24,0.82)",
    paper_bgcolor = "rgba(0,0,0,0)",
    font          = dict(family="DM Sans, sans-serif", color="#94A3B8", size=12),
    xaxis         = dict(
        gridcolor="rgba(99,102,241,0.1)", linecolor="rgba(99,102,241,0.15)",
        tickfont=dict(color="#6B7A99"),
    ),
    yaxis         = dict(
        gridcolor="rgba(99,102,241,0.1)", linecolor="rgba(99,102,241,0.15)",
        tickfont=dict(color="#6B7A99"),
    ),
    legend        = dict(
        bgcolor="rgba(12,20,40,0.8)", bordercolor="rgba(99,102,241,0.2)",
        borderwidth=1, font=dict(color="#94A3B8"),
    ),
    title         = dict(font=dict(family="Syne, sans-serif", color="#fff", size=16)),
    margin        = dict(l=16, r=16, t=48, b=16),
    hoverlabel    = dict(
        bgcolor="rgba(12,20,40,0.95)", bordercolor="rgba(99,102,241,0.4)",
        font=dict(family="DM Sans", color="#E8EAF6"),
    ),
)

DECISION_COLORS = {
    "ALLOW":         "#10B981",
    "WARN":          "#F59E0B",
    "AUTO_OPTIMISE": "#8B5CF6",
    "BLOCK":         "#F43F5E",
}

STAGE_ORDER = [
    "checkout", "build", "unit_test", "integration_test",
    "security_scan", "docker_build", "deploy_staging", "deploy_prod",
]


# ── KPI glass card helper (CSS animated counter, CONSTRAINT-G) ────────────────
def kpi_card(label: str, value: str, delta: str = "", color: str = "#6366F1", sub: str = "") -> str:
    delta_html = ""
    if delta:
        is_pos = "+" in delta
        d_color = "#10B981" if is_pos else "#F43F5E"
        delta_html = f'<div style="color:{d_color};font-size:.78rem;margin-top:4px;">{delta}</div>'
    sub_html = f'<div style="color:#6B7A99;font-size:.75rem;margin-top:6px;">{sub}</div>' if sub else ""
    return f"""
    <div class="glass-card" style="text-align:left;cursor:default;">
        <div style="color:#6B7A99;font-size:.72rem;text-transform:uppercase;
                    letter-spacing:.1em;margin-bottom:8px;">{label}</div>
        <div class="metric-value" style="color:{color};">{value}</div>
        {delta_html}{sub_html}
    </div>
    """


# ── Demo data generators ──────────────────────────────────────────────────────
def _demo_trend(days: int = 30) -> pd.DataFrame:
    rng   = np.random.default_rng(42)
    today = date.today()
    dates, costs, decisions = [], [], []
    for i in range(days - 1, -1, -1):
        d     = today - timedelta(days=i)
        base  = 0.08
        noise = rng.uniform(0.7, 1.4)
        wknd  = 0.5 if d.weekday() >= 5 else 1.0
        spike = 2.5 if i in (7, 18, 25) else 1.0
        cost  = round(base * noise * wknd * spike, 6)
        dec   = "BLOCK" if cost > 0.17 else "AUTO_OPTIMISE" if cost > 0.10 else "WARN" if cost > 0.07 else "ALLOW"
        dates.append(d); costs.append(cost); decisions.append(dec)
    return pd.DataFrame({"date": dates, "total_cost": costs, "decision": decisions})


def _demo_stage_costs() -> pd.DataFrame:
    rng  = np.random.default_rng(7)
    base = {"checkout":0.0012,"build":0.0180,"unit_test":0.0095,
            "integration_test":0.0240,"security_scan":0.0070,
            "docker_build":0.0310,"deploy_staging":0.0140,"deploy_prod":0.0390}
    decs = ["ALLOW","ALLOW","WARN","AUTO_OPTIMISE","ALLOW","WARN","ALLOW","BLOCK"]
    return pd.DataFrame([
        {"stage_name": s, "billed_cost": base[s]*rng.uniform(.85,1.35), "pade_decision": d}
        for s, d in zip(STAGE_ORDER, decs)
    ])


def _demo_heatmap():
    rng = np.random.default_rng(99)
    z = []
    for s in STAGE_ORDER:
        row = [round(rng.uniform(0.1,0.6)*(1.5 if 8<=h<=18 else 0.7),3) for h in range(24)]
        z.append(row)
    return STAGE_ORDER, list(range(24)), z


def _demo_provider() -> dict:
    return {"AWS": 0.0842, "GCP": 0.0451, "Azure": 0.0318, "Unknown": 0.0089}


def _demo_forecast(horizon: int = 7):
    rng   = random.Random(77)
    today = date.today()
    hist  = [{"date": str(today - timedelta(days=30-i)),
               "cost": round(0.08 + 0.01*np.sin(i/4)*rng.uniform(.9,1.1), 6)}
              for i in range(30)]
    fc    = [{"date": str(today + timedelta(days=i+1)),
               "cost": round(0.09 + i*0.002, 6),
               "lower": round(0.07 + i*0.001, 6),
               "upper": round(0.11 + i*0.003, 6)}
              for i in range(horizon)]
    return {"historical": hist, "forecast": fc, "is_demo": True}


# ── Chart renderers ───────────────────────────────────────────────────────────
def _render_forecast(forecast_data: dict) -> None:
    """FEATURE-1: Historical line + dashed forecast + shaded CI bands."""
    st.markdown('<div class="section-label">🔮 Cost Forecast — Next 7 Days</div>', unsafe_allow_html=True)
    if forecast_data.get("is_demo"):
        st.caption("ℹ️ Demo forecast — simulate pipeline runs to see real predictions.")

    hist = forecast_data.get("historical", [])
    fc   = forecast_data.get("forecast",   [])
    if not hist and not fc:
        st.info("No data available for forecast.")
        return

    hist_dates = [h["date"] for h in hist]
    hist_costs = [h["cost"] for h in hist]
    fc_dates   = [f["date"] for f in fc]
    fc_costs   = [f["cost"] for f in fc]
    fc_lower   = [f["lower"] for f in fc]
    fc_upper   = [f["upper"] for f in fc]

    # Bridge: connect last historical point to first forecast
    if hist_dates and fc_dates and False:
        bridge_date = hist_dates[-1:]
        bridge_cost = hist_costs[-1:]
    else:
        bridge_date = bridge_cost = []

    fig = go.Figure()

    # Trace 1: Historical (solid indigo line)
    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_costs,
        mode="lines+markers",
        name="Historical",
        line=dict(color="#6366F1", width=2),
        marker=dict(size=4, color="#6366F1"),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.06)",
        hovertemplate="<b>%{x}</b><br>Cost: $%{y:.4f}<extra></extra>",
    ))

    # Trace 2: CI band (shaded fill between lower and upper)
    all_fc_x = bridge_date + fc_dates + list(reversed(bridge_date + fc_dates))
    all_fc_y = (bridge_cost + fc_upper) + list(reversed(bridge_cost + fc_lower))
    fig.add_trace(go.Scatter(
        x=bridge_date + fc_dates + list(reversed(bridge_date + fc_dates)),
        y=(bridge_cost + fc_upper) + list(reversed(bridge_cost + fc_lower)),
        fill="toself",
        fillcolor="rgba(249,115,22,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="80% CI Band",
        hoverinfo="skip",
        showlegend=True,
    ))

    # Trace 3: Forecast mean (dashed coral)
    fig.add_trace(go.Scatter(
        x=bridge_date + fc_dates,
        y=bridge_cost + fc_costs,
        mode="lines+markers",
        name="Forecast (ETS)",
        line=dict(color="#F97316", width=2, dash="dash"),
        marker=dict(size=6, color="#F97316", symbol="diamond"),
        hovertemplate="<b>%{x}</b><br>Forecast: $%{y:.4f}<extra></extra>",
    ))

    forecast_layout = dict(PLOTLY_LAYOUT)
    forecast_layout["yaxis"] = dict(**PLOTLY_LAYOUT["yaxis"], tickprefix="$")
    forecast_layout["legend"] = dict(**PLOTLY_LAYOUT["legend"], orientation="h", x=0, y=1.12)
    forecast_layout["height"] = 320
    fig.update_layout(
        **forecast_layout,
        title_text="Cost Forecast — Holt-Winters ETS + 80% Confidence Band",
    )
    # Vertical divider between historical and forecast
    if hist_dates and fc_dates and False:
        fig.add_vline(
            x=hist_dates[-1], line_dash="dot",
            line_color="rgba(99,102,241,0.4)", line_width=1,
            annotation_text="Forecast →",
            annotation_font_color="#6366F1",
            annotation_position="top right",
        )
    if hist_dates and fc_dates:
        fig.add_shape(
            type="line",
            x0=hist_dates[-1],
            x1=hist_dates[-1],
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(dash="dot", color="rgba(99,102,241,0.4)", width=1),
        )
        fig.add_annotation(
            x=hist_dates[-1],
            y=1,
            xref="x",
            yref="paper",
            text="Forecast ->",
            showarrow=False,
            font=dict(color="#6366F1"),
            xanchor="left",
            yanchor="bottom",
        )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_trend(df: pd.DataFrame, is_demo: bool) -> None:
    st.markdown('<div class="section-label">📈 30-Day Cost Trend</div>', unsafe_allow_html=True)
    if is_demo:
        st.caption("ℹ️ Demo data — simulate a pipeline run to see real trends.")
    colors = [DECISION_COLORS.get(d, "#6366F1") for d in df["decision"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["total_cost"],
        mode="lines+markers",
        line=dict(color="#6366F1", width=2),
        marker=dict(size=6, color=colors),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
        name="Daily Cost",
        hovertemplate="<b>%{x}</b><br>$%{y:.4f}<extra></extra>",
    ))
    trend_layout = dict(PLOTLY_LAYOUT)
    trend_layout["title_text"] = "Daily Pipeline Spend (Last 30 Days)"
    trend_layout["yaxis"] = dict(**PLOTLY_LAYOUT["yaxis"], tickprefix="$")
    trend_layout["showlegend"] = False
    fig.update_layout(**trend_layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_stage_bar(df: pd.DataFrame, is_demo: bool) -> None:
    st.markdown('<div class="section-label">📊 Cost by Stage</div>', unsafe_allow_html=True)
    if is_demo:
        st.caption("ℹ️ Demo data.")
    df_s = df.sort_values("billed_cost", ascending=True)
    colors = [DECISION_COLORS.get(d, "#6366F1") for d in df_s["pade_decision"]]
    fig = go.Figure(go.Bar(
        y=df_s["stage_name"], x=df_s["billed_cost"],
        orientation="h", marker_color=colors, marker_line_width=0,
        text=[f"${c:.4f}" for c in df_s["billed_cost"]],
        textposition="outside",
        textfont=dict(color="rgba(255,255,255,0.7)", size=10),
        hovertemplate="<b>%{y}</b><br>$%{x:.4f}<extra></extra>",
    ))
    stage_layout = dict(PLOTLY_LAYOUT)
    stage_layout["title_text"] = "Billed Cost per Stage"
    stage_layout["xaxis"] = dict(**PLOTLY_LAYOUT["xaxis"], tickprefix="$")
    stage_layout["showlegend"] = False
    stage_layout["height"] = 340
    fig.update_layout(**stage_layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_heatmap(stages, hours, z, is_demo: bool) -> None:
    st.markdown('<div class="section-label">🔥 CRS Heatmap (Stage × Hour)</div>', unsafe_allow_html=True)
    if is_demo:
        st.caption("ℹ️ Demo heatmap.")
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{h:02d}:00" for h in hours], y=stages,
        colorscale=[[0,"#10B981"],[0.5,"#F59E0B"],[0.75,"#8B5CF6"],[1,"#F43F5E"]],
        colorbar=dict(tickfont=dict(color="#6B7A99"),
                      title=dict(text="CRS", font=dict(color="#94A3B8"))),
        hovertemplate="Stage:<b>%{y}</b><br>Hour:%{x}<br>CRS:%{z:.3f}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title_text="Mean CRS by Stage × Hour", height=320)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_provider_donut(prov: dict, is_demo: bool) -> None:
    st.markdown('<div class="section-label">☁️ Cost by Provider</div>', unsafe_allow_html=True)
    if is_demo:
        st.caption("ℹ️ Demo distribution.")
    labels = list(prov.keys())
    values = list(prov.values())
    colors = ["#F97316", "#6366F1", "#10B981", "#6B7A99"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62,
        marker=dict(colors=colors[:len(labels)], line=dict(color="rgba(0,0,0,0)", width=2)),
        textinfo="label+percent",
        textfont=dict(size=12, color="#94A3B8"),
        hovertemplate="<b>%{label}</b><br>$%{value:.4f}<br>%{percent}<extra></extra>",
    ))
    total = sum(values)
    fig.update_layout(**PLOTLY_LAYOUT, title_text="Provider Distribution",
                      showlegend=False, height=320,
                      annotations=[dict(text=f"<b>${total:.4f}</b>", x=0.5, y=0.5,
                                        showarrow=False, font=dict(size=16, color="#fff", family="Syne"))])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Main page ─────────────────────────────────────────────────────────────────
def show():
    st.markdown(
        cinematic_header(
            "Predictive Intelligence Screen",
            "30-day spend analysis, stage attribution, CRS heatmap, and 7-day AI forecast.",
            icon="ANALYTICS",
            status="Forecast Engine Active",
        ),
        unsafe_allow_html=True,
    )

    _, col_ref = st.columns([4, 1])
    with col_ref:
        if st.button("↻ Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ── Fetch data ────────────────────────────────────────────────────────────
    alerts_data  = get_recent_alerts(limit=500) or []
    forecast_raw = get_forecast(horizon_days=7)

    # ── KPI calculations ──────────────────────────────────────────────────────
    if alerts_data:
        df_all = pd.DataFrame(alerts_data)
        for col in ["billed_cost", "crs_score"]:
            if col in df_all.columns:
                df_all[col] = pd.to_numeric(df_all[col], errors="coerce").fillna(0)
        if "created_at" in df_all.columns:
            df_all["created_at"] = pd.to_datetime(df_all["created_at"], utc=True)
            df_today = df_all[df_all["created_at"].dt.date == datetime.now(timezone.utc).date()]
        else:
            df_today = df_all
        total_today     = df_today["billed_cost"].sum() if "billed_cost" in df_today else 0
        active_pipes    = df_today["run_id"].nunique() if "run_id" in df_today else 0
        avg_crs         = float(df_today["crs_score"].mean()) if "crs_score" in df_today and len(df_today) else 0.0
        anomaly_today   = len(df_today[df_today.get("pade_decision", pd.Series(dtype=str)).isin(
            ["WARN", "AUTO_OPTIMISE", "BLOCK"])]) if "pade_decision" in df_today else 0
        has_real        = True
    else:
        total_today = active_pipes = avg_crs = anomaly_today = 0
        has_real    = False

    # ── Glass KPI cards ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    avg_color = "#F59E0B" if avg_crs > 0.5 else "#10B981"
    kpis = [
        (c1, "💰 Total Spend Today",  f"${total_today:.4f}", "",  "#6366F1", "All billed stages"),
        (c2, "🔄 Active Pipelines",   str(active_pipes),      "",  "#8B5CF6", "Unique runs last 24h"),
        (c3, "🎯 Avg CRS Score",      f"{avg_crs:.3f}",       "",  avg_color, "Mean risk score today"),
        (c4, "⚠️  Anomalies Today",   str(anomaly_today),    "",  "#F43F5E", "WARN + OPT + BLOCK"),
    ]
    for col, label, value, delta, color, sub in kpis:
        with col:
            st.markdown(kpi_card(label, value, delta, color, sub), unsafe_allow_html=True)

    st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)

    # ── Forecast chart (FEATURE-1) ────────────────────────────────────────────
    if forecast_raw:
        _render_forecast(forecast_raw)
    else:
        _render_forecast(_demo_forecast(7))

    # ── 30-day trend ──────────────────────────────────────────────────────────
    if has_real and "created_at" in df_all.columns:
        df_all["date"] = df_all["created_at"].dt.date
        trend_df = df_all.groupby("date").agg(
            total_cost=("billed_cost", "sum"),
            decision=("pade_decision", lambda x: x.mode()[0] if len(x) else "ALLOW"),
        ).reset_index()
        trend_demo = len(trend_df) < 2
        if trend_demo:
            trend_df = _demo_trend()
    else:
        trend_df   = _demo_trend()
        trend_demo = True
    _render_trend(trend_df, trend_demo)

    # ── Stage bar + provider donut side by side ───────────────────────────────
    ch1, ch2 = st.columns([3, 2])
    with ch1:
        if has_real and "stage_name" in df_all.columns:
            stage_df   = df_all.groupby(["stage_name", "pade_decision"]).agg(billed_cost=("billed_cost","sum")).reset_index()
            stage_demo = len(stage_df) == 0
        else:
            stage_df   = _demo_stage_costs()
            stage_demo = True
        _render_stage_bar(stage_df, stage_demo)

    with ch2:
        if has_real and "provider" in df_all.columns:
            prov_d    = df_all.groupby("provider")["billed_cost"].sum().to_dict()
            prov_demo = not bool(prov_d)
        else:
            prov_d    = {}
            prov_demo = True
        _render_provider_donut(_demo_provider() if prov_demo else prov_d, prov_demo)

    # ── CRS heatmap ───────────────────────────────────────────────────────────
    if has_real and "stage_name" in df_all.columns and "crs_score" in df_all.columns:
        df_all["hour"] = df_all["created_at"].dt.hour
        pivot = df_all.pivot_table(index="stage_name", columns="hour", values="crs_score", aggfunc="mean")
        stages_h = [s for s in STAGE_ORDER if s in pivot.index]
        z_h = [[float(pivot.loc[s, h]) if h in pivot.columns else 0.0 for h in range(24)] for s in stages_h]
        heat_demo = len(stages_h) == 0
        if heat_demo:
            stages_h, _, z_h = _demo_heatmap()
    else:
        stages_h, _, z_h = _demo_heatmap()
        heat_demo = True
    _render_heatmap(stages_h, list(range(24)), z_h, heat_demo)


show()

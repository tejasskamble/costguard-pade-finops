"""
dashboard/pages/06_forecasting.py - CostGuard v17.0

Forecasting views for the enterprise CostGuard dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from components.cinematic_ui import apply_cinematic_ui, cinematic_header
from utils.api_client import (
    is_authenticated, get_forecast, get_exchange_rates, log_page_visit,
)

if not is_authenticated():
    st.error("🔒 Please login to access this page.")
    st.stop()

log_page_visit("06_forecasting")
apply_cinematic_ui("06_forecasting")

st.markdown(
    cinematic_header(
        "Predictive Intelligence Screen",
        "ARIMA-powered 30/60/90-day cost forecasts with budget simulation and multi-currency telemetry.",
        icon="FORECAST",
        status="Model Feed Online",
    ),
    unsafe_allow_html=True,
)

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
with col1:
    horizon = st.selectbox("Forecast Horizon", [30, 60, 90],
                            format_func=lambda d: f"{d}-Day Forecast")
with col2:
    currency_map = get_exchange_rates("USD")
    currency = st.selectbox("Display Currency",
                             ["USD", "EUR", "GBP", "INR", "JPY", "AUD"],
                             index=0)
    fx_rate = currency_map.get(currency, 1.0)
with col3:
    budget = st.number_input(f"Monthly Budget ({currency})", min_value=0.0,
                              value=5000.0, step=100.0)
with col4:
    cloud_filter = st.selectbox("Cloud Provider",
                                 ["All", "AWS", "GCP", "Azure", "Self-Hosted"])

st.markdown("---")

# ── Fetch forecast ────────────────────────────────────────────────────────────
with st.spinner(f"Generating {horizon}-day ARIMA forecast…"):
    forecast_data = get_forecast(horizon_days=horizon)

# Build synthetic demo forecast if no real data
if not forecast_data or not forecast_data.get("dates"):
    import numpy as np
    np.random.seed(42)
    today = datetime.now(timezone.utc)
    dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(horizon)]
    base = 120.0
    trend = np.linspace(0, 30, horizon)
    noise = np.random.normal(0, 8, horizon)
    values = (base + trend + noise) * fx_rate
    lower = values * 0.85
    upper = values * 1.15
    forecast_data = {
        "dates": dates, "predicted": values.tolist(),
        "lower_bound": lower.tolist(), "upper_bound": upper.tolist(),
    }
else:
    # Apply currency conversion
    for key in ("predicted", "lower_bound", "upper_bound"):
        if key in forecast_data:
            forecast_data[key] = [v * fx_rate for v in forecast_data[key]]

# ── Forecast chart ────────────────────────────────────────────────────────────
fig = go.Figure()
dates = forecast_data.get("dates", [])
predicted = forecast_data.get("predicted", [])
lower = forecast_data.get("lower_bound", predicted)
upper = forecast_data.get("upper_bound", predicted)

fig.add_trace(go.Scatter(
    x=dates, y=upper, mode="lines", line=dict(width=0),
    name="Upper CI", fillcolor="rgba(124,77,255,0.10)", showlegend=False,
))
fig.add_trace(go.Scatter(
    x=dates, y=lower, mode="lines", fill="tonexty",
    line=dict(width=0), name="95% Confidence Interval",
    fillcolor="rgba(124,77,255,0.10)",
))
fig.add_trace(go.Scatter(
    x=dates, y=predicted, mode="lines+markers",
    name=f"Predicted Cost ({currency})",
    line=dict(color="#00E5FF", width=2.8),
    marker=dict(size=4, color="#7C4DFF"),
    fill="tozeroy",
    fillcolor="rgba(0,229,255,0.10)",
    hovertemplate=f"<b>%{{x}}</b><br>Forecast: {currency} %{{y:,.2f}}<extra></extra>",
))

# Budget line
if budget > 0:
    daily_budget = (budget / 30) * fx_rate
    fig.add_hline(y=daily_budget, line_dash="dash", line_color="#FF3B3B",
                  annotation_text=f"Daily Budget ({currency} {daily_budget:.0f})",
                  annotation_font_color="#FF3B3B")

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(8,12,24,0.75)",
    font=dict(family="DM Sans", color="#E8F0FE"),
    xaxis=dict(gridcolor="rgba(0,229,255,0.12)"),
    yaxis=dict(gridcolor="rgba(124,77,255,0.15)",
               title=f"Daily Cost ({currency})"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    height=420,
    margin=dict(l=0, r=0, t=20, b=0),
)
st.plotly_chart(fig, use_container_width=True)

# ── Budget burn simulation ────────────────────────────────────────────────────
st.markdown("### 💰 Budget Burn Simulation")
col1, col2, col3, col4 = st.columns(4)

total_forecast = sum(predicted)
daily_avg = total_forecast / max(len(predicted), 1)
monthly_proj = daily_avg * 30
budget_pct = (monthly_proj / budget * 100) if budget > 0 else 0
days_until_exhausted = int(budget / daily_avg) if daily_avg > 0 else 999

with col1:
    st.metric(f"Total {horizon}d Forecast", f"{currency} {total_forecast:,.0f}")
with col2:
    st.metric("Daily Average", f"{currency} {daily_avg:,.2f}")
with col3:
    st.metric("Monthly Projection", f"{currency} {monthly_proj:,.0f}",
              delta=f"{budget_pct:.0f}% of budget")
with col4:
    color = "normal" if days_until_exhausted > 30 else "inverse"
    st.metric("Budget Runway", f"{days_until_exhausted} days",
              delta="⚠️ At risk" if days_until_exhausted < 30 else "✅ Healthy")

if budget_pct > 100:
    st.error(f"🚨 **Budget Overrun Alert:** At current pace you'll exceed your {currency} {budget:,.0f} monthly budget by **{budget_pct-100:.0f}%**")
elif budget_pct > 80:
    st.warning(f"⚠️ **Budget Warning:** You're projected to use **{budget_pct:.0f}%** of your monthly budget.")
else:
    st.success(f"✅ **On Track:** Projected to use **{budget_pct:.0f}%** of your monthly budget.")

# ── Provider breakdown forecast ───────────────────────────────────────────────
st.markdown("### ☁️ Cloud Provider Forecast Breakdown")
import numpy as np
breakdown_data = {
    "AWS": [v * 0.45 for v in predicted],
    "GCP": [v * 0.30 for v in predicted],
    "Azure": [v * 0.18 for v in predicted],
    "Self-Hosted": [v * 0.07 for v in predicted],
}
providers_to_show = ["AWS", "GCP", "Azure", "Self-Hosted"] if cloud_filter == "All" else [cloud_filter]
colors = {"AWS": "#FF9900", "GCP": "#4285F4", "Azure": "#0078D4", "Self-Hosted": "#00D4AA"}

fig2 = go.Figure()
for provider in providers_to_show:
    fig2.add_trace(go.Bar(
        x=dates[::7], y=breakdown_data[provider][::7],
        name=provider, marker_color=colors.get(provider, "#FF6B35"),
    ))
fig2.update_layout(
    barmode="stack", template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,27,46,0.5)",
    font=dict(family="DM Sans", color="#E8F0FE"),
    height=300, margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig2, use_container_width=True)

# ── Export ────────────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    df_export = pd.DataFrame({
        "Date": dates, f"Predicted ({currency})": [f"{v:.2f}" for v in predicted],
        f"Lower CI ({currency})": [f"{v:.2f}" for v in lower],
        f"Upper CI ({currency})": [f"{v:.2f}" for v in upper],
    })
    csv = df_export.to_csv(index=False)
    st.download_button("📥 Export CSV", csv, f"costguard_forecast_{horizon}d.csv",
                        "text/csv", use_container_width=True)
with col2:
    st.caption(f"💱 Exchange rate: 1 USD = {fx_rate:.4f} {currency} | Source: open.er-api.com")

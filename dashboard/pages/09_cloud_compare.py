"""
dashboard/pages/09_cloud_compare.py - CostGuard v17.0

Cross-provider cost and anomaly comparison views.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from components.cinematic_ui import apply_cinematic_ui, cinematic_header
from utils.api_client import is_authenticated, get_exchange_rates, log_page_visit

if not is_authenticated():
    st.error("🔒 Please login to access this page.")
    st.stop()

log_page_visit("09_cloud_compare")
apply_cinematic_ui("09_cloud_compare")

st.markdown(
    cinematic_header(
        "Cloud Cost Comparator",
        "Side-by-side AWS, GCP, and Azure cost analysis with currency conversion and savings simulation.",
        icon="CLOUD",
        status="Multi-Cloud Scanner Online",
    ),
    unsafe_allow_html=True,
)

# ── Currency ──────────────────────────────────────────────────────────────────
rates = get_exchange_rates("USD")
col_cur, col_reg = st.columns([2, 3])
with col_cur:
    currency = st.selectbox("Display Currency",
                             ["USD", "EUR", "GBP", "INR", "JPY", "AUD", "SGD", "CAD"])
    fx = rates.get(currency, 1.0)
    st.caption(f"💱 1 USD = {fx:.4f} {currency} · via open.er-api.com")
with col_reg:
    region = st.selectbox("Region", [
        "US East (N. Virginia)", "US West (Oregon)",
        "Europe (Frankfurt)", "Asia Pacific (Mumbai)",
        "Asia Pacific (Singapore)", "South America (São Paulo)",
    ])

st.markdown("---")

# ── Workload sliders ──────────────────────────────────────────────────────────
st.markdown("### ⚙️ Configure Workload")
col1, col2, col3, col4 = st.columns(4)
with col1:
    cpu = st.slider("vCPU Cores", 1, 128, 8)
with col2:
    ram = st.slider("RAM (GB)", 1, 512, 32)
with col3:
    storage = st.slider("Storage (TB)", 0.1, 100.0, 2.0, 0.1)
with col4:
    transfer = st.slider("Data Transfer (TB/mo)", 0.0, 100.0, 1.0, 0.5)

# ── Cloud pricing model (simplified USD/month estimates) ─────────────────────
REGION_MULT = {
    "US East (N. Virginia)": 1.00, "US West (Oregon)": 1.00,
    "Europe (Frankfurt)": 1.12,    "Asia Pacific (Mumbai)": 0.95,
    "Asia Pacific (Singapore)": 1.08, "South America (São Paulo)": 1.35,
}
rm = REGION_MULT.get(region, 1.0)

def _compute_costs(cpu, ram, storage_tb, transfer_tb, rm):
    """Return monthly cost estimates in USD for AWS, GCP, Azure."""
    # AWS pricing model (general purpose, on-demand)
    aws = (
        cpu * 35 * rm +          # vCPU compute
        ram * 4.5 * rm +          # RAM
        storage_tb * 1024 * 0.023 * rm +  # EBS gp3
        transfer_tb * 1024 * 0.09 * rm    # Data transfer out
    )
    # GCP pricing model (~8% cheaper on compute)
    gcp = (
        cpu * 32 * rm +
        ram * 4.2 * rm +
        storage_tb * 1024 * 0.020 * rm +
        transfer_tb * 1024 * 0.085 * rm
    )
    # Azure pricing model
    azure = (
        cpu * 34 * rm +
        ram * 4.6 * rm +
        storage_tb * 1024 * 0.024 * rm +
        transfer_tb * 1024 * 0.087 * rm
    )
    return aws, gcp, azure

aws_usd, gcp_usd, azure_usd = _compute_costs(cpu, ram, storage, transfer, rm)

# Convert to selected currency
aws_c = aws_usd * fx
gcp_c = gcp_usd * fx
azure_c = azure_usd * fx

best = min([("AWS", aws_c), ("GCP", gcp_c), ("Azure", azure_c)], key=lambda x: x[1])
worst = max([("AWS", aws_c), ("GCP", gcp_c), ("Azure", azure_c)], key=lambda x: x[1])
savings = worst[1] - best[1]
savings_pct = (savings / worst[1] * 100) if worst[1] > 0 else 0

# ── Provider cards ────────────────────────────────────────────────────────────
st.markdown("### 💰 Monthly Cost Estimate")

PROVIDER_META = {
    "AWS": {"icon": "🟧", "color": "#FF9900", "desc": "Amazon Web Services"},
    "GCP": {"icon": "🔵", "color": "#4285F4", "desc": "Google Cloud Platform"},
    "Azure": {"icon": "🟦", "color": "#0078D4", "desc": "Microsoft Azure"},
}

col1, col2, col3 = st.columns(3)
for col, (name, cost_c, cost_usd) in zip(
    [col1, col2, col3],
    [("AWS", aws_c, aws_usd), ("GCP", gcp_c, gcp_usd), ("Azure", azure_c, azure_usd)]
):
    meta = PROVIDER_META[name]
    is_best = name == best[0]
    border = "#00D4AA" if is_best else "rgba(255,107,53,.15)"
    badge = '<div style="background:rgba(0,212,170,.1);color:#00D4AA;border:1px solid rgba(0,212,170,.3);padding:3px 10px;border-radius:20px;font-size:.7rem;font-weight:700;display:inline-block;margin-bottom:8px;">⭐ BEST VALUE</div>' if is_best else ""
    with col:
        st.markdown(f"""
        <div style="background:#0D1B2E;border:2px solid {border};border-radius:14px;padding:20px 22px;text-align:center;">
          {badge}
          <div style="font-size:2rem;margin-bottom:8px;">{meta['icon']}</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:{meta['color']};">{name}</div>
          <div style="font-size:.8rem;color:#6B7A99;margin-bottom:12px;">{meta['desc']}</div>
          <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#fff;">
            {currency} {cost_c:,.0f}
          </div>
          <div style="font-size:.8rem;color:#6B7A99;margin-top:4px;">per month</div>
          <div style="font-size:.75rem;color:#444D5E;margin-top:4px;">USD {cost_usd:,.0f} / mo</div>
        </div>""", unsafe_allow_html=True)

# ── Savings callout ───────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:rgba(0,212,170,.06);border:1px solid rgba(0,212,170,.2);
            border-radius:12px;padding:20px 24px;margin:20px 0;text-align:center;">
  <span style="font-size:1.5rem;">💡</span>
  <strong style="color:#00D4AA;font-size:1.1rem;">
    Switch from {worst[0]} → {best[0]} to save {currency} {savings:,.0f}/month ({savings_pct:.1f}%)
  </strong>
  <div style="color:#6B7A99;font-size:.85rem;margin-top:4px;">
    Annual savings potential: <strong style="color:#00D4AA;">{currency} {savings*12:,.0f}</strong>
  </div>
</div>""", unsafe_allow_html=True)

# ── Cost breakdown pie charts ─────────────────────────────────────────────────
st.markdown("### 📊 Cost Breakdown by Component")
fig_cols = st.columns(3)
for i, (name, cpu_cost, ram_cost, storage_cost, transfer_cost) in enumerate([
    ("AWS", cpu*35*rm*fx, ram*4.5*rm*fx, storage*1024*0.023*rm*fx, transfer*1024*0.09*rm*fx),
    ("GCP", cpu*32*rm*fx, ram*4.2*rm*fx, storage*1024*0.020*rm*fx, transfer*1024*0.085*rm*fx),
    ("Azure", cpu*34*rm*fx, ram*4.6*rm*fx, storage*1024*0.024*rm*fx, transfer*1024*0.087*rm*fx),
]):
    meta = PROVIDER_META[name]
    fig = go.Figure(go.Pie(
        labels=["Compute (CPU)", "Memory (RAM)", "Storage", "Data Transfer"],
        values=[max(cpu_cost,0.01), max(ram_cost,0.01), max(storage_cost,0.01), max(transfer_cost,0.01)],
        hole=0.6,
        marker_colors=["#FF6B35", "#C084FC", "#4A9EFF", "#00D4AA"],
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False, height=220,
        title=dict(text=f"{meta['icon']} {name}", font=dict(color=meta['color'], size=14)),
        margin=dict(l=10, r=10, t=40, b=10),
        annotations=[dict(
            text=f"{currency}<br>{[aws_c,gcp_c,azure_c][i]:,.0f}",
            x=0.5, y=0.5, font_size=11, showarrow=False, font_color="#E8F0FE",
        )],
    )
    with fig_cols[i]:
        st.plotly_chart(fig, use_container_width=True)

# ── Savings bar chart ─────────────────────────────────────────────────────────
st.markdown("### 📈 Cost Comparison Bar Chart")
fig_bar = go.Figure(go.Bar(
    x=["AWS", "GCP", "Azure"],
    y=[aws_c, gcp_c, azure_c],
    marker_color=["#FF9900", "#4285F4", "#0078D4"],
    text=[f"{currency} {v:,.0f}" for v in [aws_c, gcp_c, azure_c]],
    textposition="auto",
))
fig_bar.update_layout(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,27,46,0.5)",
    font=dict(family="DM Sans", color="#E8F0FE"),
    yaxis_title=f"Monthly Cost ({currency})",
    height=320, margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown(f"""
<div style="text-align:right;font-size:.75rem;color:#444D5E;">
  Pricing estimates are illustrative and based on on-demand rates (2024).<br>
  Exchange rates from <strong>open.er-api.com</strong> (free tier, no API key).
  Actual costs may vary by region, commitment, and usage tier.
</div>""", unsafe_allow_html=True)

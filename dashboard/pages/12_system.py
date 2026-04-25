"""CostGuard system health and platform information page."""
import time

import requests as _req
import streamlit as st

from utils.api_client import (
    get_api_http_base_url,
    get_pade_status,
    is_authenticated,
    log_page_visit,
)

if not is_authenticated():
    st.error('Please login to access this page.')
    st.stop()

log_page_visit('12_system')

API_BASE_HTTP = get_api_http_base_url()

st.markdown(
    """
<style>
.cg-page-header{background:linear-gradient(135deg,rgba(255,107,53,.08) 0%,rgba(44,62,122,.06) 100%);
  border:1px solid rgba(255,107,53,.15);border-radius:16px;padding:24px 28px;margin-bottom:24px;}
.cg-page-header h1{font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;color:#fff;margin:0;}
.cg-page-header p{color:#6B7A99;margin:6px 0 0;}
.health-card{background:#0D1B2E;border:1px solid rgba(255,107,53,.15);border-radius:14px;
  padding:20px 22px;text-align:center;}
.health-ok{border-color:rgba(0,212,170,.3);}
.health-warn{border-color:rgba(255,149,0,.3);}
.health-err{border-color:rgba(255,59,92,.3);}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="cg-page-header">
  <h1>System Health and About</h1>
  <p>Real-time backend health, model readiness, governance status, and platform metadata for CostGuard v17.0.</p>
</div>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(['Health', 'About', 'Technology Stack'])

with tab1:
    col_ref, _ = st.columns([1, 4])
    with col_ref:
        if st.button('Refresh Status'):
            st.rerun()

    st.markdown('#### Service Health Cards')

    t0 = time.time()
    try:
        health_resp = _req.get(f'{API_BASE_HTTP}/health', timeout=5)
        latency_ms = (time.time() - t0) * 1000
        health_ok = health_resp.status_code == 200
        health_data = health_resp.json() if health_ok else {}
        health_status = 'Healthy' if health_ok else 'Unhealthy'
        health_cls = 'health-ok' if health_ok else 'health-err'
    except Exception:
        latency_ms = (time.time() - t0) * 1000
        health_status = 'Unreachable'
        health_cls = 'health-err'
        health_data = {}

    pade_data = get_pade_status() or {}
    pade_ok = bool(pade_data.get('status') == 'ok' or pade_data.get('model_loaded'))
    pade_status = 'Model Loaded' if pade_ok else 'Fallback Ready'
    pade_cls = 'health-ok' if pade_ok else 'health-warn'

    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ('API Backend', health_status, health_cls, f"v{health_data.get('version', '-')}", f'Latency: {latency_ms:.0f}ms'),
        ('PADE Engine', pade_status, pade_cls, pade_data.get('model_type', 'canonical-pade'), f"CRS ready: {pade_data.get('crs_ready', '-')}"),
        ('Governance', health_data.get('governance', 'inline-fallback'), 'health-ok' if health_ok else 'health-warn', 'OPA + inline policy parity', 'Structured cost-control decisions'),
        ('Domains', 'D0 -> L1 -> L2', 'health-ok', 'Synthetic -> TravisTorrent -> BitBrains', 'Immutable IEEE sequence'),
    ]

    for column, (title, status, css, sub1, sub2) in zip([col1, col2, col3, col4], cards):
        with column:
            st.markdown(
                f"""
            <div class="health-card {css}">
              <div style="font-size:.8rem;color:#6B7A99;margin-bottom:6px;">{title}</div>
              <div style="font-weight:700;color:#E8F0FE;font-size:.95rem;">{status}</div>
              <div style="font-size:.75rem;color:#6B7A99;margin-top:4px;">{sub1}</div>
              <div style="font-size:.73rem;color:#444D5E;">{sub2}</div>
            </div>""",
                unsafe_allow_html=True,
            )

    st.markdown('---')
    st.markdown('#### API Latency History (this session)')
    if 'latency_history' not in st.session_state:
        st.session_state['latency_history'] = []
    st.session_state['latency_history'].append(latency_ms)
    st.session_state['latency_history'] = st.session_state['latency_history'][-30:]

    history = st.session_state['latency_history']
    if len(history) > 1:
        import plotly.graph_objects as go

        fig = go.Figure(
            go.Scatter(
                y=history,
                mode='lines+markers',
                line=dict(color='#FF6B35', width=2),
                marker=dict(size=4, color='#FF6B35'),
                fill='tozeroy',
                fillcolor='rgba(255,107,53,.08)',
            )
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(13,27,46,0.5)',
            height=200,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title='Latency (ms)',
            xaxis_title='Request #',
            font=dict(family='DM Sans', color='#E8F0FE'),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption('Latency history populates as you use the dashboard.')

    if health_data:
        with st.expander('Raw Health Response'):
            st.json(health_data)
    if pade_data:
        with st.expander('Raw PADE Status'):
            st.json(pade_data)

with tab2:
    st.markdown(
        """
    <div style="background:#0D1B2E;border:1px solid rgba(255,107,53,.15);border-radius:16px;padding:32px;">
      <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:900;
                  background:linear-gradient(135deg,#FF6B35,#C084FC);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  margin-bottom:12px;">CostGuard v17.0</div>
      <div style="color:#B0BDD0;font-size:.95rem;line-height:1.8;margin-bottom:20px;">
        <strong style="color:#E8F0FE;">CostGuard</strong> is a FinOps intelligence platform for automated cloud cost monitoring,
        anomaly detection, policy-based governance, and pipeline optimization. The active research stack is the
        <strong style="color:#FF6B35;">canonical 3-domain PADE pipeline</strong> with structured OPA governance and unified IEEE analytics.
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px;">
        <div style="background:#111D35;border-radius:10px;padding:16px;">
          <div style="color:#FF6B35;font-weight:700;margin-bottom:8px;">Academic Info</div>
          <div style="color:#B0BDD0;font-size:.85rem;line-height:1.8;">
            Institution: Sir Parshurambhau College (Autonomous), Pune<br>
            Programme: M.Sc. Computer Science 2025-26<br>
            Target venues: IEEE Software / IEEE TNNLS<br>
            Theme: Graph-based anomaly detection and autonomous optimization for CI/CD pipelines
          </div>
        </div>
        <div style="background:#111D35;border-radius:10px;padding:16px;">
          <div style="color:#FF6B35;font-weight:700;margin-bottom:8px;">System Info</div>
          <div style="color:#B0BDD0;font-size:.85rem;line-height:1.8;">
            Version: 17.0.0-enterprise<br>
            PADE: Canonical C4 + C5 architectures<br>
            Governance: OPA + inline parity fallback<br>
            Domains: Synthetic, TravisTorrent, BitBrains
          </div>
        </div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('---')
    st.markdown('#### Changelog')
    changelog = {
        'v17.0': 'Canonical 3-domain IEEE stack, structured OPA governance, unified analytics core, BWT-aligned reporting',
        'Legacy Support Release': 'Forgot-password OTP, ML Training Lab, support enquiry system, premium dashboard pages',
        'Legacy Enterprise Release': 'Budget guardrails, async job queue, and multi-cloud analytics foundations',
        'Legacy Research Release': 'PADE ensemble integration with live ETA tracking and anomaly scoring',
    }
    for version, notes in changelog.items():
        st.markdown(f'**{version}** - {notes}')

with tab3:
    st.markdown('#### Technology Stack')
    stack = [
        ('FastAPI 0.111', 'High-performance async Python API framework'),
        ('PostgreSQL + asyncpg', 'Transactional storage with async access'),
        ('Streamlit 1.35', 'Enterprise dashboard for operations and research workflows'),
        ('PyTorch + PyG', 'Canonical C4/C5 anomaly models and graph processing'),
        ('OPA + Rego', 'Policy-based governance with structured decision responses'),
        ('OpenAI', 'Natural language cost analysis and anomaly recommendations'),
        ('Prometheus + Grafana', 'Metrics, observability, and dashboard health monitoring'),
        ('Unified IEEE Analytics', 'Aggregate reporting, BWT computation, LaTeX, and vector figures'),
        ('Docker Compose', 'Local enterprise deployment with optional OPA and observability services'),
    ]
    cols = st.columns(2)
    for index, (tech, desc) in enumerate(stack):
        with cols[index % 2]:
            st.markdown(
                f"""
            <div style="background:#0D1B2E;border:1px solid rgba(255,107,53,.1);border-radius:10px;
                        padding:12px 16px;margin:6px 0;display:flex;gap:12px;align-items:center;">
              <div>
                <div style="font-weight:600;color:#E8F0FE;font-size:.85rem;">{tech}</div>
                <div style="font-size:.76rem;color:#6B7A99;">{desc}</div>
              </div>
            </div>""",
                unsafe_allow_html=True,
            )

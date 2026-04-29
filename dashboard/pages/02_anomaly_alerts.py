"""
dashboard/pages/02_anomaly_alerts.py - CostGuard v17.0

Live anomaly alert monitoring, DAG inspection, and remediation downloads.
"""
import random
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.cinematic_ui import apply_cinematic_ui, cinematic_header
from utils.api_client import (
    get_recent_alerts, simulate_pipeline, get_dag, remediate, get_api_http_base_url
)

apply_cinematic_ui("02_anomaly_alerts")

PLOTLY_LAYOUT = dict(
    plot_bgcolor  = "#0C1428",
    paper_bgcolor = "#0C1428",
    font          = dict(family="DM Sans, sans-serif", color="#94A3B8", size=12),
    margin        = dict(l=16, r=16, t=48, b=16),
    hoverlabel    = dict(
        bgcolor="rgba(12,20,40,0.95)", bordercolor="rgba(99,102,241,0.4)",
        font=dict(family="DM Sans", color="#E8EAF6"),
    ),
)

STAGE_ORDER = [
    "checkout", "build", "unit_test", "integration_test",
    "security_scan", "docker_build", "deploy_staging", "deploy_prod",
]

DECISION_COLORS = {
    "ALLOW":         "#10B981",
    "WARN":          "#F59E0B",
    "AUTO_OPTIMISE": "#8B5CF6",
    "BLOCK":         "#F43F5E",
}

NODE_POSITIONS = {
    "checkout":         (0.50, 0.90),
    "build":            (0.30, 0.70),
    "unit_test":        (0.70, 0.70),
    "integration_test": (0.50, 0.50),
    "security_scan":    (0.30, 0.30),
    "docker_build":     (0.70, 0.30),
    "deploy_staging":   (0.40, 0.10),
    "deploy_prod":      (0.60, 0.10),
}

DAG_EDGES = [
    ("checkout", "build"), ("checkout", "unit_test"),
    ("build", "integration_test"), ("unit_test", "integration_test"),
    ("integration_test", "security_scan"), ("integration_test", "docker_build"),
    ("security_scan", "deploy_staging"), ("docker_build", "deploy_staging"),
    ("deploy_staging", "deploy_prod"),
]


def badge_html(decision: str) -> str:
    cls = {
        "ALLOW": "badge-allow", "WARN": "badge-warn",
        "AUTO_OPTIMISE": "badge-optimise", "BLOCK": "badge-block",
    }.get(str(decision), "badge-allow")
    label = {"ALLOW":"Allow","WARN":"Warn","AUTO_OPTIMISE":"Optimise","BLOCK":"Block"}.get(str(decision), str(decision))
    return f'<span class="badge {cls}">{label}</span>'


def crs_color(score: float) -> str:
    if score < 0.5:  return "#10B981"
    if score < 0.75: return "#F59E0B"
    if score < 0.9:  return "#8B5CF6"
    return "#F43F5E"


# ── Demo alerts ───────────────────────────────────────────────────────────────
def _demo_alerts() -> list:
    rng  = random.Random(12)
    rows = []
    decs = ["WARN", "AUTO_OPTIMISE", "BLOCK", "WARN", "AUTO_OPTIMISE"]
    for i, dec in enumerate(decs):
        crs = {"WARN":0.60,"AUTO_OPTIMISE":0.80,"BLOCK":0.92}[dec] + rng.uniform(-.03,.03)
        rows.append({
            "run_id":           f"demo-{i:04d}-aaaa-bbbb-cccc",
            "stage_name":       STAGE_ORDER[i % len(STAGE_ORDER)],
            "crs_score":        round(crs, 4),
            "pade_decision":    dec,
            "billed_cost":      round(rng.uniform(.005,.05), 6),
            "ai_recommendation": (
                f"The {STAGE_ORDER[i % len(STAGE_ORDER)]} stage has a CRS of {crs:.3f}, "
                f"indicating a cost spike above baseline. "
                f"{'Switch to spot instances' if dec=='BLOCK' else 'Enable build caching'} immediately. "
                f"Expected savings: {'65%' if dec=='BLOCK' else '30%'} on next run."
            ),
            "created_at": (datetime.now(timezone.utc) - timedelta(minutes=i*17)).isoformat(),
        })
    return rows


# ── CRS gauge ─────────────────────────────────────────────────────────────────
def _gauge(crs_val: float, title: str = "Peak CRS") -> go.Figure:
    color = crs_color(crs_val)
    fig   = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=crs_val,
        number=dict(font=dict(family="Syne",color="#fff",size=40),valueformat=".3f"),
        delta=dict(reference=0.5,valueformat=".3f",
                   increasing=dict(color="#F43F5E"),decreasing=dict(color="#10B981")),
        domain={"x":[0,1],"y":[0,1]},
        title=dict(text=title,font=dict(family="Syne",color="#94A3B8",size=13)),
        gauge=dict(
            axis=dict(range=[0,1],tickcolor="#374151",tickfont=dict(color="#6B7A99",size=10)),
            bar=dict(color=color,thickness=0.25),
            bgcolor="rgba(255,255,255,0.03)", borderwidth=0,
            steps=[
                dict(range=[0,0.5],  color="rgba(16,185,129,0.12)"),
                dict(range=[0.5,0.75],color="rgba(245,158,11,0.12)"),
                dict(range=[0.75,0.9],color="rgba(139,92,246,0.12)"),
                dict(range=[0.9,1],  color="rgba(244,63,94,0.12)"),
            ],
            threshold=dict(line=dict(color=color,width=3),thickness=0.8,value=crs_val),
        ),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=280)
    return fig


# ── FEATURE-2: DAG Explorer ───────────────────────────────────────────────────
def render_dag_explorer(run_id: str = None) -> None:
    """
    Interactive GATv2 Pipeline DAG with Plotly network graph.
    Node colour = CRS score (emerald→amber→violet→rose).
    Node size scales with billed_cost.
    CORRECT pattern from few-shot spec — no st.graphviz_chart.
    """
    dag_data = get_dag(run_id)
    is_demo  = dag_data is None
    if is_demo:
        # Build demo DAG
        rng = random.Random(42)
        nodes = [{
            "id": s, "label": s.replace("_", "\n"),
            "crs": round(rng.uniform(0.1, 0.7), 3),
            "billed_cost": round(rng.uniform(0.005, 0.04), 6),
            "decision": "ALLOW", "ai_rec": "",
            "x": NODE_POSITIONS[s][0], "y": NODE_POSITIONS[s][1],
        } for s in STAGE_ORDER]
        edges = [{"src": s, "dst": d} for s, d in DAG_EDGES]
    else:
        nodes = dag_data.get("nodes", [])
        edges = dag_data.get("edges", [])

    if is_demo:
        st.caption("ℹ️ Demo DAG — simulate a run to see live CRS scores per stage.")

    # Build edge traces
    edge_x, edge_y = [], []
    for e in edges:
        src = e.get("src") or e.get("source")
        dst = e.get("dst") or e.get("target")
        if src not in NODE_POSITIONS or dst not in NODE_POSITIONS:
            continue
        x0, y0 = NODE_POSITIONS[src]
        x1, y1 = NODE_POSITIONS[dst]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    fig = go.Figure()

    # Edge trace
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1.5, color="rgba(99,102,241,0.3)"),
        hoverinfo="none",
        showlegend=False,
    ))

    # Simulated flow packets along each edge for a live-topology feel.
    flow_phase = time.time() % 1.0
    flow_x, flow_y = [], []
    for e in edges:
        src = e.get("src") or e.get("source")
        dst = e.get("dst") or e.get("target")
        if src not in NODE_POSITIONS or dst not in NODE_POSITIONS:
            continue
        x0, y0 = NODE_POSITIONS[src]
        x1, y1 = NODE_POSITIONS[dst]
        flow_x.append(x0 + (x1 - x0) * flow_phase)
        flow_y.append(y0 + (y1 - y0) * flow_phase)
    fig.add_trace(go.Scatter(
        x=flow_x,
        y=flow_y,
        mode="markers",
        marker=dict(size=8, color="#00E5FF", opacity=0.85),
        hoverinfo="skip",
        name="Flow",
    ))

    # Node sizes scaled to billed_cost
    max_cost  = max((n.get("billed_cost", 0.01) for n in nodes), default=0.04)
    node_sizes = [
        max(30, min(60, int(30 + 30 * n.get("billed_cost", 0.01) / max(max_cost, 0.001))))
        for n in nodes
    ]

    # Node trace
    fig.add_trace(go.Scatter(
        x=[NODE_POSITIONS.get(n["id"], (0.5, 0.5))[0] for n in nodes],
        y=[NODE_POSITIONS.get(n["id"], (0.5, 0.5))[1] for n in nodes],
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=[n["crs"] for n in nodes],
            colorscale=[
                [0.00, "#10B981"], [0.50, "#F59E0B"],
                [0.75, "#8B5CF6"], [1.00, "#F43F5E"],
            ],
            cmin=0, cmax=1,
            line=dict(color="rgba(12,20,40,0.8)", width=2),
            colorbar=dict(
                title=dict(text="CRS", font=dict(color="#94A3B8")),
                tickfont=dict(color="#6B7A99"),
                bgcolor="rgba(12,20,40,0.6)",
                thickness=12,
            ),
        ),
        text=[n["label"] for n in nodes],
        textposition="bottom center",
        textfont=dict(color="#E8EAF6", size=9, family="DM Sans"),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "CRS: %{marker.color:.3f}<br>"
            "Cost: $%{customdata[0]:.4f}<br>"
            "Decision: %{customdata[1]}<extra></extra>"
        ),
        customdata=[[n.get("billed_cost",0), n.get("decision","ALLOW")] for n in nodes],
        name="Pipeline Stages",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title_text="Pipeline DAG — Live CRS Scores",
        showlegend=False,
        height=420,
        xaxis=dict(visible=False, range=[-0.05, 1.05]),
        yaxis=dict(visible=False, range=[-0.05, 1.05]),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Detail panel: show ai_rec for highest-CRS node
    if nodes:
        worst = max(nodes, key=lambda n: n.get("crs", 0))
        ai_r  = worst.get("ai_rec", "")
        if ai_r:
            col_c = crs_color(worst["crs"])
            st.markdown(f"""
            <div style="background:rgba(12,20,40,.8);border:1px solid {col_c}33;
                        border-left:4px solid {col_c};border-radius:12px;
                        padding:14px 18px;margin-top:8px;">
              <div style="color:{col_c};font-weight:700;font-size:.88rem;margin-bottom:6px;">
                🔴 Highest Risk: {worst["id"]} — CRS {worst["crs"]:.3f}
              </div>
              <div style="color:#E8EAF6;font-size:.85rem;line-height:1.7;">{ai_r}</div>
            </div>
            """, unsafe_allow_html=True)


# ── FEATURE-3: Slack-style alert card ─────────────────────────────────────────
def render_alert_card(alert: dict) -> str:
    dec   = alert.get("decision") or alert.get("pade_decision", "ALLOW")
    color = DECISION_COLORS.get(dec, "#6366F1")
    stage = alert.get("stage_name", "—")
    run   = str(alert.get("run_id", ""))[:12]
    crs   = float(alert.get("crs_score", 0))
    cost  = float(alert.get("cost") or alert.get("billed_cost", 0))
    ts    = alert.get("created_at", "")
    return f"""
    <div style="background:rgba(12,20,40,0.8);border:1px solid {color}33;
                border-left:3px solid {color};border-radius:12px;
                padding:14px 18px;margin-bottom:8px;
                animation:slideIn 0.3s ease;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="color:#fff;font-weight:600;font-size:.9rem;">
                {stage} — {run}…
            </span>
            <span style="color:{color};font-size:.8rem;font-weight:700;">{dec}</span>
        </div>
        <div style="color:#6B7A99;font-size:.78rem;margin-top:4px;">
            CRS: <span style="color:{color};font-family:'JetBrains Mono',monospace;">{crs:.3f}</span>
            &nbsp;·&nbsp; ${cost:.4f}
            &nbsp;·&nbsp; {str(ts)[:16]}
        </div>
    </div>
    """


# ── Main page ─────────────────────────────────────────────────────────────────
def show():
    st.markdown(
        cinematic_header(
            "Live Pipeline Visualizer",
            "Real-time CRS scores, AI recommendations, DAG topology, and one-click remediation exports.",
            icon="PIPELINE",
            status="Telemetry Stream Active",
        ),
        unsafe_allow_html=True,
    )

    # ── Tabs: Alerts | DAG Explorer | Live Feed ───────────────────────────────
    tab_alerts, tab_dag, tab_feed = st.tabs(
        ["📋 Alert Feed", "🔗 Pipeline DAG Explorer", "📡 Live Feed"]
    )

    # ═══════════════════════════════════════════════════
    # TAB 1: Alert Feed + Simulate
    # ═══════════════════════════════════════════════════
    with tab_alerts:
        # Simulate panel
        with st.expander("▶ Simulate Pipeline Run", expanded=True):
            st.markdown('<div style="color:#6B7A99;font-size:.88rem;margin-bottom:12px;">Inject a synthetic run — triggers full PADE → PEG → email pipeline without the Go agent.</div>', unsafe_allow_html=True)
            sim_c1, sim_c2 = st.columns([2, 1])
            with sim_c1:
                anomaly_level = st.slider("Anomaly Level (0=normal, 1=severe)", 0.0, 1.0, 0.5, 0.05, key="anom_sl")
            with sim_c2:
                stage_sel = st.selectbox("Target Stage", STAGE_ORDER, index=3, key="stage_sel")

            if st.button("▶ Simulate Run", use_container_width=True, key="sim_btn"):
                with st.spinner(f"Running PADE inference on '{stage_sel}'…"):
                    result = simulate_pipeline(anomaly_level, stage_sel)
                if result and result.get("detail"):
                    # HF-16: api_client now always returns JSON; a "detail" key
                    # means the backend returned an error — surface it to the user
                    st.error(f"❌ Simulation error: {result['detail']}")
                elif result:
                    crs     = result.get("crs", 0.5)
                    dec     = result.get("decision", "ALLOW")
                    ai_rec  = result.get("ai_recommendation", "")
                    savings = result.get("projected_savings")
                    action  = result.get("action_taken")
                    run_id  = result.get("run_id", "")
                    col_clr = DECISION_COLORS.get(dec, "#6366F1")
                    st.markdown(f"""
                    <div style="background:rgba(12,20,40,.8);border:1px solid {col_clr}33;
                                border-left:4px solid {col_clr};border-radius:14px;
                                padding:20px 24px;margin-top:14px;">
                      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">
                        <div>
                          <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.2rem;color:{col_clr};">{dec}</div>
                          <div style="color:#6B7A99;font-size:.78rem;margin-top:2px;">Run {run_id[:12]}…</div>
                        </div>
                        <div style="text-align:right;">
                          <div style="font-family:'JetBrains Mono',monospace;font-size:2rem;font-weight:700;color:{col_clr};">{crs:.3f}</div>
                          <div style="color:#6B7A99;font-size:.7rem;">CRS SCORE</div>
                        </div>
                      </div>
                      <div style="background:rgba(255,255,255,.04);border-radius:8px;height:6px;margin-bottom:14px;overflow:hidden;">
                        <div style="width:{int(crs*100)}%;height:100%;background:{col_clr};border-radius:8px;"></div>
                      </div>
                      {"<div style='color:#34D399;font-size:.85rem;margin-bottom:10px;'>⚙️ " + str(action) + (" — ↓" + str(savings) + "% savings" if savings else "") + "</div>" if action else ""}
                      {"<div style='background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.2);border-left:3px solid #6366F1;border-radius:10px;padding:14px 16px;'><div style='color:#A5B4FC;font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;'>🤖 AI Recommendation</div><div style='color:#E8EAF6;font-size:.88rem;line-height:1.7;'>" + ai_rec + "</div></div>" if ai_rec else ""}
                    </div>
                    """, unsafe_allow_html=True)
                    if dec == "BLOCK":
                        st.toast(f"🚫 BLOCK on '{stage_sel}'! Email alert queued.", icon="🚨")
                    elif dec == "AUTO_OPTIMISE":
                        st.toast(f"⚙️ Auto-optimised '{stage_sel}'.", icon="⚙️")
                else:
                    st.error(
                        f"Simulation failed — check that the backend is reachable at {get_api_http_base_url()}."
                    )

        # Alert table
        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
        alerts     = get_recent_alerts(limit=50)
        is_demo    = not bool(alerts)
        if is_demo:
            alerts = _demo_alerts()
            st.caption("ℹ️ Showing demo alerts — simulate a run to see real data.")

        if not alerts:
            st.markdown("""
            <div style="background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.2);
                        border-radius:16px;padding:40px;text-align:center;">
                <div style="font-size:3rem;margin-bottom:12px;">✅</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#fff;">
                    All Clear — No Active Alerts
                </div>
            </div>
            """, unsafe_allow_html=True)
            return

        df = pd.DataFrame(alerts)
        for col in ["crs_score","billed_cost"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"])

        # KPI cards
        active_count = df[df.get("pade_decision", pd.Series(dtype=str)).isin(["WARN","AUTO_OPTIMISE","BLOCK"])].shape[0] if "pade_decision" in df else 0
        max_crs      = float(df["crs_score"].max()) if "crs_score" in df else 0.0
        blocked_cost = df[df.get("pade_decision","").eq("BLOCK")]["billed_cost"].sum() if "pade_decision" in df else 0.0
        auto_count   = df[df.get("pade_decision","").eq("AUTO_OPTIMISE")].shape[0] if "pade_decision" in df else 0

        c1, c2, c3, c4 = st.columns(4)
        for col, label, value, color, sub in [
            (c1, "🚨 Anomalies",      str(active_count), "#F43F5E","WARN+OPT+BLOCK"),
            (c2, "🎯 Peak CRS",       f"{max_crs:.3f}",  crs_color(max_crs),"Highest risk"),
            (c3, "💸 Cost Blocked",   f"${blocked_cost:.4f}","#8B5CF6","Prevented"),
            (c4, "⚙️  Auto-Optimised", str(auto_count),  "#6366F1","Fixes applied"),
        ]:
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                  <div class="kpi-glow" style="background:{color};"></div>
                  <div class="kpi-label">{label}</div>
                  <div class="kpi-value" style="color:{color};">{value}</div>
                  <div class="kpi-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

        # Gauge + timeline
        g1, g2 = st.columns([1, 2])
        with g1:
            st.plotly_chart(_gauge(max_crs), use_container_width=True, config={"displayModeBar": False})
        with g2:
            if "created_at" in df.columns and "crs_score" in df.columns:
                df_s = df.sort_values("created_at")
                fig_tl = go.Figure()
                fig_tl.add_trace(go.Scatter(
                    x=df_s["created_at"], y=df_s["crs_score"],
                    mode="lines+markers",
                    line=dict(color="#6366F1", width=2),
                    marker=dict(size=7, color=[crs_color(v) for v in df_s["crs_score"]]),
                    fill="tozeroy", fillcolor="rgba(99,102,241,0.06)",
                    hovertemplate="<b>%{x|%H:%M:%S}</b><br>CRS:%{y:.3f}<extra></extra>",
                ))
                for thresh, color, label in [(0.5,"#F59E0B","WARN"),(0.75,"#8B5CF6","OPT"),(0.9,"#F43F5E","BLOCK")]:
                    fig_tl.add_hline(y=thresh, line_dash="dash", line_color=color, line_width=1,
                                     annotation_text=label, annotation_font_color=color, annotation_position="right")
                fig_tl.update_layout(**PLOTLY_LAYOUT, height=280, title_text="CRS Timeline", showlegend=False,
                                     yaxis=dict(gridcolor="rgba(99,102,241,0.1)", tickfont=dict(color="#6B7A99"), range=[0,1.05]))
                st.plotly_chart(fig_tl, use_container_width=True, config={"displayModeBar": False})

        # Alert rows with expandable detail + FEATURE-5 YAML download
        st.markdown('<div class="section-label">📋 Alert Details</div>', unsafe_allow_html=True)
        for _, row in df.iterrows():
            dec    = row.get("pade_decision","ALLOW")
            crs_v  = float(row.get("crs_score", 0))
            stage  = row.get("stage_name","—")
            cost   = float(row.get("billed_cost", 0))
            ai_rec = row.get("ai_recommendation","")
            run_id = str(row.get("run_id",""))
            col_c  = DECISION_COLORS.get(dec,"#6366F1")

            with st.expander(f"{dec} · {stage} · CRS {crs_v:.3f}", expanded=False):
                exp_c1, exp_c2 = st.columns([1, 2])
                with exp_c1:
                    st.plotly_chart(_gauge(crs_v, stage), use_container_width=True, config={"displayModeBar":False})
                with exp_c2:
                    if ai_rec:
                        st.markdown(f"""
                        <div style="background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.2);
                                    border-left:3px solid #6366F1;border-radius:12px;
                                    padding:16px 20px;margin-top:8px;">
                          <div style="color:#A5B4FC;font-size:.72rem;font-weight:700;
                                      letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px;">
                            🤖 AI Recommendation
                          </div>
                          <div style="color:#E8EAF6;font-size:.88rem;line-height:1.7;">{ai_rec}</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.info("No AI recommendation — ensure OPENAI_API_KEY is set.")

                # FEATURE-5: YAML download button for BLOCK decisions (VG-7)
                if dec == "BLOCK" and not run_id.startswith("demo-"):
                    yaml_bytes = remediate(run_id)
                    if yaml_bytes:
                        st.download_button(
                            label="⬇ Download Fix YAML",
                            data=yaml_bytes,
                            file_name=f"costguard-fix-{run_id[:8]}.yml",
                            mime="application/x-yaml",
                            key=f"yaml_{run_id[:12]}",
                        )
                    else:
                        st.caption("YAML patch available after first real BLOCK decision.")
                elif dec == "BLOCK" and run_id.startswith("demo-"):
                    # Demo placeholder
                    demo_yaml = f"# Demo YAML patch for {stage}\n# Run a real simulation to get an actual patch."
                    st.download_button(
                        label="⬇ Download Demo Fix YAML",
                        data=demo_yaml.encode(),
                        file_name=f"costguard-demo-fix.yml",
                        mime="application/x-yaml",
                        key=f"yaml_demo_{stage}",
                    )

    # ═══════════════════════════════════════════════════
    # TAB 2: FEATURE-2 DAG Explorer
    # ═══════════════════════════════════════════════════
    with tab_dag:
        st.markdown('<div class="section-label">🔗 GATv2 Pipeline DAG Explorer</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#6B7A99;font-size:.88rem;margin-bottom:16px;">Live CRS scores per stage from the most recent pipeline run. Node size = billed cost. Node colour = CRS risk level.</div>', unsafe_allow_html=True)

        # Run selector
        col_run, col_dag_ref = st.columns([3, 1])
        with col_run:
            from utils.api_client import get_run_ids
            run_ids = get_run_ids(limit=20)
            if run_ids:
                selected_run = st.selectbox("Select Pipeline Run", ["Latest"] + run_ids, key="dag_run")
                dag_run_id = None if selected_run == "Latest" else selected_run
            else:
                dag_run_id = None
                st.caption("No runs found — showing demo DAG.")
        with col_dag_ref:
            if st.button("↻ Refresh DAG", use_container_width=True, key="dag_refresh"):
                st.rerun()

        render_dag_explorer(dag_run_id)

        # Stage legend
        st.markdown('<div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:8px;">' +
            "".join(f'<span style="background:rgba(12,20,40,.6);border:1px solid rgba(99,102,241,.2);'
                    f'border-radius:8px;padding:4px 12px;color:#94A3B8;font-size:.78rem;">'
                    f'{s.replace("_"," ").title()}</span>' for s in STAGE_ORDER) +
            "</div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════
    # TAB 3: FEATURE-3 Live SSE Feed
    # ═══════════════════════════════════════════════════
    with tab_feed:
        st.markdown('<div class="section-label">📡 Live Alert Feed</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#6B7A99;font-size:.88rem;margin-bottom:16px;">Real-time alerts pushed via PostgreSQL LISTEN/NOTIFY. Refreshes every 10 seconds.</div>', unsafe_allow_html=True)

        # Auto-refresh via session state counter
        if "feed_refresh" not in st.session_state:
            st.session_state.feed_refresh = 0

        col_feed_r, col_feed_f = st.columns([3, 1])
        with col_feed_f:
            if st.button("↻ Refresh Feed", use_container_width=True, key="feed_btn"):
                st.session_state.feed_refresh += 1
                st.rerun()

        live_alerts = get_recent_alerts(limit=20) or _demo_alerts()
        is_demo_feed = not bool(get_recent_alerts(limit=1))
        if is_demo_feed:
            st.caption("ℹ️ Demo feed — simulate runs to see live alerts.")

        feed_html = "".join(render_alert_card({
            "decision":   a.get("pade_decision", "ALLOW"),
            "stage_name": a.get("stage_name", "—"),
            "run_id":     a.get("run_id", ""),
            "crs_score":  a.get("crs_score", 0),
            "cost":       a.get("billed_cost", 0),
            "created_at": str(a.get("created_at",""))[:16],
        }) for a in (live_alerts[:15]))

        st.markdown(f"""
        <div style="background:rgba(12,20,40,.4);border:1px solid rgba(99,102,241,.15);
                    border-radius:16px;padding:16px;max-height:520px;overflow-y:auto;">
            {feed_html if feed_html else '<div style="color:#6B7A99;text-align:center;padding:40px;">No active alerts.</div>'}
        </div>
        """, unsafe_allow_html=True)


show()

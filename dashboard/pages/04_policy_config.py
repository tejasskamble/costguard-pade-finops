"""Policy configuration dashboard for CostGuard governance."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.api_client import get_policy, update_policy, upload_checkpoint, is_authenticated

PLOTLY_LAYOUT = dict(
    plot_bgcolor="#0C1428",
    paper_bgcolor="#0C1428",
    font=dict(family="DM Sans, sans-serif", color="#94A3B8", size=12),
    margin=dict(l=8, r=8, t=8, b=40),
)
STAGES = [
    "build",
    "integration_test",
    "security_scan",
    "docker_build",
    "deploy_staging",
    "deploy_prod",
]
BRANCH_OPTIONS = ["main", "release", "production", "develop", "feature", "hotfix"]


CATALOGUE = [
    {"Action": "Enable build caching", "Savings": "30%", "Applies To": "build"},
    {"Action": "Scope integration tests", "Savings": "45%", "Applies To": "integration_test"},
    {"Action": "Trim security scan scope", "Savings": "35%", "Applies To": "security_scan"},
    {"Action": "Reuse Docker layers", "Savings": "25%", "Applies To": "docker_build"},
    {"Action": "Move staging to cheaper runners", "Savings": "20%", "Applies To": "deploy_staging"},
    {"Action": "Protect production deployment", "Savings": "Risk reduction", "Applies To": "deploy_prod"},
]


def _bundle_from_policy(policy: dict) -> dict:
    bundle = policy.get("policy_bundle") or {}
    thresholds = bundle.get("thresholds") or {}
    rules = bundle.get("rules") or {}
    return {
        "version": bundle.get("version", "v17.0"),
        "thresholds": {
            "warn_threshold": float(thresholds.get("warn_threshold", policy.get("warn_threshold", 0.50))),
            "auto_optimise_threshold": float(
                thresholds.get("auto_optimise_threshold", policy.get("auto_optimise_threshold", 0.75))
            ),
            "block_threshold": float(thresholds.get("block_threshold", policy.get("block_threshold", 0.90))),
        },
        "rules": {
            "protected_branches": list(rules.get("protected_branches", ["main", "release", "production"])),
            "sensitive_stages": list(rules.get("sensitive_stages", ["security_scan", "deploy_staging", "deploy_prod"])),
            "block_pr_prod_deploys": bool(rules.get("block_pr_prod_deploys", True)),
            "require_core_team_for_sensitive_stages": bool(
                rules.get("require_core_team_for_sensitive_stages", True)
            ),
            "stage_cost_ceiling_usd": dict(
                rules.get(
                    "stage_cost_ceiling_usd",
                    {
                        "build": 0.05,
                        "integration_test": 0.08,
                        "security_scan": 0.04,
                        "docker_build": 0.09,
                        "deploy_staging": 0.06,
                        "deploy_prod": 0.08,
                    },
                )
            ),
        },
    }


def show() -> None:
    if not is_authenticated():
        st.error("Please login to access this page.")
        st.stop()

    st.markdown(
        """
        <div class="page-header">
            <h1>Policy Configuration</h1>
            <p>Manage CostGuard OPA governance, CRS thresholds, and backend checkpoint uploads.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    policy = get_policy()
    if not policy:
        st.error("Could not load policy from the backend.")
        return

    bundle = _bundle_from_policy(policy)
    warn_val = float(policy["warn_threshold"])
    auto_val = float(policy["auto_optimise_threshold"])
    block_val = float(policy["block_threshold"])
    rules = bundle["rules"]

    c1, c2, c3 = st.columns(3)
    c1.metric("WARN", f"{warn_val:.2f}")
    c2.metric("AUTO_OPTIMISE", f"{auto_val:.2f}")
    c3.metric("BLOCK", f"{block_val:.2f}")

    fig = go.Figure()
    for x0, x1, color, label in [
        (0.0, warn_val, "#10B981", "ALLOW"),
        (warn_val, auto_val, "#F59E0B", "WARN"),
        (auto_val, block_val, "#8B5CF6", "AUTO_OPTIMISE"),
        (block_val, 1.0, "#F43F5E", "BLOCK"),
    ]:
        width = max(0.0, x1 - x0)
        if width <= 0:
            continue
        fig.add_trace(
            go.Bar(
                x=[width],
                y=["CRS"],
                orientation="h",
                marker_color=color,
                base=[x0],
                name=label,
                text=[f"{label} {x0:.2f}-{x1:.2f}"],
                textposition="inside",
            )
        )
    fig.update_layout(**PLOTLY_LAYOUT, barmode="stack", height=110, yaxis=dict(showticklabels=False), xaxis=dict(range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("### Governance Controls")
    with st.form("policy_form"):
        warn = st.slider("WARN threshold", 0.0, 1.0, warn_val, 0.01)
        auto = st.slider("AUTO_OPTIMISE threshold", 0.0, 1.0, auto_val, 0.01)
        block = st.slider("BLOCK threshold", 0.0, 1.0, block_val, 0.01)

        protected_branches = st.multiselect(
            "Protected branches",
            BRANCH_OPTIONS,
            default=[b for b in rules.get("protected_branches", []) if b in BRANCH_OPTIONS],
        )
        sensitive_stages = st.multiselect(
            "Sensitive stages",
            STAGES,
            default=[s for s in rules.get("sensitive_stages", []) if s in STAGES],
        )
        block_pr_prod = st.checkbox(
            "Block production deploys from PR runs",
            value=bool(rules.get("block_pr_prod_deploys", True)),
        )
        require_core = st.checkbox(
            "Require core-team members for sensitive stages on protected branches",
            value=bool(rules.get("require_core_team_for_sensitive_stages", True)),
        )

        st.markdown("#### Stage Cost Ceilings (USD)")
        ceiling_cols = st.columns(3)
        stage_cost_ceiling_usd = {}
        for idx, stage in enumerate(STAGES):
            with ceiling_cols[idx % 3]:
                stage_cost_ceiling_usd[stage] = st.number_input(
                    f"{stage}",
                    min_value=0.0,
                    value=float(rules.get("stage_cost_ceiling_usd", {}).get(stage, 0.05)),
                    step=0.01,
                    format="%.2f",
                )

        save = st.form_submit_button("Save Policy", use_container_width=True)
        reset = st.form_submit_button("Reset Defaults", use_container_width=True)

        if save:
            if warn >= auto or auto >= block:
                st.error("Thresholds must be ordered WARN < AUTO_OPTIMISE < BLOCK.")
            else:
                next_bundle = {
                    "version": "v17.0",
                    "thresholds": {
                        "warn_threshold": warn,
                        "auto_optimise_threshold": auto,
                        "block_threshold": block,
                    },
                    "rules": {
                        "protected_branches": protected_branches,
                        "sensitive_stages": sensitive_stages,
                        "block_pr_prod_deploys": block_pr_prod,
                        "require_core_team_for_sensitive_stages": require_core,
                        "stage_cost_ceiling_usd": stage_cost_ceiling_usd,
                    },
                }
                if update_policy(warn, auto, block, next_bundle):
                    st.success("Policy updated.")
                    st.rerun()
                else:
                    st.error("Policy update failed.")

        if reset:
            default_bundle = _bundle_from_policy({})
            if update_policy(0.50, 0.75, 0.90, default_bundle):
                st.success("Policy reset to defaults.")
                st.rerun()
            else:
                st.error("Policy reset failed.")

    st.markdown("### Active Governance Bundle")
    st.json(bundle)

    st.markdown("### Auto-Optimisation Catalogue")
    st.dataframe(pd.DataFrame(CATALOGUE), use_container_width=True, hide_index=True)

    st.markdown("### Backend Checkpoint Upload")
    uploaded = st.file_uploader("Upload backend inference checkpoint (.pt)", type=["pt"])
    if uploaded is not None and st.button("Load Checkpoint"):
        result = upload_checkpoint(uploaded.read())
        if result and result.get("status") == "loaded":
            st.success(result.get("message", "Model loaded successfully."))
        elif result and result.get("detail"):
            st.error(str(result["detail"]))
        else:
            st.error("Checkpoint upload failed.")


show()

"""
dashboard/pages/11_notifications.py - CostGuard v17.0

Notification preferences and delivery checks for CostGuard.
"""
import streamlit as st
from components.cinematic_ui import apply_cinematic_ui, cinematic_header
from utils.api_client import (
    is_authenticated, get_notification_prefs, update_notification_prefs,
    send_test_notification, log_page_visit,
)

if not is_authenticated():
    st.error("🔒 Please login to access this page.")
    st.stop()

log_page_visit("11_notifications")
apply_cinematic_ui("11_notifications")

st.markdown(
    cinematic_header(
        "Notification Grid",
        "Configure alert delivery for anomaly events, budget thresholds, and scheduled digests.",
        icon="ALERTS",
        status="Dispatch Network Ready",
    ),
    unsafe_allow_html=True,
)

# ── Load current preferences ──────────────────────────────────────────────────
with st.spinner("Loading preferences…"):
    prefs = get_notification_prefs() or {
        "email_enabled": True, "slack_enabled": False, "slack_webhook": None,
        "budget_threshold": None, "anomaly_sensitivity": "medium", "digest_frequency": "daily",
    }

tab1, tab2, tab3 = st.tabs(["⚙️ Preferences", "🧪 Test Notification", "📜 Notification Log"])

# ── Tab 1: Preferences ────────────────────────────────────────────────────────
with tab1:
    with st.form("notif_prefs_form"):
        # Email preferences
        st.markdown('<div class="pref-card"><h4>📧 Email Notifications</h4>', unsafe_allow_html=True)
        email_enabled = st.checkbox("Enable email notifications",
                                     value=prefs.get("email_enabled", True))
        st.markdown("Emails are sent to your registered account email address.", unsafe_allow_html=False)
        st.markdown('</div>', unsafe_allow_html=True)

        # Slack preferences
        st.markdown('<div class="pref-card"><h4>💬 Slack Notifications</h4>', unsafe_allow_html=True)
        slack_enabled = st.checkbox("Enable Slack notifications",
                                     value=prefs.get("slack_enabled", False))
        slack_webhook = st.text_input(
            "Slack Webhook URL (optional)",
            value=prefs.get("slack_webhook") or "",
            placeholder="https://hooks.slack.com/services/...",
            type="password" if prefs.get("slack_webhook") else "default",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Budget threshold
        st.markdown('<div class="pref-card"><h4>💰 Budget Threshold Alert</h4>', unsafe_allow_html=True)
        current_threshold = prefs.get("budget_threshold")
        budget_enabled = st.checkbox("Enable budget alerts",
                                      value=current_threshold is not None)
        budget_threshold = None
        if budget_enabled:
            budget_threshold = st.number_input(
                "Alert me when monthly pipeline cost exceeds ($)",
                min_value=0.0, step=50.0,
                value=float(current_threshold) if current_threshold else 500.0,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # Anomaly sensitivity
        st.markdown('<div class="pref-card"><h4>🎯 Anomaly Detection Sensitivity</h4>', unsafe_allow_html=True)
        sensitivity_opts = ["low", "medium", "high", "paranoid"]
        current_sens = prefs.get("anomaly_sensitivity", "medium")
        sensitivity_idx = sensitivity_opts.index(current_sens) if current_sens in sensitivity_opts else 1
        anomaly_sensitivity = st.select_slider(
            "Sensitivity Level",
            options=sensitivity_opts,
            value=sensitivity_opts[sensitivity_idx],
            format_func=lambda x: {
                "low": "🟢 Low — Only critical anomalies",
                "medium": "🟡 Medium — Balanced (recommended)",
                "high": "🟠 High — Catch more anomalies",
                "paranoid": "🔴 Paranoid — Alert on everything",
            }[x],
        )
        st.markdown("""
        <div style="background:#111D35;border-radius:8px;padding:12px;font-size:.83rem;color:#B0BDD0;margin-top:8px;">
          💡 Higher sensitivity catches more anomalies but may increase false positives.
          CRS threshold adjustments: Low=0.7, Medium=0.5, High=0.3, Paranoid=0.1
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Digest schedule
        st.markdown('<div class="pref-card"><h4>📊 Report Digest Schedule</h4>', unsafe_allow_html=True)
        digest_opts = ["daily", "weekly", "never"]
        current_digest = prefs.get("digest_frequency", "daily")
        digest_idx = digest_opts.index(current_digest) if current_digest in digest_opts else 0
        digest_frequency = st.radio(
            "Receive digest report",
            digest_opts,
            index=digest_idx,
            format_func=lambda x: {
                "daily": "📅 Daily (6 AM IST)",
                "weekly": "📆 Weekly (Monday 6 AM IST)",
                "never": "🔕 Never",
            }[x],
            horizontal=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        saved = st.form_submit_button("💾 Save Preferences", use_container_width=True)

    if saved:
        new_prefs = {
            "email_enabled": email_enabled,
            "slack_enabled": slack_enabled,
            "slack_webhook": slack_webhook.strip() or None,
            "budget_threshold": float(budget_threshold) if budget_enabled and budget_threshold else None,
            "anomaly_sensitivity": anomaly_sensitivity,
            "digest_frequency": digest_frequency,
        }
        with st.spinner("Saving preferences…"):
            ok = update_notification_prefs(new_prefs)
        if ok:
            st.success("✅ Preferences saved successfully!")
        else:
            st.error("❌ Failed to save preferences. Please try again.")

# ── Tab 2: Test Notification ──────────────────────────────────────────────────
with tab2:
    st.markdown("#### 🧪 Send a Test Notification")
    st.markdown("""
    <div style="background:#111D35;border-radius:12px;padding:16px 20px;margin-bottom:20px;
                font-size:.88rem;color:#B0BDD0;">
      Click the button below to send a test notification to all enabled channels
      (email and/or Slack). Use this to verify your notification settings are working.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 3])
    with col1:
        if st.button("📤 Send Test Notification", use_container_width=True):
            with st.spinner("Dispatching test notification…"):
                result = send_test_notification()
            if result:
                channels = result.get("channels", {})
                for channel, status in channels.items():
                    if "sent" in str(status).lower():
                        st.success(f"✅ {channel.title()}: {status}")
                    else:
                        st.warning(f"⚠️ {channel.title()}: {status}")
                if not channels:
                    st.info("No notification channels configured. Enable email or Slack in Preferences.")
            else:
                st.error("❌ Failed to send test notification.")

    with col2:
        current_channels = []
        if prefs.get("email_enabled"):
            current_channels.append("📧 Email")
        if prefs.get("slack_enabled") and prefs.get("slack_webhook"):
            current_channels.append("💬 Slack")
        st.markdown(f"""
        <div style="background:#0D1B2E;border-radius:10px;padding:16px;font-size:.85rem;">
          <div style="color:#6B7A99;font-size:.75rem;text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px;">ACTIVE CHANNELS</div>
          {"<br>".join(f'<span style="color:#00D4AA;">{c}</span>' for c in current_channels) if current_channels else '<span style="color:#6B7A99;">None configured</span>'}
        </div>""", unsafe_allow_html=True)

# ── Tab 3: Notification Log ───────────────────────────────────────────────────
with tab3:
    st.markdown("#### 📜 Recent Notifications")
    st.info("Notification history will appear here once the system has sent alerts. Enable anomaly detection and run pipeline ingestion to generate alerts.")
    st.markdown("""
    <div style="background:#111D35;border-radius:12px;padding:16px 20px;font-size:.85rem;color:#6B7A99;">
      🔍 Notification log stores the last 30 days of sent alerts, digests, and test notifications.
      Triggered by: anomaly threshold breaches, budget overruns, and scheduled digests.
    </div>""", unsafe_allow_html=True)

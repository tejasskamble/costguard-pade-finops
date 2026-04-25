"""
dashboard/app.py — CostGuard v17.0 Enterprise Edition
Premium dark theme with brand CSS, forgot-password OTP wizard,
branded sidebar navigation with Gravatar avatar, notification bell.
"""
import hashlib
import importlib.util
import os
import sys
import requests as _requests
import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="CostGuard v17.0",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.api_client import (
    login, register, get_me, get_recent_alerts,
    set_token, get_token, is_authenticated,
    forgot_password, verify_otp, reset_password,
    get_gravatar_url, get_my_profile, get_api_http_base_url,
)

API_BASE = get_api_http_base_url()

# ── Google Fonts ──────────────────────────────────────────────────────────────
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800'
    '&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400'
    '&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

# ── Master CSS (Section 7 premium theme) ──────────────────────────────────────
MASTER_CSS = """
<style>
:root {
  --brand:#FF6B35; --brand-hover:#E85A28; --brand-dark:#C44D1E;
  --navy:#2C3E7A; --gold:#FFD700; --luxury:#C084FC;
  --bg:#060B18; --surface:#0D1B2E; --elevated:#111D35;
  --glass:rgba(13,27,46,0.85);
  --border:rgba(255,107,53,0.15); --border-active:rgba(255,107,53,0.5);
  --text:#E8F0FE; --muted:#6B7A99;
  --success:#00D4AA; --warn:#FF9500; --danger:#FF3B5C; --info:#4A9EFF;
}
.stApp { background:var(--bg) !important; color:var(--text) !important; }
.stApp::before {
  content:''; position:fixed; inset:0; z-index:0; pointer-events:none;
  background:
    radial-gradient(ellipse 100% 60% at 15% -5%, rgba(255,107,53,.08) 0%, transparent 55%),
    radial-gradient(ellipse 70% 50% at 85% 95%, rgba(44,62,122,.12) 0%, transparent 50%),
    radial-gradient(ellipse 50% 40% at 50% 50%, rgba(192,132,252,.04) 0%, transparent 65%);
}
.main .block-container { position:relative; z-index:1; padding:1.5rem 2rem 4rem !important; }
#MainMenu, footer, .stDeployButton { display:none !important; }
[data-testid="stSidebar"] {
  background:rgba(6,11,24,0.97) !important;
  border-right:1px solid var(--border) !important;
}
/* ── Page header ── */
.cg-page-header {
  background:linear-gradient(135deg,rgba(255,107,53,.08) 0%,rgba(44,62,122,.06) 100%);
  border:1px solid var(--border); border-radius:16px;
  padding:24px 28px; margin-bottom:24px; position:relative; overflow:hidden;
}
.cg-page-header::before {
  content:''; position:absolute; top:-40px; right:-40px;
  width:180px; height:180px; border-radius:50%;
  background:radial-gradient(circle,rgba(255,107,53,.12) 0%,transparent 70%);
}
.cg-page-header h1 {
  font-family:'Syne',sans-serif; font-size:1.9rem; font-weight:800;
  color:#fff; letter-spacing:-0.02em; margin:0;
}
.cg-page-header p { color:var(--muted); margin:6px 0 0; font-size:.95rem; }
/* ── KPI cards ── */
.kpi-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px; margin:20px 0; }
.kpi-card {
  background:var(--surface); border:1px solid var(--border); border-radius:16px;
  padding:20px 22px; position:relative; overflow:hidden;
  transition:transform .25s cubic-bezier(.34,1.56,.64,1), border-color .2s, box-shadow .2s;
}
.kpi-card:hover {
  transform:translateY(-5px) scale(1.01); border-color:var(--border-active);
  box-shadow:0 20px 40px -10px rgba(255,107,53,.15);
}
.kpi-label { font-size:.68rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); margin-bottom:8px; }
.kpi-value { font-family:'Syne',sans-serif; font-size:2.1rem; font-weight:800; color:#fff; letter-spacing:-0.03em; line-height:1; }
.kpi-delta-pos { font-size:.82rem; color:var(--success); margin-top:6px; }
.kpi-delta-neg { font-size:.82rem; color:var(--danger); margin-top:6px; }
/* ── Buttons ── */
.stButton > button {
  background:linear-gradient(135deg,#FF6B35,#E85A28) !important;
  color:#fff !important; border:none !important; border-radius:10px !important;
  font-weight:600 !important; font-size:.9rem !important; padding:11px 24px !important;
  box-shadow:0 4px 15px rgba(255,107,53,.35) !important; transition:all .2s ease !important;
}
.stButton > button:hover {
  transform:translateY(-2px) !important;
  box-shadow:0 8px 25px rgba(255,107,53,.45) !important;
}
/* ── Forms ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] > div > div {
  background:var(--elevated) !important; border:1px solid var(--border) !important;
  border-radius:10px !important; color:var(--text) !important;
}
/* ── Badges ── */
.badge { display:inline-flex; align-items:center; gap:5px; padding:3px 10px; border-radius:40px;
  font-size:.68rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
.badge-allow    { background:rgba(0,212,170,.1);  color:#00D4AA; border:1px solid rgba(0,212,170,.3); }
.badge-warn     { background:rgba(255,149,0,.1);  color:#FF9500; border:1px solid rgba(255,149,0,.3); }
.badge-block    { background:rgba(255,59,92,.1);  color:#FF3B5C; border:1px solid rgba(255,59,92,.3); }
.badge-optimise { background:rgba(192,132,252,.1);color:#C084FC; border:1px solid rgba(192,132,252,.3); }
.badge-admin    { background:rgba(255,215,0,.1);  color:#FFD700; border:1px solid rgba(255,215,0,.3); }
/* ── Progress bar ── */
.stProgress > div > div > div > div {
  background:linear-gradient(90deg,var(--brand),#C084FC) !important; border-radius:4px !important;
}
/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] { font-weight:600 !important; color:var(--muted) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color:var(--brand) !important; border-bottom-color:var(--brand) !important; }
/* ── Metric ── */
[data-testid="stMetric"] { background:var(--surface); border-radius:12px; padding:16px !important; }
[data-testid="stMetricValue"] { color:#fff !important; font-family:'Syne',sans-serif !important; }
/* ── Expanders ── */
[data-testid="stExpander"] { background:var(--surface) !important; border:1px solid var(--border) !important; border-radius:12px !important; }
/* ── Scrollbar ── */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:rgba(255,107,53,.3); border-radius:3px; }
/* ── Auth card ── */
.auth-card {
  max-width:440px; margin:40px auto; background:var(--surface);
  border:1px solid var(--border); border-radius:20px; padding:40px;
}
.auth-card h1 { font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800;
  background:linear-gradient(135deg,#FF6B35,#C084FC);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; text-align:center; }
/* ── OTP boxes ── */
.otp-row { display:flex; gap:10px; justify-content:center; margin:20px 0; }
.otp-dig {
  width:52px; height:62px; background:var(--elevated);
  border:2px solid var(--border); border-radius:12px;
  font-size:1.8rem; font-weight:800; text-align:center; color:#FF6B35;
  font-family:'JetBrains Mono',monospace;
  transition:border-color .2s; outline:none;
}
.otp-dig:focus { border-color:#FF6B35; box-shadow:0 0 0 3px rgba(255,107,53,.2); }
/* ── Sidebar nav ── */
.nav-group-label {
  font-size:.62rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase;
  color:#FF6B35; margin:16px 0 6px 4px; display:block;
}
.nav-item { padding:8px 12px; font-size:.88rem; color:#8B97B5;
  border-radius:8px; margin-bottom:2px; }
.nav-item-active {
  background:rgba(255,107,53,.12); border:1px solid rgba(255,107,53,.25);
  border-radius:8px; padding:8px 12px; font-size:.88rem; font-weight:600; color:#FF6B35;
  margin-bottom:2px;
}
</style>
"""
st.markdown(MASTER_CSS, unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def page_header(icon: str, title: str, subtitle: str) -> None:
    """Render a branded page header div."""
    st.markdown(f"""
    <div class="cg-page-header">
      <h1>{icon} {title}</h1>
      <p>{subtitle}</p>
    </div>""", unsafe_allow_html=True)


def kpi_card(label: str, value: str, delta: str = "", positive: bool = True, icon: str = "") -> str:
    """Return HTML for a single KPI card."""
    delta_cls = "kpi-delta-pos" if positive else "kpi-delta-neg"
    delta_html = f'<div class="{delta_cls}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{icon} {label}</div>
      <div class="kpi-value">{value}</div>
      {delta_html}
    </div>"""


# ── Sidebar navigation ────────────────────────────────────────────────────────

def render_sidebar():
    """Render branded sidebar with Gravatar, nav groups, and sign-out."""
    with st.sidebar:
        # Brand logo
        st.markdown("""
        <div style="padding:20px 0 16px;text-align:center;
                    border-bottom:1px solid rgba(255,107,53,.15);margin-bottom:16px;">
          <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:900;
                      background:linear-gradient(135deg,#FF6B35,#C084FC);
                      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                      letter-spacing:-0.03em;">🛡️ CostGuard</div>
          <div style="font-size:.65rem;color:#6B7A99;letter-spacing:.15em;
                      text-transform:uppercase;margin-top:4px;">v17.0 Enterprise</div>
        </div>""", unsafe_allow_html=True)

        if is_authenticated():
            profile = st.session_state.get("user_profile", {})
            email = profile.get("email", "user@costguard.io")
            name = profile.get("full_name", "User")
            role = profile.get("role", "viewer")
            gravatar = get_gravatar_url(email, 36)
            role_color = {"admin": "#FFD700", "analyst": "#4A9EFF", "viewer": "#00D4AA"}.get(role, "#8B97B5")
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:10px 12px;
                        background:rgba(255,107,53,.06);border:1px solid rgba(255,107,53,.12);
                        border-radius:10px;margin-bottom:20px;">
              <img src="{gravatar}" style="width:36px;height:36px;border-radius:50%;
                   border:2px solid rgba(255,107,53,.3);">
              <div>
                <div style="font-size:.82rem;font-weight:600;color:#E8F0FE;">{name[:18]}</div>
                <div style="font-size:.62rem;color:{role_color};text-transform:uppercase;
                            font-weight:700;letter-spacing:.06em;">{role}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        nav_groups = {
            "📊 Analytics": [
                ("🏠 Dashboard",        None),              # main app.py IS the home page
                ("💰 Pipeline Costs",   "01_pipeline_costs"),
                ("🚨 Anomaly Alerts",   "02_anomaly_alerts"),
                ("🔮 Forecasting",      "06_forecasting"),
                ("☁️ Cloud Compare",    "09_cloud_compare"),
            ],
            "🤖 Intelligence": [
                ("💬 Cost Query (AI)",  "03_cost_query"),
                ("🧠 ML Training Lab",  "05_ml_training"),
                ("📜 Run History",      "10_run_history"),
                ("📦 Post-Run Artifacts", "13_post_run_artifacts"),
            ],
            "⚙️ Configuration": [
                ("🛡️ Policy Config",   "04_policy_config"),
                ("🔔 Notifications",    "11_notifications"),
            ],
            "👥 Account": [
                ("👤 Team & Users",     "07_team"),
                ("📨 Support",          "08_support"),
                ("❤️ System Health",    "12_system"),
            ],
        }
        for group_label, pages in nav_groups.items():
            st.markdown(f'<span class="nav-group-label">{group_label}</span>',
                        unsafe_allow_html=True)
            for label, page_file in pages:
                # FIX 6: use st.page_link() so clicks actually navigate
                # HIDDEN FIX: Home has no separate file — link to app.py itself
                if page_file is None:
                    st.page_link("app.py", label=label, use_container_width=True)
                else:
                    st.page_link(
                        f"pages/{page_file}.py",
                        label=label,
                        use_container_width=True,
                    )

        st.markdown("---")
        if st.button("🚪 Sign Out", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ── Forgot-password wizard ────────────────────────────────────────────────────

def render_forgot_password():
    """Three-step forgot-password OTP wizard."""
    step = st.session_state.get("fp_step", 1)

    # Animated step progress bar
    steps = ["1 · Enter Email", "2 · Verify OTP", "3 · New Password"]
    st.markdown(f"""
    <div style="display:flex;gap:0;margin:0 0 28px;border-radius:10px;overflow:hidden;">
      {"".join(
        f'<div style="flex:1;padding:10px;text-align:center;font-size:.78rem;font-weight:700;'
        f'background:{"rgba(255,107,53,.9)" if i+1==step else ("rgba(255,107,53,.2)" if i+1<step else "rgba(255,255,255,.04)")};'
        f'color:{"#fff" if i+1<=step else "#6B7A99"};">{s}</div>'
        for i, s in enumerate(steps)
      )}
    </div>""", unsafe_allow_html=True)

    if step == 1:
        st.markdown("#### Enter your registered email address")
        fp_email = st.text_input("Email Address", key="fp_email_input",
                                  placeholder="you@example.com")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("📧 Send OTP", use_container_width=True):
                if not fp_email or "@" not in fp_email:
                    st.error("Please enter a valid email address.")
                else:
                    with st.spinner("Sending OTP…"):
                        result = forgot_password(fp_email)
                    if result and result.get("message") and not result.get("detail"):
                        st.session_state["fp_email"] = fp_email
                        st.session_state["fp_step"] = 2
                        st.success("✅ " + result["message"])
                        st.rerun()
                    elif result and result.get("detail"):
                        st.error(f"❌ {result['detail']}")
                    else:
                        st.error("❌ Failed to send OTP. Please try again.")
        with col2:
            if st.button("← Back to Login", use_container_width=True):
                st.session_state["show_forgot"] = False
                st.rerun()

    elif step == 2:
        fp_email = st.session_state.get("fp_email", "")
        st.markdown(f"#### Enter the 6-digit OTP sent to **{fp_email[:3]}***")

        # 6-box OTP widget with auto-focus JS
        st.markdown("""
        <style>
        .otp-row{display:flex;gap:10px;justify-content:center;margin:16px 0;}
        .otp-dig{width:52px;height:62px;background:#111D35;border:2px solid rgba(255,107,53,.4);
          border-radius:12px;font-size:1.8rem;font-weight:800;text-align:center;color:#FF6B35;
          font-family:'JetBrains Mono',monospace;outline:none;}
        .otp-dig:focus{border-color:#FF6B35;box-shadow:0 0 0 3px rgba(255,107,53,.2);}
        </style>
        <div class="otp-row">
          <input class="otp-dig" maxlength="1" id="o1" oninput="if(this.value.length===1)document.getElementById('o2').focus()">
          <input class="otp-dig" maxlength="1" id="o2" oninput="if(this.value.length===1)document.getElementById('o3').focus()">
          <input class="otp-dig" maxlength="1" id="o3" oninput="if(this.value.length===1)document.getElementById('o4').focus()">
          <input class="otp-dig" maxlength="1" id="o4" oninput="if(this.value.length===1)document.getElementById('o5').focus()">
          <input class="otp-dig" maxlength="1" id="o5" oninput="if(this.value.length===1)document.getElementById('o6').focus()">
          <input class="otp-dig" maxlength="1" id="o6">
        </div>
        """, unsafe_allow_html=True)

        otp_input = st.text_input("Enter OTP (6 digits)", max_chars=6,
                                   label_visibility="collapsed",
                                   placeholder="Enter 6-digit OTP",
                                   key="fp_otp_input")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✅ Verify OTP", use_container_width=True):
                if not otp_input or len(otp_input) != 6 or not otp_input.isdigit():
                    st.error("Please enter a valid 6-digit OTP.")
                else:
                    with st.spinner("Verifying…"):
                        result = verify_otp(fp_email, otp_input)
                    if result and result.get("valid"):
                        st.session_state["fp_reset_token"] = result["reset_token"]
                        st.session_state["fp_step"] = 3
                        st.success("✅ OTP verified! Set your new password.")
                        st.rerun()
                    elif result and result.get("detail"):
                        st.error(f"❌ {result['detail']}")
                    else:
                        st.error("❌ OTP verification failed. Please try again.")
        with col2:
            if st.button("🔄 Resend OTP", use_container_width=True):
                with st.spinner("Resending…"):
                    forgot_password(fp_email)
                st.info("A new OTP has been sent.")

    elif step == 3:
        st.markdown("#### Set your new password")
        new_pw = st.text_input("New Password", type="password", key="fp_new_pw",
                                placeholder="Min 8 chars, 1 uppercase, 1 digit")
        confirm_pw = st.text_input("Confirm Password", type="password", key="fp_confirm_pw")

        # Strength indicator
        if new_pw:
            strength = sum([
                len(new_pw) >= 8, any(c.isupper() for c in new_pw),
                any(c.isdigit() for c in new_pw), len(new_pw) >= 12,
            ])
            colors = ["#FF3B5C", "#FF9500", "#FFD700", "#00D4AA"]
            labels = ["Weak", "Fair", "Good", "Strong"]
            st.markdown(f"""
            <div style="margin:8px 0;">
              <div style="display:flex;gap:4px;margin-bottom:4px;">
                {"".join(f'<div style="flex:1;height:4px;border-radius:2px;background:{colors[strength-1] if i<strength else "#1E293B"};"></div>' for i in range(4))}
              </div>
              <span style="font-size:.75rem;color:{colors[strength-1]};">{labels[strength-1]}</span>
            </div>""", unsafe_allow_html=True)

        if st.button("🔒 Reset Password", use_container_width=True):
            if not new_pw or not confirm_pw:
                st.error("Please fill in both password fields.")
            elif new_pw != confirm_pw:
                st.error("Passwords do not match.")
            elif len(new_pw) < 8:
                st.error("Password must be at least 8 characters.")
            else:
                reset_token = st.session_state.get("fp_reset_token", "")
                with st.spinner("Resetting password…"):
                    result = reset_password(reset_token, new_pw, confirm_pw)
                if result and "successfully" in result.get("message", "").lower():
                    st.success("🎉 Password reset successfully!")
                    st.balloons()
                    # Clear state, redirect to login after 3s
                    for k in ["fp_step", "fp_email", "fp_reset_token", "show_forgot"]:
                        st.session_state.pop(k, None)
                    st.info("Redirecting to login…")
                    st.rerun()
                else:
                    detail = result.get("detail", "Reset failed.") if result else "Request failed."
                    st.error(f"❌ {detail}")


# ── Auth forms ────────────────────────────────────────────────────────────────

def render_auth():
    """Render login / register / forgot-password forms."""
    st.markdown("""
    <div style="text-align:center;padding:40px 0 20px;">
      <div style="font-family:'Syne',sans-serif;font-size:2.8rem;font-weight:900;
                  background:linear-gradient(135deg,#FF6B35 0%,#FF3B5C 50%,#C084FC 100%);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  letter-spacing:-0.04em;">🛡️ CostGuard</div>
      <div style="color:#6B7A99;font-size:.9rem;margin-top:8px;letter-spacing:.05em;">
        v17.0 Enterprise · FinOps Intelligence Platform</div>
    </div>""", unsafe_allow_html=True)

    # Show forgot-password wizard if requested
    if st.session_state.get("show_forgot"):
        if "fp_step" not in st.session_state:
            st.session_state["fp_step"] = 1
        render_forgot_password()
        return

    tab1, tab2 = st.tabs(["🔑 Sign In", "📝 Create Account"])

    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email Address", placeholder="you@example.com")
            password = st.text_input("Password", type="password", placeholder="Your password")
            submitted = st.form_submit_button("Sign In →", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("Please enter both email and password.")
            else:
                with st.spinner("Authenticating…"):
                    try:
                        token = login(email, password)
                    except ValueError as e:
                        st.error(f"❌ {e}")
                        token = None
                if token:
                    set_token(token)
                    st.session_state["authenticated"] = True
                    # Fetch profile for sidebar
                    try:
                        resp = _requests.get(
                            f"{API_BASE}/api/users/me",
                            headers={"Authorization": f"Bearer {token}"}, timeout=5
                        )
                        if resp.status_code == 200:
                            st.session_state["user_profile"] = resp.json()
                        else:
                            me = _requests.get(
                                f"{API_BASE}/api/auth/me",
                                headers={"Authorization": f"Bearer {token}"}, timeout=5
                            ).json()
                            st.session_state["user_profile"] = me
                    except Exception:
                        st.session_state["user_profile"] = {"email": email, "full_name": "User", "role": "viewer"}
                    st.success("✅ Signed in successfully!")
                    st.rerun()

        # Forgot password link
        st.markdown(
            '<div style="text-align:right;margin-top:8px;">'
            '<a href="#" style="color:#FF6B35;font-size:.83rem;text-decoration:none;" '
            'onclick="void(0)">Forgot Password?</a></div>',
            unsafe_allow_html=True
        )
        if st.button("🔐 Forgot Password?", use_container_width=False):
            st.session_state["show_forgot"] = True
            st.session_state["fp_step"] = 1
            st.rerun()

    with tab2:
        with st.form("register_form"):
            rname = st.text_input("Full Name", placeholder="Your name")
            remail = st.text_input("Email Address", placeholder="you@example.com")
            rpass = st.text_input("Password", type="password",
                                   placeholder="Min 8 chars, 1 uppercase, 1 digit")
            rpass2 = st.text_input("Confirm Password", type="password")
            reg_submitted = st.form_submit_button("Create Account →", use_container_width=True)

        if reg_submitted:
            errors = []
            if not rname.strip(): errors.append("Name is required")
            if not remail or "@" not in remail: errors.append("Valid email required")
            if len(rpass) < 8: errors.append("Password min 8 characters")
            if rpass != rpass2: errors.append("Passwords do not match")
            if errors:
                for e in errors: st.error(f"❌ {e}")
            else:
                with st.spinner("Creating account…"):
                    from utils.api_client import register as do_register
                    result = do_register(remail, rpass, rname)
                if result and result.get("access_token"):
                    set_token(result["access_token"])
                    st.session_state["authenticated"] = True
                    st.session_state["user_profile"] = {
                        "email": remail, "full_name": rname, "role": "viewer"
                    }
                    st.success(f"✅ Account created! Welcome, {rname}!")
                    st.balloons()
                    st.rerun()
                else:
                    detail = result.get("detail", "Registration failed.") if result else "Registration failed."
                    st.error(f"❌ {detail}")


# ── Dashboard home page ───────────────────────────────────────────────────────

def render_dashboard():
    """Main dashboard: KPI cards + live alert feed."""
    render_sidebar()
    page_header("🏠", "Dashboard", "Real-time pipeline cost intelligence — CostGuard v17.0")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pipeline Cost", "$0.00", delta="No data yet")
    with col2:
        st.metric("Active Alerts", "0", delta="All clear")
    with col3:
        st.metric("PADE CRS Score", "—", delta="Model ready")
    with col4:
        st.metric("Budget Used", "0%", delta="On track")

    st.markdown("---")

    # Quick-action cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="kpi-card">
          <div class="kpi-label">💰 Pipeline Costs</div>
          <div class="kpi-value" style="font-size:1.1rem;">View breakdown by stage,<br>provider, and branch.</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="kpi-card">
          <div class="kpi-label">🧠 ML Training Lab</div>
          <div class="kpi-value" style="font-size:1.1rem;">Train LSTM + GAT<br>anomaly detection models.</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="kpi-card">
          <div class="kpi-label">🔮 Cost Forecast</div>
          <div class="kpi-value" style="font-size:1.1rem;">ARIMA 90-day forecast<br>with budget burn simulation.</div>
        </div>""", unsafe_allow_html=True)

    # Live alerts feed
    st.markdown("### 🚨 Recent Alerts")
    with st.spinner("Loading alerts…"):
        alerts = get_recent_alerts(limit=10)
    if alerts:
        import pandas as pd
        df = pd.DataFrame(alerts)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No alerts yet. Ingest pipeline runs to start monitoring.")


# ── Main entrypoint ───────────────────────────────────────────────────────────

if not is_authenticated():
    render_auth()
else:
    render_dashboard()

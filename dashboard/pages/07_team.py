"""
dashboard/pages/07_team.py - CostGuard v17.0

Team and user management for the enterprise dashboard.
"""
import streamlit as st
import pandas as pd
from utils.api_client import (
    is_authenticated, list_users, invite_user, update_user_role,
    get_user_activity, get_my_profile, get_gravatar_url, log_page_visit,
)

if not is_authenticated():
    st.error("🔒 Please login to access this page.")
    st.stop()

log_page_visit("07_team")

st.markdown("""
<style>
.cg-page-header{background:linear-gradient(135deg,rgba(255,107,53,.08) 0%,rgba(44,62,122,.06) 100%);
  border:1px solid rgba(255,107,53,.15);border-radius:16px;padding:24px 28px;margin-bottom:24px;}
.cg-page-header h1{font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;color:#fff;margin:0;}
.cg-page-header p{color:#6B7A99;margin:6px 0 0;}
.user-card{background:#0D1B2E;border:1px solid rgba(255,107,53,.12);border-radius:12px;padding:16px 20px;margin:8px 0;display:flex;align-items:center;gap:16px;}
</style>""", unsafe_allow_html=True)

st.markdown("""
<div class="cg-page-header">
  <h1>👥 Team & User Management</h1>
  <p>Manage your team, assign roles, track activity, and invite new members</p>
</div>""", unsafe_allow_html=True)

# Get current user profile to check admin status
profile = st.session_state.get("user_profile", {})
role = profile.get("role", "viewer")
is_admin = role == "admin"

tab1, tab2, tab3 = st.tabs(["👤 My Profile", "👥 Team Roster", "📋 Activity Log"])

# ── Tab 1: My Profile ─────────────────────────────────────────────────────────
with tab1:
    with st.spinner("Loading profile…"):
        my_profile = get_my_profile() or profile

    col1, col2 = st.columns([1, 2])
    with col1:
        email = my_profile.get("email", "")
        gravatar_url = get_gravatar_url(email, 80)
        st.markdown(f"""
        <div style="text-align:center;padding:20px;">
          <img src="{gravatar_url}" style="width:80px;height:80px;border-radius:50%;
               border:3px solid rgba(255,107,53,.5);margin-bottom:12px;display:block;margin-left:auto;margin-right:auto;">
          <div style="font-size:1.1rem;font-weight:700;color:#E8F0FE;">{my_profile.get('full_name','')}</div>
          <div style="font-size:.8rem;color:#6B7A99;margin-top:4px;">{email}</div>
          <div style="margin-top:8px;">
            <span class="badge badge-{'admin' if role=='admin' else ('warn' if role=='analyst' else 'allow')}">{role.upper()}</span>
          </div>
          <div style="font-size:.75rem;color:#6B7A99;margin-top:8px;">
            Powered by <a href="https://gravatar.com" style="color:#FF6B35;">Gravatar</a>
          </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("#### Edit Profile")
        with st.form("profile_form"):
            new_name = st.text_input("Full Name", value=my_profile.get("full_name", ""))
            new_phone = st.text_input("Phone (optional)", value=my_profile.get("phone", "") or "")
            save = st.form_submit_button("💾 Save Changes", use_container_width=True)
        if save:
            from utils.api_client import update_my_profile
            updates = {}
            if new_name.strip(): updates["full_name"] = new_name.strip()
            if new_phone.strip(): updates["phone"] = new_phone.strip()
            if updates:
                with st.spinner("Saving…"):
                    ok = update_my_profile(updates)
                if ok:
                    st.success("✅ Profile updated!")
                    if "full_name" in updates:
                        st.session_state["user_profile"] = {**profile, **updates}
                else:
                    st.error("❌ Failed to save profile.")
            else:
                st.info("No changes to save.")

# ── Tab 2: Team Roster ────────────────────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown("#### Team Members")
    with col_b:
        if st.button("🔄 Refresh"):
            st.rerun()

    if is_admin:
        with st.spinner("Loading team…"):
            users = list_users()

        if users:
            for u in users:
                u_role = u.get("role", "viewer")
                u_email = u.get("email", "")
                badge_cls = {"admin": "badge-admin", "analyst": "badge-warn", "viewer": "badge-allow"}.get(u_role, "badge-allow")
                last_login = u.get("last_login_at", "Never")
                if last_login and last_login != "Never":
                    try:
                        last_login = str(last_login)[:16]
                    except Exception:
                        last_login = "Unknown"
                st.markdown(f"""
                <div class="user-card">
                  <img src="{get_gravatar_url(u_email, 40)}" style="width:40px;height:40px;
                       border-radius:50%;border:2px solid rgba(255,107,53,.3);">
                  <div style="flex:1;">
                    <div style="font-weight:600;color:#E8F0FE;">{u.get('full_name','')}</div>
                    <div style="font-size:.8rem;color:#6B7A99;">{u_email}</div>
                  </div>
                  <span class="badge {badge_cls}">{u_role.upper()}</span>
                  <div style="font-size:.75rem;color:#6B7A99;">Last login: {last_login}</div>
                </div>""", unsafe_allow_html=True)

                # Role selector (inline)
                if u.get("id") != profile.get("id"):
                    new_role = st.selectbox(
                        f"Role for {u.get('full_name', u_email)}",
                        ["admin", "analyst", "viewer"],
                        index=["admin", "analyst", "viewer"].index(u_role),
                        key=f"role_{u.get('id')}",
                        label_visibility="collapsed",
                    )
                    if new_role != u_role:
                        if st.button(f"Update role → {new_role}", key=f"btn_role_{u.get('id')}"):
                            ok = update_user_role(u.get("id"), new_role)
                            if ok:
                                st.success(f"✅ Role updated to {new_role}")
                                st.rerun()
        else:
            st.info("No team members found.")

        st.markdown("---")
        st.markdown("#### ✉️ Invite Team Member")
        with st.form("invite_form"):
            inv_email = st.text_input("Email Address *")
            inv_name = st.text_input("Full Name *")
            inv_role = st.selectbox("Role", ["viewer", "analyst", "admin"])
            inv_submit = st.form_submit_button("📧 Send Invitation", use_container_width=True)
        if inv_submit:
            if not inv_email or "@" not in inv_email:
                st.error("Please enter a valid email address.")
            elif not inv_name.strip():
                st.error("Full name is required.")
            else:
                with st.spinner("Sending invitation…"):
                    result = invite_user(inv_email, inv_name, inv_role)
                if result:
                    st.success(f"✅ Invitation sent to {inv_email} as {inv_role.title()}")
                else:
                    st.error("❌ Failed to send invitation. Email may already be registered.")
    else:
        st.info("👤 Team roster management requires admin access. Contact your administrator to view or manage team members.")

# ── Tab 3: Activity Log ───────────────────────────────────────────────────────
with tab3:
    st.markdown("#### Your Recent Activity")
    with st.spinner("Loading activity…"):
        activity = get_user_activity()
    if activity:
        df = pd.DataFrame(activity)
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
        display = df[["action", "page", "created_at"]].rename(
            columns={"action": "Action", "page": "Page", "created_at": "Timestamp"}
        )
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.info("No activity recorded yet. Activity is tracked automatically as you use the platform.")

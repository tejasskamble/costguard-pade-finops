"""
dashboard/pages/08_support.py - CostGuard v17.0
Customer Support & Enquiry: form with file upload, FAQ accordion,
enquiry history, auto-acknowledgement email, Tawk.to chat placeholder.
"""
import os

import streamlit as st
import pandas as pd
from utils.api_client import (
    is_authenticated, submit_enquiry, get_my_enquiries, get_faq,
    log_page_visit, get_token,
)

SUPPORT_EMAIL = os.getenv("COSTGUARD_SUPPORT_EMAIL", "support@costguard.local")

if not is_authenticated():
    st.error("🔒 Please login to access this page.")
    st.stop()

log_page_visit("08_support")

st.markdown("""
<style>
.cg-page-header{background:linear-gradient(135deg,rgba(255,107,53,.08) 0%,rgba(44,62,122,.06) 100%);
  border:1px solid rgba(255,107,53,.15);border-radius:16px;padding:24px 28px;margin-bottom:24px;}
.cg-page-header h1{font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;color:#fff;margin:0;}
.cg-page-header p{color:#6B7A99;margin:6px 0 0;}
.status-open{color:#FF9500;} .status-resolved{color:#00D4AA;} .status-closed{color:#6B7A99;}
.enq-row{background:#0D1B2E;border:1px solid rgba(255,107,53,.12);border-radius:10px;
  padding:14px 18px;margin:6px 0;display:flex;align-items:center;gap:16px;}
</style>""", unsafe_allow_html=True)

st.markdown("""
<div class="cg-page-header">
  <h1>📨 Support & Enquiry</h1>
  <p>Get help from our team — typical response within 2 hours for business-hours submissions</p>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📝 New Enquiry", "📋 My Enquiries", "❓ FAQ"])

# ── Tab 1: New Enquiry ────────────────────────────────────────────────────────
with tab1:
    profile = st.session_state.get("user_profile", {})

    with st.form("enquiry_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name *",
                                  value=profile.get("full_name", ""),
                                  max_chars=255)
            email = st.text_input("Email Address *",
                                   value=profile.get("email", ""),
                                   max_chars=255)
        with col2:
            phone = st.text_input("Phone (optional)", max_chars=20)
            priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])

        subject = st.text_input("Subject *", max_chars=255,
                                 placeholder="Brief description of your issue")
        category = st.selectbox("Category", [
            "Technical", "Billing", "Feature Request", "Bug Report", "Other"
        ])
        message = st.text_area("Message *", max_chars=2000, height=150,
                                placeholder="Describe your issue in detail (minimum 20 characters)…")

        char_count = len(message)
        st.caption(f"{char_count}/2000 characters {'🔴' if char_count > 1800 else '🟢'}")

        attachment = st.file_uploader(
            "Attach Screenshot (optional) — PNG, JPG, WEBP — max 5MB",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False,
        )

        submitted = st.form_submit_button("📤 Submit Enquiry", use_container_width=True)

    if submitted:
        errors = []
        if not name.strip(): errors.append("Full name is required")
        if not email.strip() or "@" not in email: errors.append("Valid email is required")
        if not subject.strip(): errors.append("Subject is required")
        if len(message.strip()) < 20: errors.append("Message must be at least 20 characters")

        if errors:
            for e in errors:
                st.error(f"❌ {e}")
        else:
            form_data = {
                "name": name.strip(), "email": email.strip(), "phone": phone,
                "subject": subject.strip(), "category": category.lower(),
                "priority": priority.lower(), "message": message.strip(),
            }
            with st.spinner("Submitting your enquiry…"):
                result = submit_enquiry(form_data, attachment)

            if result:
                enq_id = result.get("id", 0)
                st.success(f"""
                ✅ **Enquiry #{enq_id:05d} submitted successfully!**

                We'll respond to **{email}** within 2 hours.
                Track your enquiry in the **My Enquiries** tab.
                """)
                st.balloons()
            # Error is shown by submit_enquiry helper if it fails

    # Tawk.to live chat widget placeholder
    st.markdown("---")
    st.markdown("#### 💬 Live Chat")
    st.markdown(f"""
    <div style="background:#111D35;border:1px solid rgba(255,107,53,.15);border-radius:12px;
                padding:20px;text-align:center;color:#6B7A99;">
      <div style="font-size:1.5rem;margin-bottom:8px;">💬</div>
      <div style="font-weight:600;color:#E8F0FE;margin-bottom:4px;">Live Chat Available</div>
      <div style="font-size:.85rem;">
        Live chat is available during business hours (9 AM – 6 PM IST, Mon–Fri).
        Outside business hours, use the enquiry form above.
      </div>
      <div style="margin-top:12px;font-size:.75rem;color:#444D5E;">
        Powered by Tawk.to — <a href="https://tawk.to" style="color:#FF6B35;">tawk.to</a>
      </div>
    </div>""", unsafe_allow_html=True)

# ── Tab 2: My Enquiries ───────────────────────────────────────────────────────
with tab2:
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown("#### Your Support History")
    with col_b:
        if st.button("🔄 Refresh", key="refresh_enq"):
            st.rerun()

    with st.spinner("Loading enquiries…"):
        enquiries = get_my_enquiries()

    if enquiries:
        status_colors = {
            "open": "#FF9500", "in_progress": "#4A9EFF",
            "resolved": "#00D4AA", "closed": "#6B7A99",
        }
        priority_colors = {
            "critical": "#FF3B5C", "high": "#FF9500",
            "medium": "#4A9EFF", "low": "#00D4AA",
        }
        for enq in enquiries:
            sc = status_colors.get(enq.get("status", "open"), "#6B7A99")
            pc = priority_colors.get(enq.get("priority", "medium"), "#4A9EFF")
            created = str(enq.get("created_at", ""))[:16]
            st.markdown(f"""
            <div class="enq-row">
              <div style="flex:1;">
                <div style="font-weight:600;color:#E8F0FE;">#{enq.get('id',0):05d} — {enq.get('subject','')}</div>
                <div style="font-size:.78rem;color:#6B7A99;margin-top:3px;">{created} · {enq.get('category','').title()}</div>
              </div>
              <span style="padding:3px 10px;border-radius:20px;font-size:.7rem;font-weight:700;
                           text-transform:uppercase;background:{pc}22;color:{pc};border:1px solid {pc}44;">
                {enq.get('priority','medium').upper()}
              </span>
              <span style="padding:3px 10px;border-radius:20px;font-size:.7rem;font-weight:700;
                           text-transform:uppercase;background:{sc}22;color:{sc};border:1px solid {sc}44;">
                {enq.get('status','open').replace('_',' ').upper()}
              </span>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("You haven't submitted any enquiries yet. Use the **New Enquiry** tab to get started.")

# ── Tab 3: FAQ ────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### Frequently Asked Questions")
    with st.spinner("Loading FAQ…"):
        faq = get_faq()

    if not faq:
        # Fallback static FAQ
        faq = [
            {"q": "How does CostGuard detect anomalies?",
             "a": "CostGuard uses the canonical PADE v17.0 engine - a BahdanauBiLSTM + GATv2 ensemble - to compute a Cost Risk Score (CRS) for each pipeline run, then classifies it as ALLOW / WARN / AUTO_OPTIMISE / BLOCK."},
            {"q": "How do I reset my password?",
             "a": "Click 'Forgot Password?' on the login page. Enter your email to receive a 6-digit OTP valid for 10 minutes."},
            {"q": "What cloud providers are supported?",
             "a": "AWS, GCP, Azure, and Self-Hosted. Multi-cloud cost attribution is available in Enterprise edition."},
        ]

    for i, item in enumerate(faq):
        with st.expander(f"❓ {item.get('q', '')}"):
            st.markdown(f"""<div style="color:#B0BDD0;line-height:1.7;padding:4px 0;">
                {item.get('a', '')}</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="background:#111D35;border-radius:12px;padding:16px 20px;text-align:center;">
      <div style="color:#B0BDD0;font-size:.88rem;">
        Can't find your answer? <strong style="color:#FF6B35;">Submit an enquiry</strong> and
        our team will respond within 2 hours.<br>
        📧 <a href="mailto:{SUPPORT_EMAIL}" style="color:#FF6B35;">{SUPPORT_EMAIL}</a>
        &nbsp;|&nbsp; Sir Parshurambhau College (Autonomous), Pune
      </div>
    </div>""", unsafe_allow_html=True)

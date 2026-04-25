"""
dashboard/pages/03_cost_query.py - CostGuard v17.0

Natural language cost analysis and query history for CostGuard.
"""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.api_client import stream_query, post_query, get_query_history, save_query

PLOTLY_LAYOUT = dict(
    plot_bgcolor  = "#0C1428",
    paper_bgcolor = "#0C1428",
    font          = dict(family="DM Sans, sans-serif", color="#94A3B8", size=12),
    xaxis         = dict(gridcolor="rgba(99,102,241,0.1)", tickfont=dict(color="#6B7A99")),
    yaxis         = dict(gridcolor="rgba(99,102,241,0.1)", tickfont=dict(color="#6B7A99")),
    margin        = dict(l=16, r=16, t=40, b=16),
    hoverlabel    = dict(bgcolor="rgba(12,20,40,0.95)", font=dict(family="DM Sans", color="#E8EAF6")),
)

STARTER_PROMPTS = [
    "Which pipeline stage costs the most this week?",
    "Show me all BLOCK decisions in the last 24 hours",
    "What is the average cost per pipeline run?",
    "Which provider has the highest anomaly rate?",
    "List the top 5 most expensive stages this month",
]


def _auto_chart(df: pd.DataFrame) -> None:
    """Render a bar chart when result has 1 categorical + 1 numeric column. (CONSTRAINT-B)"""
    if df.empty or len(df.columns) < 2:
        return
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not cat_cols or not num_cols:
        return
    x_col, y_col = cat_cols[0], num_cols[0]
    fig = go.Figure(go.Bar(
        x=df[x_col], y=df[y_col],
        marker_color="#6366F1", marker_line_width=0,
        hovertemplate=f"<b>%{{x}}</b><br>{y_col}: %{{y}}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title_text=f"{y_col} by {x_col}", showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_history_msg(msg: dict) -> None:
    """Render a single message from session state."""
    if msg["role"] == "user":
        st.markdown(
            f'<div class="chat-user chat-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chat-ai chat-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        if msg.get("sql"):
            with st.expander("📎 View generated SQL", expanded=False):
                st.code(msg["sql"], language="sql")
        if msg.get("rows") and msg.get("columns"):
            df = pd.DataFrame(msg["rows"], columns=msg["columns"])
            st.markdown('<div style="background:rgba(99,102,241,.06);border:1px solid rgba(99,102,241,.15);border-radius:14px;overflow:hidden;padding:4px;margin-top:8px;">', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            _auto_chart(df)


def show():
    st.markdown("""
    <div class="page-header">
        <h1>💬 AI Cost Query</h1>
        <p>Ask plain-English questions — answers stream word-by-word via GPT-4o-mini.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── NEW-BUG-4 FIX: Restore history from DB on first load ──────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Populate from DB if empty (first page load in session)
        history = get_query_history(limit=10)
        if history:
            for h in reversed(history):  # oldest first
                st.session_state.messages.append({
                    "role":    "user",
                    "content": h.get("question", ""),
                })
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": h.get("nl_answer", ""),
                    "sql":     h.get("sql_generated"),
                    "rows":    None,   # don't re-fetch rows from history
                    "columns": None,
                })

    # ── Empty state with starter prompts ─────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center;padding:32px 0 24px;">
            <div style="font-size:3rem;margin-bottom:12px;">🤖</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;color:#fff;">
                Ask anything about your pipeline costs
            </div>
            <div style="color:#6B7A99;margin-top:8px;font-size:.92rem;">
                Answers stream word-by-word · SQL shown inline · Auto-charts for tabular data
            </div>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(len(STARTER_PROMPTS))
        for i, (col, prompt) in enumerate(zip(cols, STARTER_PROMPTS)):
            with col:
                if st.button(prompt, key=f"starter_{i}", use_container_width=True):
                    st.session_state._pending_prompt = prompt
                    st.rerun()
        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

    # ── Render existing messages ──────────────────────────────────────────────
    for msg in st.session_state.messages:
        _render_history_msg(msg)

    # ── Handle pending starter prompt ────────────────────────────────────────
    pending = st.session_state.pop("_pending_prompt", None)

    # ── Chat input ────────────────────────────────────────────────────────────
    prompt = st.chat_input("Ask about your pipeline costs…") or pending

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(
            f'<div class="chat-user chat-bubble">{prompt}</div>',
            unsafe_allow_html=True,
        )

        # ── FEATURE-6: Streaming response ────────────────────────────────────
        sql_received  = None
        rows_received = None
        cols_received = None
        row_count     = 0
        full_answer   = ""

        answer_placeholder = st.empty()
        accumulated_text   = ""

        with st.spinner("Translating to SQL…"):
            stream = stream_query(prompt)
            first_event = next(stream, None)

        if first_event and "error" in first_event:
            # Streaming unavailable — fall back to blocking /api/query
            with st.spinner("Fetching answer…"):
                result = post_query(prompt)
            if result:
                full_answer   = result.get("natural_language_answer", "")
                sql_received  = result.get("sql_generated")
                rows_received = result.get("rows")
                cols_received = result.get("columns")
                row_count     = result.get("row_count", 0)
        else:
            # First event: SQL metadata
            if first_event:
                sql_received  = first_event.get("sql")
                rows_received = first_event.get("rows")
                cols_received = first_event.get("columns")
                row_count     = first_event.get("row_count", 0)

            # Show SQL immediately if available
            if sql_received:
                with st.expander("📎 View generated SQL", expanded=False):
                    st.code(sql_received, language="sql")

            # Stream tokens word-by-word
            st.markdown('<div class="chat-ai chat-bubble" id="stream-target">', unsafe_allow_html=True)
            stream_container = st.empty()
            for event in stream:
                if event.get("done"):
                    break
                token = event.get("token", "")
                if token:
                    accumulated_text += token
                    stream_container.markdown(
                        f'<div style="color:#E8EAF6;font-size:.95rem;line-height:1.6;">{accumulated_text}▌</div>',
                        unsafe_allow_html=True,
                    )
            # Final render without cursor
            stream_container.markdown(
                f'<div style="color:#E8EAF6;font-size:.95rem;line-height:1.6;">{accumulated_text}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
            full_answer = accumulated_text

        # Render metadata block (row count + elapsed)
        if row_count:
            st.markdown(f"""
            <div style="display:flex;gap:16px;align-items:center;margin:8px 0 4px;
                        color:#6B7A99;font-size:.8rem;">
              <span>📋 {row_count} row{"s" if row_count != 1 else ""}</span>
            </div>
            """, unsafe_allow_html=True)

        # Render data table and auto-chart
        if rows_received and cols_received:
            df_result = pd.DataFrame(rows_received, columns=cols_received)
            st.markdown('<div style="background:rgba(99,102,241,.06);border:1px solid rgba(99,102,241,.15);border-radius:14px;overflow:hidden;margin-top:4px;padding:4px;">', unsafe_allow_html=True)
            st.dataframe(df_result, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            _auto_chart(df_result)

        # Store in session state
        st.session_state.messages.append({
            "role":    "assistant",
            "content": full_answer or "Sorry, I couldn't process that query.",
            "sql":     sql_received,
            "rows":    rows_received,
            "columns": cols_received,
        })

        # NEW-BUG-4: Persist to DB (fire-and-forget)
        if full_answer:
            save_query(
                question=prompt,
                sql=sql_received or "",
                nl_answer=full_answer,
                row_count=row_count,
            )

    # ── Clear chat button ─────────────────────────────────────────────────────
    if st.session_state.messages:
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        col_l, col_c, col_r = st.columns([3, 1, 3])
        with col_c:
            if st.button("Clear Chat", use_container_width=True, key="clear_chat"):
                st.session_state.messages = []
                st.rerun()


show()

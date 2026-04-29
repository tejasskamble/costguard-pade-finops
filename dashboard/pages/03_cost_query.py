"""
dashboard/pages/03_cost_query.py - CostGuard v17.0

Natural language cost analysis and query history for CostGuard.
"""
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.cinematic_ui import apply_cinematic_ui, cinematic_header
from utils.api_client import get_query_history, post_query, save_query, stream_query

apply_cinematic_ui("03_cost_query")

PLOTLY_LAYOUT = dict(
    plot_bgcolor="rgba(8,12,24,0.82)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#94A3B8", size=12),
    xaxis=dict(gridcolor="rgba(99,102,241,0.1)", tickfont=dict(color="#6B7A99")),
    yaxis=dict(gridcolor="rgba(99,102,241,0.1)", tickfont=dict(color="#6B7A99")),
    margin=dict(l=16, r=16, t=40, b=16),
    hoverlabel=dict(bgcolor="rgba(12,20,40,0.95)", font=dict(family="DM Sans", color="#E8EAF6")),
)

STARTER_PROMPTS = [
    "Which pipeline stage costs the most this week?",
    "Show all BLOCK decisions in the last 24 hours",
    "What is the average cost per pipeline run?",
    "Which provider has the highest anomaly rate?",
    "List the top 5 most expensive stages this month",
]


def _highlight_response_metrics(text: str) -> str:
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for pattern, repl in [
        (r"(\$[0-9][0-9,]*(?:\.[0-9]+)?)", r'<span style="color:#00FFB2;font-weight:700;">\1</span>'),
        (r"(\b[0-9]+(?:\.[0-9]+)?%)", r'<span style="color:#FFD54F;font-weight:700;">\1</span>'),
        (r"(\bCRS\b\s*[:=]?\s*[0-9]+(?:\.[0-9]+)?)", r'<span style="color:#00E5FF;font-weight:700;">\1</span>'),
    ]:
        escaped = re.sub(pattern, repl, escaped, flags=re.IGNORECASE)
    return escaped


def _auto_chart(df: pd.DataFrame) -> None:
    """Render a bar chart when result has one categorical and one numeric column."""
    if df.empty or len(df.columns) < 2:
        return
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not cat_cols or not num_cols:
        return
    x_col, y_col = cat_cols[0], num_cols[0]
    fig = go.Figure(
        go.Bar(
            x=df[x_col],
            y=df[y_col],
            marker_color="#6366F1",
            marker_line_width=0,
            hovertemplate=f"<b>%{{x}}</b><br>{y_col}: %{{y}}<extra></extra>",
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT, title_text=f"{y_col} by {x_col}", showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_history_msg(msg: dict) -> None:
    """Render a single message from session state."""
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-user chat-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        return

    highlighted = _highlight_response_metrics(msg["content"])
    st.markdown(f'<div class="chat-ai chat-bubble">{highlighted}</div>', unsafe_allow_html=True)

    if msg.get("sql"):
        with st.expander("View generated SQL", expanded=False):
            st.code(msg["sql"], language="sql")

    if msg.get("rows") and msg.get("columns"):
        df = pd.DataFrame(msg["rows"], columns=msg["columns"])
        st.markdown(
            '<div style="background:rgba(99,102,241,.06);border:1px solid rgba(99,102,241,.15);border-radius:14px;overflow:hidden;padding:4px;margin-top:8px;">',
            unsafe_allow_html=True,
        )
        st.dataframe(df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        _auto_chart(df)


def show() -> None:
    st.markdown(
        cinematic_header(
            "JARVIS Cost Query Panel",
            "Ask plain-English questions and receive streaming FinOps intelligence.",
            icon="ASSISTANT",
            status="Voice Interface Standby",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="jarvis-console">
          <span class="jarvis-voice">Voice Ready
            <span class="voice-bars"><span></span><span></span><span></span><span></span></span>
          </span>
          <div style="margin-top:0.6rem;color:#9DB4DA;font-size:0.82rem;">
            Response stream uses SQL grounding with live natural-language synthesis.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
        history = get_query_history(limit=10)
        if history:
            for item in reversed(history):
                st.session_state.messages.append({"role": "user", "content": item.get("question", "")})
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": item.get("nl_answer", ""),
                        "sql": item.get("sql_generated"),
                        "rows": None,
                        "columns": None,
                    }
                )

    if not st.session_state.messages:
        st.markdown(
            """
            <div style="text-align:center;padding:32px 0 24px;">
                <div style="font-size:3rem;margin-bottom:12px;">AI</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;color:#fff;">
                    Ask anything about your pipeline costs
                </div>
                <div style="color:#6B7A99;margin-top:8px;font-size:.92rem;">
                    Streamed answers, SQL visibility, and auto-charts for tabular results.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cols = st.columns(len(STARTER_PROMPTS))
        for idx, (col, starter) in enumerate(zip(cols, STARTER_PROMPTS)):
            with col:
                if st.button(starter, key=f"starter_{idx}", use_container_width=True):
                    st.session_state._pending_prompt = starter
                    st.rerun()
        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        _render_history_msg(msg)

    pending = st.session_state.pop("_pending_prompt", None)
    prompt = st.chat_input("Ask about your pipeline costs...") or pending

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f'<div class="chat-user chat-bubble">{prompt}</div>', unsafe_allow_html=True)

        typing_placeholder = st.empty()
        typing_placeholder.markdown(
            """
            <div class="chat-ai chat-bubble">
              Analyzing telemetry
              <span class="voice-bars" style="margin-left:0.6rem;"><span></span><span></span><span></span><span></span></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sql_received = None
        rows_received = None
        cols_received = None
        row_count = 0
        full_answer = ""
        accumulated_text = ""

        with st.spinner("Translating to SQL..."):
            stream = stream_query(prompt)
            first_event = next(stream, None)

        if first_event and "error" in first_event:
            with st.spinner("Fetching answer..."):
                result = post_query(prompt)
            if result:
                full_answer = result.get("natural_language_answer", "")
                sql_received = result.get("sql_generated")
                rows_received = result.get("rows")
                cols_received = result.get("columns")
                row_count = result.get("row_count", 0)
            typing_placeholder.empty()
        else:
            if first_event:
                sql_received = first_event.get("sql")
                rows_received = first_event.get("rows")
                cols_received = first_event.get("columns")
                row_count = first_event.get("row_count", 0)

            if sql_received:
                with st.expander("View generated SQL", expanded=False):
                    st.code(sql_received, language="sql")

            typing_placeholder.empty()
            st.markdown('<div class="chat-ai chat-bubble" id="stream-target">', unsafe_allow_html=True)
            stream_container = st.empty()
            for event in stream:
                if event.get("done"):
                    break
                token = event.get("token", "")
                if token:
                    accumulated_text += token
                    highlighted = _highlight_response_metrics(accumulated_text)
                    stream_container.markdown(
                        f'<div style="color:#E8EAF6;font-size:.95rem;line-height:1.6;">{highlighted}...</div>',
                        unsafe_allow_html=True,
                    )
            highlighted = _highlight_response_metrics(accumulated_text)
            stream_container.markdown(
                f'<div style="color:#E8EAF6;font-size:.95rem;line-height:1.6;">{highlighted}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
            full_answer = accumulated_text

        if row_count:
            suffix = "s" if row_count != 1 else ""
            st.markdown(
                f"""
                <div style="display:flex;gap:16px;align-items:center;margin:8px 0 4px;color:#6B7A99;font-size:.8rem;">
                  <span>{row_count} row{suffix}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if rows_received and cols_received:
            df_result = pd.DataFrame(rows_received, columns=cols_received)
            st.markdown(
                '<div style="background:rgba(99,102,241,.06);border:1px solid rgba(99,102,241,.15);border-radius:14px;overflow:hidden;margin-top:4px;padding:4px;">',
                unsafe_allow_html=True,
            )
            st.dataframe(df_result, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            _auto_chart(df_result)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_answer or "Sorry, I could not process that query.",
                "sql": sql_received,
                "rows": rows_received,
                "columns": cols_received,
            }
        )

        if full_answer:
            save_query(
                question=prompt,
                sql=sql_received or "",
                nl_answer=full_answer,
                row_count=row_count,
            )

    if st.session_state.messages:
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        _, mid, _ = st.columns([3, 1, 3])
        with mid:
            if st.button("Clear Chat", use_container_width=True, key="clear_chat"):
                st.session_state.messages = []
                st.rerun()


show()

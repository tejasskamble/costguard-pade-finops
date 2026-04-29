"""Shared cinematic UI loader for Streamlit pages."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


_REPO_ROOT = Path(__file__).resolve().parents[2]
_UI_ROOT = _REPO_ROOT / "frontend_ui"


@lru_cache(maxsize=None)
def _asset_text(name: str) -> str:
    path = _UI_ROOT / name
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def apply_cinematic_ui(page_key: str = "global") -> None:
    """Inject shared CSS and JS cinematic effects."""
    css = _asset_text("styles.css")
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    js_loaded_key = f"_cg_js_loaded_{page_key}"
    if not st.session_state.get(js_loaded_key):
        js = _asset_text("animations.js")
        if js:
            safe_js = js.replace("</script>", "<\\/script>")
            components.html(
                f"<script>{safe_js}</script>",
                height=0,
                width=0,
                scrolling=False,
            )
        st.session_state[js_loaded_key] = True


def cinematic_header(title: str, subtitle: str, icon: str = "CONTROL", status: str = "LIVE") -> str:
    """Render the shared command-center header."""
    return f"""
    <div class="cg-command-header">
      <h1>{icon} {title}</h1>
      <p>{subtitle}</p>
      <span class="cg-status-chip">{status}</span>
    </div>
    """

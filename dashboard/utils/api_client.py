"""
dashboard/utils/api_client.py - CostGuard v17.0 dashboard client.

Shared API helpers for the live Streamlit application.
"""
import json
import logging
import os
import requests
import streamlit as st
from typing import Any, Dict, Generator, List, Optional

from costguard_runtime import retry_sync

logger = logging.getLogger(__name__)
REQUEST_TIMEOUT = 15
GENERIC_ERROR_BODY = "We couldn't complete that request. Please try again or check backend logs."
STREAM_ERROR_BODY = "The live stream was interrupted. Please retry the request."
RETRYABLE_REQUEST_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


def _get_runtime_value(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        secret_value = st.secrets.get(key)
        if secret_value:
            return str(secret_value)
    except Exception:
        pass
    env_value = os.getenv(key)
    if env_value:
        return env_value
    return default


def _resolve_api_http_base() -> str:
    raw = (_get_runtime_value("COSTGUARD_API_BASE_URL", "http://localhost:7860") or "").rstrip("/")
    if raw.endswith("/api"):
        raw = raw[:-4]
    return raw.rstrip("/")


API_HTTP_BASE = _resolve_api_http_base()
API_BASE = f"{API_HTTP_BASE}/api"


def get_api_http_base_url() -> str:
    return API_HTTP_BASE


def _response_detail(resp: requests.Response) -> str:
    try:
        payload = resp.json()
        detail = payload.get("detail")
        if isinstance(detail, dict):
            return detail.get("message") or json.dumps(detail)
        if detail:
            return str(detail)
    except Exception:
        pass
    return f"HTTP {resp.status_code}"


# ── Themed helpers ────────────────────────────────────────────────────────────

def _err(icon: str, title: str, body: str) -> None:
    st.markdown(f"""
    <div style="background:rgba(244,63,94,.08);border:1px solid rgba(244,63,94,.25);
                border-left:4px solid #F43F5E;border-radius:12px;
                padding:14px 18px;margin:8px 0;display:flex;gap:14px;">
        <span style="font-size:1.4rem;flex-shrink:0;">{icon}</span>
        <div>
            <div style="color:#FB7185;font-weight:700;font-size:.9rem;">{title}</div>
            <div style="color:#94A3B8;font-size:.83rem;margin-top:4px;">{body}</div>
        </div>
    </div>""", unsafe_allow_html=True)


def _warn(icon: str, title: str, body: str) -> None:
    st.markdown(f"""
    <div style="background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.25);
                border-left:4px solid #F59E0B;border-radius:12px;
                padding:14px 18px;margin:8px 0;display:flex;gap:14px;">
        <span style="font-size:1.4rem;flex-shrink:0;">{icon}</span>
        <div>
            <div style="color:#FCD34D;font-weight:700;font-size:.9rem;">{title}</div>
            <div style="color:#94A3B8;font-size:.83rem;margin-top:4px;">{body}</div>
        </div>
    </div>""", unsafe_allow_html=True)


# ── Token management ─────────────────────────────────────────────────────────

def get_token() -> Optional[str]:
    return st.session_state.get("token")

def set_token(t: Optional[str]) -> None:
    st.session_state.token = t

def is_authenticated() -> bool:
    return bool(st.session_state.get("authenticated", False))

def _headers() -> Dict[str, str]:
    token = get_token()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _perform_request(method: str, url: str, **kwargs) -> requests.Response:
    """Execute a bounded-retry HTTP request for the dashboard client."""
    return retry_sync(
        lambda: requests.request(method, url, headers=_headers(), timeout=REQUEST_TIMEOUT, **kwargs),
        attempts=2,
        delay=0.25,
        exceptions=RETRYABLE_REQUEST_EXCEPTIONS,
        logger=logger,
        label=f"{method.upper()} {url}",
    )


# ── Generic safe request ──────────────────────────────────────────────────────

def safe_request(method: str, url: str, show_errors: bool = True, **kwargs) -> Optional[Any]:
    try:
        resp = _perform_request(method, url, **kwargs)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 401:
            if show_errors:
                _warn("🔑", "Session Expired", _response_detail(resp) or "Please sign out and sign back in.")
            st.session_state.update({"authenticated": False, "token": None})
            return None
        if resp.status_code == 403:
            if show_errors:
                _warn("🛡️", "Access Denied", _response_detail(resp) or "This action requires an admin account.")
            return None
        if resp.status_code == 404:
            if show_errors:
                _warn("🔍", "Not Found", f"Resource not found: `{url}`")
            return None
        if 400 <= resp.status_code < 500:
            if show_errors:
                _warn("⚠️", f"Request Rejected ({resp.status_code})", _response_detail(resp))
            return None
        if resp.status_code >= 500 and show_errors:
            _err("💥", f"Server Error ({resp.status_code})", "Check backend logs.")
        return None
    except requests.exceptions.ConnectionError:
        if show_errors:
            _err("📡", "Backend Unreachable", f"Cannot connect to `{API_HTTP_BASE}`.")
        return None
    except requests.exceptions.Timeout:
        if show_errors:
            _err("⏱️", "Request Timed Out", "Server is slow — try again.")
        return None
    except Exception as exc:
        logger.warning("Dashboard request failed for %s %s: %s", method.upper(), url, exc)
        if show_errors:
            _err("⚠️", "Unexpected Error", GENERIC_ERROR_BODY)
        return None


# ── Auth ──────────────────────────────────────────────────────────────────────

def login(email: str, password: str) -> Optional[str]:
    try:
        resp = retry_sync(
            lambda: requests.post(
                f"{API_BASE}/auth/token",
                data={"username": email, "password": password},
                timeout=10,
            ),
            attempts=2,
            delay=0.25,
            exceptions=RETRYABLE_REQUEST_EXCEPTIONS,
            logger=logger,
            label="dashboard login",
        )
        if resp.status_code == 200:
            return resp.json().get("access_token")
        # FIX 9: surface the server's error detail so UI can distinguish
        # network failure vs wrong credentials vs account locked
        try:
            detail = resp.json().get("detail", "Authentication failed")
        except Exception:
            detail = f"HTTP {resp.status_code}"
        raise ValueError(detail)
    except ValueError:
        raise
    except Exception as exc:
        logger.warning("Dashboard login failed for %s: %s", email, exc)
        _err("⚠️", "Login Error", GENERIC_ERROR_BODY)
        return None


def register(email: str, password: str, full_name: str) -> Optional[Dict]:
    try:
        resp = retry_sync(
            lambda: requests.post(
                f"{API_BASE}/auth/register",
                json={"email": email, "password": password, "full_name": full_name},
                timeout=10,
            ),
            attempts=2,
            delay=0.25,
            exceptions=RETRYABLE_REQUEST_EXCEPTIONS,
            logger=logger,
            label="dashboard register",
        )
        # FIX 1: always try to parse JSON so error detail reaches the UI
        try:
            return resp.json()
        except Exception:
            return {"detail": f"HTTP {resp.status_code} — server error"}
    except Exception as exc:
        logger.warning("Dashboard register failed for %s: %s", email, exc)
        _err("⚠️", "Register Error", GENERIC_ERROR_BODY)
        return None


def get_me() -> Optional[Dict]:
    return safe_request("GET", f"{API_BASE}/auth/me", show_errors=False)


# ── Alerts & attribution ──────────────────────────────────────────────────────

def get_recent_alerts(limit: int = 50) -> Optional[List[Dict]]:
    return safe_request("GET", f"{API_BASE}/alerts/recent?limit={limit}", show_errors=False)


def get_attribution(run_id: str) -> Optional[List[Dict]]:
    return safe_request("GET", f"{API_BASE}/attribution/{run_id}")


def get_run_ids(limit: int = 50) -> List[str]:
    data = get_recent_alerts(limit=limit)
    if data:
        return list(dict.fromkeys(a["run_id"] for a in data if "run_id" in a))
    return []


def get_daily_summary() -> Optional[Dict]:
    return safe_request("GET", f"{API_BASE}/alerts/summary", show_errors=False)


# ── Policy ────────────────────────────────────────────────────────────────────

def get_policy() -> Optional[Dict]:
    return safe_request("GET", f"{API_BASE}/policy")


def update_policy(warn: float, auto: float, block: float, policy_bundle: Optional[Dict] = None) -> bool:
    payload: Dict[str, Any] = {
        "warn_threshold": warn,
        "auto_optimise_threshold": auto,
        "block_threshold": block,
    }
    if policy_bundle is not None:
        payload["policy_bundle"] = policy_bundle
    result = safe_request("PUT", f"{API_BASE}/policy", json=payload)
    return result is not None


# ── LCQI ─────────────────────────────────────────────────────────────────────

def post_query(question: str) -> Optional[Dict]:
    return safe_request("POST", f"{API_BASE}/query", json={"question": question})


def stream_query(question: str) -> Generator[dict, None, None]:
    """
    FEATURE-6: Streams NL answer tokens from GET /api/query/stream.
    Yields dicts: {'sql': ..., 'columns': ..., 'rows': ..., 'row_count': ...}
                  then {'token': '...'} per word
                  then {'done': True}
    """
    try:
        with requests.get(
            f"{API_BASE}/query/stream",
            params={"question": question},
            headers=_headers(),
            stream=True,
            timeout=60,
        ) as resp:
            if resp.status_code != 200:
                yield {"error": f"HTTP {resp.status_code}"}
                return
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8")
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        yield {"done": True}
                        return
                    try:
                        yield json.loads(payload)
                    except json.JSONDecodeError:
                        logger.debug("Ignoring non-JSON stream payload from query endpoint")
    except requests.exceptions.ConnectionError:
        yield {"error": "Backend unreachable"}
    except Exception as exc:
        logger.warning("Streaming query failed: %s", exc)
        yield {"error": STREAM_ERROR_BODY}


# ── NEW-BUG-4: Query history ──────────────────────────────────────────────────

def get_query_history(limit: int = 20) -> List[Dict]:
    data = safe_request("GET", f"{API_BASE}/query/history?limit={limit}", show_errors=False)
    return data or []


def save_query(question: str, sql: str, nl_answer: str, row_count: int) -> None:
    """Fire-and-forget: persist query to history (ignore failures)."""
    try:
        _perform_request(
            "POST",
            f"{API_BASE}/query/save",
            json={
                "question": question,
                "sql_generated": sql,
                "nl_answer": nl_answer,
                "row_count": row_count,
            },
            timeout=5,
        )
    except Exception as exc:
        logger.debug("Query history save skipped: %s", exc)


# ── FEATURE-1: Forecast ──────────────────────────────────────────────────────

def get_forecast(horizon_days: int = 7) -> Optional[Dict]:
    return safe_request(
        "GET", f"{API_BASE}/attribution/forecast?horizon_days={horizon_days}",
        show_errors=False,
    )


# ── FEATURE-2: DAG ────────────────────────────────────────────────────────────

def get_dag(run_id: Optional[str] = None) -> Optional[Dict]:
    url = f"{API_BASE}/pade/dag"
    if run_id:
        url += f"?run_id={run_id}"
    return safe_request("GET", url, show_errors=False)


# ── FEATURE-5: Remediation YAML ───────────────────────────────────────────────

def remediate(run_id: str) -> Optional[bytes]:
    """Download the remediation YAML for a BLOCK run_id."""
    try:
        resp = retry_sync(
            lambda: requests.get(
                f"{API_BASE}/remediate/{run_id}",
                headers=_headers(),
                timeout=10,
            ),
            attempts=2,
            delay=0.25,
            exceptions=RETRYABLE_REQUEST_EXCEPTIONS,
            logger=logger,
            label=f"remediation download {run_id}",
        )
        if resp.status_code == 200:
            return resp.content
        if resp.status_code == 404:
            return None
        _err("⚠️", "Remediation Error", f"HTTP {resp.status_code}")
        return None
    except Exception as exc:
        logger.warning("Remediation download failed for %s: %s", run_id, exc)
        _err("⚠️", "Remediation Error", GENERIC_ERROR_BODY)
        return None


# ── PADE status ───────────────────────────────────────────────────────────────

def get_pade_status() -> Dict:
    return safe_request("GET", f"{API_BASE}/pade/status", show_errors=False) or {}


# ── Simulation ───────────────────────────────────────────────────────────────

def simulate_pipeline(anomaly_level: float, stage_name: str) -> Optional[Dict]:
    try:
        resp = retry_sync(
            lambda: requests.post(
                f"{API_BASE}/ingest/simulate",
                params={"anomaly_level": anomaly_level, "stage_name": stage_name},
                headers=_headers(),
                timeout=30,
            ),
            attempts=2,
            delay=0.25,
            exceptions=RETRYABLE_REQUEST_EXCEPTIONS,
            logger=logger,
            label="pipeline simulation",
        )
        # FIX 1: same pattern as register() — always parse JSON to surface errors
        try:
            return resp.json()
        except Exception:
            return {"detail": f"HTTP {resp.status_code} — server error"}
    except requests.exceptions.Timeout:
        _err("⏱️", "Simulation Timed Out", "PADE inference is slow — try again.")
        return None
    except Exception as exc:
        logger.warning("Pipeline simulation failed: %s", exc)
        _err("⚠️", "Simulation Error", GENERIC_ERROR_BODY)
        return None


# ── Checkpoint upload ─────────────────────────────────────────────────────────

def upload_checkpoint(file_bytes: bytes) -> Optional[Dict]:
    try:
        resp = retry_sync(
            lambda: requests.post(
                f"{API_BASE}/pade/load-checkpoint",
                files={"file": ("gat.pt", file_bytes, "application/octet-stream")},
                headers=_headers(),
                timeout=30,
            ),
            attempts=2,
            delay=0.25,
            exceptions=RETRYABLE_REQUEST_EXCEPTIONS,
            logger=logger,
            label="checkpoint upload",
        )
        try:
            payload = resp.json()
        except Exception:
            payload = {"detail": f"HTTP {resp.status_code}"}
        return payload
    except Exception as exc:
        logger.warning("Checkpoint upload failed: %s", exc)
        _err("⚠️", "Upload Failed", GENERIC_ERROR_BODY)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CostGuard v17.0 - Extended dashboard API helpers
# ══════════════════════════════════════════════════════════════════════════════

import hashlib


# ── Forgot-password OTP flow ──────────────────────────────────────────────────

def forgot_password(email: str) -> Optional[Dict]:
    """POST /api/auth/forgot-password → returns message dict."""
    try:
        resp = requests.post(
            f"{API_BASE}/auth/forgot-password",
            json={"email": email}, timeout=15,
        )
        return resp.json()
    except Exception as exc:
        logger.warning("Forgot-password request failed for %s: %s", email, exc)
        _err("⚠️", "Request Error", GENERIC_ERROR_BODY)
        return None


def verify_otp(email: str, otp: str) -> Optional[Dict]:
    """POST /api/auth/verify-otp → returns {reset_token, valid} or error."""
    try:
        resp = requests.post(
            f"{API_BASE}/auth/verify-otp",
            json={"email": email, "otp": otp}, timeout=15,
        )
        return resp.json()
    except Exception as exc:
        logger.warning("OTP verification request failed for %s: %s", email, exc)
        _err("⚠️", "OTP Verify Error", GENERIC_ERROR_BODY)
        return None


def reset_password(reset_token: str, new_password: str, confirm_password: str) -> Optional[Dict]:
    """POST /api/auth/reset-password with Bearer reset_token."""
    try:
        resp = requests.post(
            f"{API_BASE}/auth/reset-password",
            json={"new_password": new_password, "confirm_password": confirm_password},
            headers={"Authorization": f"Bearer {reset_token}"},
            timeout=15,
        )
        return resp.json()
    except Exception as exc:
        logger.warning("Password reset request failed: %s", exc)
        _err("⚠️", "Reset Error", GENERIC_ERROR_BODY)
        return None


# ── Support enquiry ───────────────────────────────────────────────────────────

def submit_enquiry(form_data: Dict, attachment=None) -> Optional[Dict]:
    """POST /api/support/enquiry with multipart form data."""
    try:
        files = {}
        if attachment:
            files["attachment"] = (attachment.name, attachment.getvalue(), attachment.type)
        resp = requests.post(
            f"{API_BASE}/support/enquiry",
            headers=_headers(),
            data=form_data,
            files=files if files else None,
            timeout=20,
        )
        if resp.status_code == 201:
            return resp.json()
        _err("❌", f"Submission Failed ({resp.status_code})", resp.json().get("detail", "Unknown error"))
        return None
    except Exception as exc:
        logger.warning("Support enquiry submission failed: %s", exc)
        _err("⚠️", "Submit Error", GENERIC_ERROR_BODY)
        return None


def get_my_enquiries(page: int = 1) -> List[Dict]:
    """GET /api/support/enquiries → list of user's enquiries."""
    data = safe_request("GET", f"{API_BASE}/support/enquiries?page={page}&limit=20")
    return data or []


def get_faq() -> List[Dict]:
    """GET /api/support/faq → FAQ list."""
    data = safe_request("GET", f"{API_BASE}/support/faq", show_errors=False)
    return data or []


# ── User & team management ────────────────────────────────────────────────────

def get_my_profile() -> Optional[Dict]:
    """GET /api/users/me → extended profile with role, gravatar, activity."""
    return safe_request("GET", f"{API_BASE}/users/me", show_errors=False)


def update_my_profile(updates: Dict) -> bool:
    """PUT /api/users/me."""
    result = safe_request("PUT", f"{API_BASE}/users/me", json=updates)
    return result is not None


def list_users(page: int = 1) -> List[Dict]:
    """GET /api/users/list → all users (admin only)."""
    data = safe_request("GET", f"{API_BASE}/users/list?page={page}&limit=50")
    return data or []


def invite_user(email: str, full_name: str, role: str) -> Optional[Dict]:
    """POST /api/users/invite."""
    return safe_request("POST", f"{API_BASE}/users/invite",
                        json={"email": email, "full_name": full_name, "role": role})


def update_user_role(user_id: int, role: str) -> bool:
    """PUT /api/users/{user_id}/role."""
    result = safe_request("PUT", f"{API_BASE}/users/{user_id}/role", json={"role": role})
    return result is not None


def get_user_activity() -> List[Dict]:
    """GET /api/users/activity."""
    data = safe_request("GET", f"{API_BASE}/users/activity?limit=50", show_errors=False)
    return data or []


def log_page_visit(page: str) -> None:
    """POST /api/users/activity - fire-and-forget page tracking."""
    try:
        requests.post(
            f"{API_BASE}/users/activity",
            json={"action": "page_visit", "page": page},
            headers=_headers(), timeout=3,
        )
    except Exception as exc:
        logger.debug("Page visit log skipped: %s", exc)


# ── Notifications ─────────────────────────────────────────────────────────────

def get_notification_prefs() -> Optional[Dict]:
    """GET /api/notifications/preferences."""
    return safe_request("GET", f"{API_BASE}/notifications/preferences", show_errors=False)


def update_notification_prefs(prefs: Dict) -> bool:
    """PUT /api/notifications/preferences."""
    result = safe_request("PUT", f"{API_BASE}/notifications/preferences", json=prefs)
    return result is not None


def send_test_notification() -> Optional[Dict]:
    """POST /api/notifications/test."""
    return safe_request("POST", f"{API_BASE}/notifications/test")


# ── PADE ML Training ──────────────────────────────────────────────────────────

def start_pade_training(config: Dict) -> Optional[Dict]:
    """POST /api/pade/train/start → {job_id, message, estimated_duration_minutes}."""
    return safe_request("POST", f"{API_BASE}/pade/train/start", json=config)


def get_training_status(job_id: int) -> Optional[Dict]:
    """GET /api/pade/train/status/{job_id}."""
    return safe_request("GET", f"{API_BASE}/pade/train/status/{job_id}", show_errors=False)


def get_training_history() -> List[Dict]:
    """GET /api/pade/train/history."""
    data = safe_request("GET", f"{API_BASE}/pade/train/history?limit=20", show_errors=False)
    return data or []


def get_baseline_report() -> Optional[Dict]:
    """GET /api/pade/train/baseline."""
    return safe_request("GET", f"{API_BASE}/pade/train/baseline", show_errors=False)


# ── Currency exchange (free API, no key required) ─────────────────────────────

@st.cache_data(ttl=3600)
def get_exchange_rates(base: str = "USD") -> Dict:
    """Fetch exchange rates from open.er-api.com (completely free, no key)."""
    try:
        resp = retry_sync(
            lambda: requests.get(f"https://open.er-api.com/v6/latest/{base}", timeout=8),
            attempts=2,
            delay=0.25,
            exceptions=RETRYABLE_REQUEST_EXCEPTIONS,
            logger=logger,
            label=f"exchange-rate fetch {base}",
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("rates", {})
    except Exception as exc:
        logger.warning(f"Exchange rate fetch error: {exc}")
    return {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "INR": 83.5}


# ── Gravatar ──────────────────────────────────────────────────────────────────

def get_gravatar_url(email: str, size: int = 40) -> str:
    """Return Gravatar URL (identicon fallback)."""
    h = hashlib.md5(email.lower().strip().encode()).hexdigest()
    return f"https://www.gravatar.com/avatar/{h}?d=identicon&s={size}"


# ── Pipeline run tags ─────────────────────────────────────────────────────────

def get_pipeline_runs(page: int = 1, limit: int = 50, filters: Optional[Dict] = None) -> List[Dict]:
    """GET /api/alerts/recent with pagination for run history page."""
    data = safe_request(
        "GET", f"{API_BASE}/alerts/recent?limit={limit}", show_errors=False
    )
    return data or []


# -- Post-run integration ------------------------------------------------------

def get_postrun_summary(
    results_root: Optional[str] = None,
    min_ensemble_f1: float = 0.80,
) -> Optional[Dict]:
    query = f"?min_ensemble_f1={min_ensemble_f1:.4f}"
    if results_root:
        from urllib.parse import quote

        query += f"&results_root={quote(results_root)}"
    return safe_request("GET", f"{API_BASE}/postrun/summary{query}")


def run_postrun_import(
    *,
    dry_run: bool = True,
    results_root: Optional[str] = None,
    chunk_size: int = 100_000,
    min_ensemble_f1: float = 0.80,
) -> Optional[Dict]:
    payload: Dict[str, Any] = {
        "dry_run": dry_run,
        "chunk_size": chunk_size,
        "min_ensemble_f1": min_ensemble_f1,
    }
    if results_root:
        payload["results_root"] = results_root
    return safe_request("POST", f"{API_BASE}/postrun/import", json=payload)


def get_postrun_import_history(limit: int = 25) -> Optional[Dict]:
    return safe_request("GET", f"{API_BASE}/postrun/import/history?limit={limit}", show_errors=False)


def get_postrun_model_registry(results_root: Optional[str] = None) -> Optional[Dict]:
    if results_root:
        from urllib.parse import quote

        return safe_request("GET", f"{API_BASE}/postrun/models?results_root={quote(results_root)}", show_errors=False)
    return safe_request("GET", f"{API_BASE}/postrun/models", show_errors=False)


def get_postrun_seed_metrics(
    metric_name: str = "f1_at_opt",
    model_name: str = "ens",
    metric_scope: str = "test",
) -> Optional[Dict]:
    return safe_request(
        "GET",
        f"{API_BASE}/postrun/graphs/seed-metrics?metric_name={metric_name}&model_name={model_name}&metric_scope={metric_scope}",
        show_errors=False,
    )


def get_postrun_domain_metrics(
    metric_name: str = "f1_at_opt",
    model_name: Optional[str] = "ens",
    metric_scope: str = "test",
) -> Optional[Dict]:
    query = f"{API_BASE}/postrun/graphs/domain-metrics?metric_name={metric_name}&metric_scope={metric_scope}"
    if model_name:
        query += f"&model_name={model_name}"
    return safe_request("GET", query, show_errors=False)


def get_postrun_anomaly_counts(
    split_name: str = "test",
    model_name: Optional[str] = "ens",
) -> Optional[Dict]:
    query = f"{API_BASE}/postrun/graphs/anomaly-counts?split_name={split_name}"
    if model_name:
        query += f"&model_name={model_name}"
    return safe_request("GET", query, show_errors=False)


def get_postrun_dataset_summaries() -> Optional[Dict]:
    return safe_request("GET", f"{API_BASE}/postrun/graphs/dataset-summaries", show_errors=False)


def get_continual_readiness(limit: int = 100) -> Optional[Dict]:
    return safe_request("GET", f"{API_BASE}/continual/retraining-readiness?limit={limit}", show_errors=False)


def get_continual_export(status: str = "pending", limit: int = 200) -> Optional[Dict]:
    return safe_request(
        "GET",
        f"{API_BASE}/continual/retraining-export?status={status}&limit={limit}",
        show_errors=False,
    )

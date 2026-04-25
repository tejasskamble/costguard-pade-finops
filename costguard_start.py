#!/usr/bin/env python3
"""CostGuard service launcher for local enterprise development."""
from __future__ import annotations

import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass
from typing import Dict, List, Optional

from costguard_runtime import configure_warning_filters, install_global_exception_hooks, retry_sync

ROOT = pathlib.Path(__file__).parent.resolve()
BACKEND_DIR = ROOT / "backend"
DASHBOARD_DIR = ROOT / "dashboard"
ENV_FILE = ROOT / ".env"
LOG_DIR = ROOT / "logs"
IS_WINDOWS = sys.platform == "win32"
VENV_PYTHON = ROOT / "venv" / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")

def _read_env_defaults() -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not ENV_FILE.exists():
        return values
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip()
    return values


_ENV_DEFAULTS = _read_env_defaults()


def _env_value(name: str, default: str) -> str:
    return os.getenv(name, _ENV_DEFAULTS.get(name, default))


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_value(name, str(default)))
    except (TypeError, ValueError):
        return default


POSTGRES_PORT = _env_int("DB_PORT", 5433)
FASTAPI_PORT = _env_int("API_PORT", 7860)
STREAMLIT_PORT = _env_int("DASHBOARD_PORT", 8501)
FASTAPI_URL = _env_value("API_BASE_URL", f"http://{_env_value('API_HOST', 'localhost')}:{FASTAPI_PORT}").rstrip("/")
STREAMLIT_URL = _env_value(
    "DASHBOARD_BASE_URL",
    f"http://{_env_value('DASHBOARD_HOST', 'localhost')}:{STREAMLIT_PORT}",
).rstrip("/")
HEALTH_URL = f"{FASTAPI_URL}/health"

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

if IS_WINDOWS:
    os.system("")

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("costguard.start")
configure_warning_filters()
install_global_exception_hooks(logger)

TORCH_GEO_WARN = """\
[WARN] torch-geometric was not installed.
       Core platform services still run, but graph-specific PADE features stay degraded.
       Install the dependency later and rerun the launcher if full GAT support is required.
"""


@dataclass
class ManagedProcess:
    """Child process plus its open log handle."""

    name: str
    process: subprocess.Popen[str]
    log_handle: object


_PROCS: List[ManagedProcess] = []
REUSE_EXISTING_STREAMLIT = False


def ok(message: str) -> None:
    print(f"  {GREEN}OK{RESET}  {message}")



def err(message: str) -> None:
    print(f"  {RED}ERR{RESET} {message}")



def warn(message: str) -> None:
    print(f"  {YELLOW}WARN{RESET} {message}")



def info(message: str) -> None:
    print(f"  {CYAN}INFO{RESET} {message}")



def banner() -> None:
    print()
    print(f"{BOLD}{CYAN}============================================================{RESET}")
    print(f"{BOLD}{CYAN} CostGuard v17.0 Launcher{RESET}")
    print(f"{BOLD}{CYAN}============================================================{RESET}")
    print()



def _safe_unlink(path: pathlib.Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except Exception as exc:
        logger.debug("Temporary file cleanup skipped for %s: %s", path, exc)



def _compose_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["COMPOSE_CONVERT_WINDOWS_PATHS"] = "1"
    env.setdefault("VIRTUAL_ENV", str(VENV_PYTHON.parent.parent))
    env["PATH"] = f"{VENV_PYTHON.parent}{os.pathsep}{env.get('PATH', '')}"
    env.setdefault("COSTGUARD_API_BASE_URL", FASTAPI_URL)
    env.setdefault("COSTGUARD_DASHBOARD_BASE_URL", STREAMLIT_URL)
    return env



def _load_env_file(env: Dict[str, str]) -> Dict[str, str]:
    if not ENV_FILE.exists():
        return env
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        env.setdefault(key.strip(), value.strip())
    return env


def ensure_project_venv() -> None:
    if not VENV_PYTHON.exists():
        raise RuntimeError(f"Expected project venv interpreter was not found: {VENV_PYTHON}")
    current_python = pathlib.Path(sys.executable).resolve()
    if current_python != VENV_PYTHON.resolve():
        raise RuntimeError(
            f"Launcher must be run with the project venv interpreter.\n"
            f"Use: {VENV_PYTHON}\n"
            f"Current: {current_python}"
        )



def _run_command(command: List[str], *, cwd: pathlib.Path, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        env=_compose_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )



def check_docker_running() -> bool:
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=15, check=False)
        return result.returncode == 0
    except FileNotFoundError:
        err("Docker was not found in PATH.")
        return False
    except subprocess.TimeoutExpired:
        err("Docker daemon did not respond within 15 seconds.")
        return False



def check_port_clear(port: int, service: str) -> bool:
    global REUSE_EXISTING_STREAMLIT
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        in_use = sock.connect_ex(("localhost", port)) == 0
    if in_use and service == "Streamlit":
        if wait_http(f"{STREAMLIT_URL}/_stcore/health", retries=2, interval=0.5):
            warn(f"Port {port} already has a healthy Streamlit service; reusing existing instance.")
            REUSE_EXISTING_STREAMLIT = True
            return True
        err(f"Port {port} is already in use for {service}.")
        return False
    if in_use and service != "Postgres":
        err(f"Port {port} is already in use for {service}.")
        return False
    return True



def deps_ok() -> bool:
    try:
        import fastapi  # noqa: F401
        import asyncpg  # noqa: F401
        import streamlit  # noqa: F401

        return True
    except ImportError as exc:
        logger.info("Dependency check failed: %s", exc)
        return False



def install_requirements() -> bool:
    requirements_file = BACKEND_DIR / "requirements.txt"
    if not requirements_file.exists():
        err(f"Missing requirements file: {requirements_file}")
        return False
    lines = requirements_file.read_text(encoding="utf-8").splitlines()
    base_packages = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
        and "torch-geometric" not in line
        and "torch-scatter" not in line
        and "torch-sparse" not in line
    ]
    temp_requirements = BACKEND_DIR / "_req_base_tmp.txt"
    temp_requirements.write_text("\n".join(base_packages), encoding="utf-8")
    try:
        base_install = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(temp_requirements), "--quiet"],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        _safe_unlink(temp_requirements)
    if base_install.returncode != 0:
        err("Dependency installation failed.")
        print(base_install.stderr[:800])
        return False
    pyg_install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "torch-geometric", "--quiet"],
        capture_output=True,
        text=True,
        check=False,
    )
    if pyg_install.returncode != 0:
        warn(TORCH_GEO_WARN.strip())
    else:
        ok("torch-geometric installed.")
    return True



def wait_for_port(port: int, *, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            if sock.connect_ex(("localhost", port)) == 0:
                return True
        time.sleep(1)
    return False



def start_postgres() -> bool:
    compose_file = ROOT / "docker-compose.yml"
    if not compose_file.exists():
        warn("docker-compose.yml not found; assuming an external Postgres instance.")
        return wait_for_port(POSTGRES_PORT, timeout=5)
    _run_command(["docker", "compose", "rm", "-f", "postgres"], cwd=ROOT, timeout=30)
    info("Pulling postgres image if needed...")
    _run_command(["docker", "compose", "pull", "postgres", "--quiet"], cwd=ROOT, timeout=120)
    result = _run_command(["docker", "compose", "up", "-d", "--wait", "postgres"], cwd=ROOT, timeout=120)
    if result.returncode != 0:
        err("docker compose up failed.")
        print((result.stderr or result.stdout)[:800])
        return False
    return wait_for_port(POSTGRES_PORT, timeout=10)



def wait_http(url: str, *, retries: int = 35, interval: float = 2.0) -> bool:
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=4) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.debug("Health probe %d/%d failed for %s: %s", attempt, retries, url, exc)
        time.sleep(interval)
    return False



def _open_log(path: pathlib.Path):
    LOG_DIR.mkdir(exist_ok=True)
    return open(path, "w", encoding="utf-8")



def start_fastapi(dev_mode: bool = False) -> ManagedProcess:
    env = _load_env_file(os.environ.copy())
    env["PYTHONPATH"] = str(BACKEND_DIR)
    command = [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", str(FASTAPI_PORT), "--log-level", "info"]
    if dev_mode and not IS_WINDOWS:
        command.append("--reload")
    log_handle = _open_log(LOG_DIR / "backend.log")
    process = subprocess.Popen(command, cwd=str(BACKEND_DIR), env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    return ManagedProcess(name="FastAPI", process=process, log_handle=log_handle)



def start_streamlit() -> ManagedProcess:
    log_handle = _open_log(LOG_DIR / "dashboard.log")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        str(STREAMLIT_PORT),
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    process = subprocess.Popen(command, cwd=str(DASHBOARD_DIR), stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    return ManagedProcess(name="Streamlit", process=process, log_handle=log_handle)



def shutdown() -> None:
    print()
    print(f"{BOLD}{YELLOW}[SHUTDOWN]{RESET} Stopping managed services...")
    for managed in reversed(_PROCS):
        if managed.process.poll() is None:
            try:
                managed.process.terminate()
                managed.process.wait(timeout=6)
                ok(f"{managed.name} stopped.")
            except Exception as exc:
                logger.warning("Graceful shutdown failed for %s: %s", managed.name, exc)
                try:
                    managed.process.kill()
                except Exception as kill_exc:
                    logger.warning("Force kill failed for %s: %s", managed.name, kill_exc)
                warn(f"{managed.name} required force termination.")
        try:
            managed.log_handle.close()
        except Exception as exc:
            logger.debug("Log handle close skipped for %s: %s", managed.name, exc)



def _handle_signal(_signum=None, _frame=None) -> None:
    shutdown()
    sys.exit(0)



def main() -> None:
    global REUSE_EXISTING_STREAMLIT
    banner()
    ensure_project_venv()
    no_browser = "--no-browser" in sys.argv
    dev_mode = "--dev" in sys.argv

    signal.signal(signal.SIGINT, _handle_signal)
    if not IS_WINDOWS:
        signal.signal(signal.SIGTERM, _handle_signal)

    print(f"{BOLD}Pre-flight checks:{RESET}")
    checks = [
        ("Docker Desktop", check_docker_running),
        (f"Port {POSTGRES_PORT} (Postgres)", lambda: check_port_clear(POSTGRES_PORT, "Postgres")),
        (f"Port {FASTAPI_PORT} (FastAPI)", lambda: check_port_clear(FASTAPI_PORT, "FastAPI")),
        (f"Port {STREAMLIT_PORT} (Streamlit)", lambda: check_port_clear(STREAMLIT_PORT, "Streamlit")),
    ]
    failures = []
    for label, probe in checks:
        result = probe()
        ok(label) if result else err(label)
        if not result and "Port" in label:
            failures.append(label)
    if failures:
        raise RuntimeError(f"Pre-flight failed for: {', '.join(failures)}")

    print()
    print(f"{BOLD}[1/4]{RESET} Starting PostgreSQL...", end=" ", flush=True)
    if not start_postgres():
        raise RuntimeError("PostgreSQL did not become ready.")
    print(f"{GREEN}OK{RESET}")

    print(f"{BOLD}[2/4]{RESET} Verifying dependencies...", end=" ", flush=True)
    if not deps_ok() and not install_requirements():
        raise RuntimeError("Dependency installation failed.")
    print(f"{GREEN}OK{RESET}")

    print(f"{BOLD}[3/4]{RESET} Starting FastAPI backend...", end=" ", flush=True)
    api_process = retry_sync(lambda: start_fastapi(dev_mode=dev_mode), attempts=2, delay=0.5, logger=logger, label="FastAPI process start")
    _PROCS.append(api_process)
    if not wait_http(HEALTH_URL, retries=35, interval=2.0):
        raise RuntimeError(f"FastAPI health check failed: {HEALTH_URL}")
    print(f"{GREEN}OK{RESET}")

    if REUSE_EXISTING_STREAMLIT:
        print(f"{BOLD}[4/4]{RESET} Reusing dashboard on {STREAMLIT_URL}...", end=" ", flush=True)
        if not wait_http(f"{STREAMLIT_URL}/_stcore/health", retries=5, interval=0.5):
            raise RuntimeError(f"Existing Streamlit on port {STREAMLIT_PORT} is not healthy.")
    else:
        print(f"{BOLD}[4/4]{RESET} Starting dashboard...", end=" ", flush=True)
        dashboard_process = retry_sync(start_streamlit, attempts=2, delay=0.5, logger=logger, label="Streamlit process start")
        _PROCS.append(dashboard_process)
        if not wait_http(STREAMLIT_URL, retries=20, interval=2.0):
            warn("Dashboard health probe is still warming up; continuing.")
    print(f"{GREEN}OK{RESET}")

    if not no_browser:
        time.sleep(1)
        webbrowser.open(STREAMLIT_URL)

    print()
    print(f"{BOLD}{CYAN}CostGuard is running.{RESET}")
    print(f"Dashboard : {STREAMLIT_URL}")
    print(f"API Docs  : {FASTAPI_URL}/docs")
    print(f"Health    : {HEALTH_URL}")
    print(f"Logs      : {LOG_DIR}")
    print(f"{YELLOW}Press Ctrl+C to stop services gracefully.{RESET}")

    while True:
        time.sleep(5)
        for managed in _PROCS:
            if managed.process.poll() is not None:
                raise RuntimeError(f"{managed.name} exited unexpectedly with code {managed.process.returncode}.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _handle_signal()
    except Exception as exc:
        logger.exception("Launcher failed: %s", exc)
        shutdown()
        err(f"Launcher failed: {exc}")
        sys.exit(1)

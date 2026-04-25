"""Central runtime configuration for CostGuard."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = str(Path(__file__).parent.parent / ".env")
_PLACEHOLDER_VALUES = {
    "",
    "changeme",
    "change-me",
    "change_me",
    "your-secret-key-change-this",
    "change_me_db_password",
    "change_me_jwt_secret",
    "change_me_grafana_password",
    "your-openai-api-key",
    "your-slack-bot-token",
    "your-app-password",
    "your-email@gmail.com",
}
_NON_PROD_ENVIRONMENTS = {"local", "development", "dev", "test"}


def _looks_like_placeholder(value: Optional[str]) -> bool:
    if value is None:
        return True
    normalized = value.strip().lower()
    return normalized in _PLACEHOLDER_VALUES or "change-this" in normalized


def _normalize_http_url(value: str) -> str:
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL: {value}")
    return value.rstrip("/")


class Settings(BaseSettings):
    ENVIRONMENT: str = "local"

    DB_HOST: str = "localhost"
    DB_PORT: int = 5433
    DB_NAME: str = "costguard"
    DB_USER: str = "tejuska_user"
    DB_PASSWORD: str = "CHANGE_ME_DB_PASSWORD"
    DB_MIN_CONN: int = 2
    DB_MAX_CONN: int = 20

    JWT_SECRET: str = "CHANGE_ME_JWT_SECRET"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_MINUTES: int = 60

    API_HOST: str = "localhost"
    API_PORT: int = 7860
    API_BASE_URL: Optional[str] = None
    DASHBOARD_HOST: str = "localhost"
    DASHBOARD_PORT: int = 8501
    DASHBOARD_BASE_URL: Optional[str] = None
    ALLOWED_ORIGINS: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:8501",
            "http://localhost:3000",
            "http://127.0.0.1:8501",
            "http://localhost:7860",
        ]
    )

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_MAX_TOKENS: int = 1000

    SLACK_BOT_TOKEN: Optional[str] = None
    SLACK_DEFAULT_CHANNEL: str = "#costguard-alerts"

    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_FROM: Optional[str] = None
    SUPPORT_EMAIL: Optional[str] = None

    PROMETHEUS_ENABLED: bool = True
    METRICS_PORT: int = 9090
    GRAFANA_ADMIN_PASSWORD: str = "CHANGE_ME_GRAFANA_PASSWORD"

    AWS_DEFAULT_REGION: str = "ap-south-1"
    GCP_DEFAULT_PROJECT: str = "costguard-cloud-intel"

    OTP_EXPIRY_MINUTES: int = 10
    RESET_TOKEN_EXPIRY_MINUTES: int = 15
    OTP_RATE_LIMIT_PER_HOUR: int = 3
    OTP_MAX_ATTEMPTS: int = 5

    RATE_LIMIT_PER_MINUTE: int = 100

    KAFKA_ENABLED: bool = False
    KAFKA_BROKERS: str = "kafka:9092"

    OPA_URL: str = "http://localhost:8181/v1/data/costguard/result"
    OPA_POSTRUN_URL: Optional[str] = "http://localhost:8181/v1/data/costguard/postrun_result"
    PADE_DATA_MODE: str = "synthetic"
    POSTRUN_RESULTS_ROOT: str = "results"
    POSTRUN_IMPORT_CHUNK_SIZE: int = 100_000
    POSTRUN_MIN_ENSEMBLE_F1: float = 0.80

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("ENVIRONMENT")
    @classmethod
    def _normalize_environment(cls, value: str) -> str:
        return (value or "local").strip().lower()

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def _parse_allowed_origins(cls, value: Any) -> Any:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [str(item).strip().rstrip("/") for item in value if str(item).strip()]
        if isinstance(value, str):
            raw = value.strip()
            if raw.startswith("["):
                parsed = json.loads(raw)
                return [str(item).strip().rstrip("/") for item in parsed if str(item).strip()]
            return [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
        return value

    @field_validator("API_BASE_URL", "DASHBOARD_BASE_URL", "OPA_URL", "OPA_POSTRUN_URL", mode="before")
    @classmethod
    def _normalize_urls(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        raw = str(value).strip()
        if not raw:
            return None
        return raw.rstrip("/")

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def api_http_base(self) -> str:
        if self.API_BASE_URL:
            return _normalize_http_url(self.API_BASE_URL)
        return f"http://{self.API_HOST}:{self.API_PORT}"

    @property
    def dashboard_http_base(self) -> str:
        if self.DASHBOARD_BASE_URL:
            return _normalize_http_url(self.DASHBOARD_BASE_URL)
        return f"http://{self.DASHBOARD_HOST}:{self.DASHBOARD_PORT}"

    @property
    def support_email(self) -> str:
        return self.SUPPORT_EMAIL or self.SMTP_USER or "support@costguard.local"

    def validate_runtime_requirements(self) -> None:
        if not self.DB_PASSWORD.strip():
            raise RuntimeError("DB_PASSWORD is required.")
        if not self.JWT_SECRET.strip():
            raise RuntimeError("JWT_SECRET is required.")
        if not self.ALLOWED_ORIGINS:
            raise RuntimeError("ALLOWED_ORIGINS must contain at least one origin.")

        _normalize_http_url(self.api_http_base)
        _normalize_http_url(self.dashboard_http_base)
        _normalize_http_url(self.OPA_URL)
        if self.OPA_POSTRUN_URL:
            _normalize_http_url(self.OPA_POSTRUN_URL)

        if self.ENVIRONMENT in _NON_PROD_ENVIRONMENTS:
            return

        if _looks_like_placeholder(self.DB_PASSWORD):
            raise RuntimeError("DB_PASSWORD must be set to a real secret outside local/test environments.")
        if _looks_like_placeholder(self.JWT_SECRET):
            raise RuntimeError("JWT_SECRET must be set to a real secret outside local/test environments.")
        if _looks_like_placeholder(self.GRAFANA_ADMIN_PASSWORD):
            raise RuntimeError("GRAFANA_ADMIN_PASSWORD must be set to a real secret outside local/test environments.")

        local_markers = ("localhost", "127.0.0.1")
        if any(marker in self.api_http_base for marker in local_markers):
            raise RuntimeError("API base URL must not point to localhost outside local/test environments.")
        if any(marker in self.dashboard_http_base for marker in local_markers):
            raise RuntimeError("Dashboard base URL must not point to localhost outside local/test environments.")
        if any(any(marker in origin for marker in local_markers) for origin in self.ALLOWED_ORIGINS):
            raise RuntimeError("ALLOWED_ORIGINS must not include localhost outside local/test environments.")


settings = Settings()

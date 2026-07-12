from __future__ import annotations

import os
from pathlib import Path

try:
    import streamlit as st
except ImportError:  # pragma: no cover - streamlit is optional outside the app
    st = None


PRODUCTION_ENV_VALUES = {"prod", "production", "live"}


def get_app_env() -> str:
    raw_value = os.getenv("APP_ENV") or os.getenv("ENV") or "development"
    return raw_value.strip().lower()


def is_production() -> bool:
    return get_app_env() in PRODUCTION_ENV_VALUES


def is_debug_enabled() -> bool:
    explicit = _read_bool(os.getenv("APP_DEBUG"))
    if explicit is not None:
        return explicit

    if is_production():
        return False

    return _read_bool(os.getenv("SHOW_DEBUG"), default=True)


def load_secret(
    name: str,
    *,
    fallback_names: tuple[str, ...] = (),
    required: bool = True,
    allow_dotenv: bool | None = None,
) -> str:
    candidates = _unique_names((name, *fallback_names))

    for candidate in candidates:
        value = _read_streamlit_secret(candidate)
        if value:
            return value

    for candidate in candidates:
        value = _read_environment_variable(candidate)
        if value:
            return value

    if allow_dotenv is None:
        allow_dotenv = not is_production()

    if allow_dotenv:
        for candidate in candidates:
            value = _read_dotenv_value(candidate)
            if value:
                return value

    if required:
        raise RuntimeError(
            f"{name} chua duoc cau hinh. Hay cung cap qua Streamlit Secrets hoac bien moi truong."
        )
    return ""


def load_tavily_api_key() -> str:
    return load_secret(
        "TAVILY_API_KEY",
        fallback_names=("tavily_api_key",),
        required=False,
    )


def _read_streamlit_secret(name: str) -> str:
    if st is None:
        return ""
    try:
        value = st.secrets.get(name, "")
    except Exception:
        return ""
    return str(value).strip().strip('"').strip("'")


def _read_environment_variable(name: str) -> str:
    return os.getenv(name, "").strip().strip('"').strip("'")


def _read_dotenv_value(name: str) -> str:
    env_path = Path(".env")
    if not env_path.exists():
        return ""

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip().strip('"').strip("'")
    return ""


def _unique_names(names: tuple[str, ...]) -> tuple[str, ...]:
    ordered: list[str] = []
    for name in names:
        cleaned = name.strip()
        if cleaned and cleaned not in ordered:
            ordered.append(cleaned)
    return tuple(ordered)


def _read_bool(value: str | None, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default

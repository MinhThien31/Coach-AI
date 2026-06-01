"""Env-backed configuration for the API. Read once at startup."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"
DEFAULT_CORS_ORIGINS = (
    "https://minhthien.io.vn,"
    "https://www.minhthien.io.vn,"
    "http://localhost:3000,"
    "http://localhost:5173"
)


def _read_dotenv(path: str | os.PathLike[str] | None) -> dict[str, str]:
    """Read simple KEY=VALUE pairs from a local .env file."""
    if path is None:
        return {}

    dotenv_path = Path(path)
    if not dotenv_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        quote = value[:1]
        if quote in {'"', "'"} and value.endswith(quote):
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        values[key] = value
    return values


def _config_value(name: str, dotenv: dict[str, str], default: str) -> str:
    value = os.getenv(name)
    if value is not None:
        return value
    return dotenv.get(name, default)


def _parse_cors_origins(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()

    origins = tuple(o.strip() for o in value.split(",") if o.strip())
    if "*" in origins:
        return origins

    required = tuple(o.strip() for o in DEFAULT_CORS_ORIGINS.split(",") if o.strip())
    return tuple(dict.fromkeys((*origins, *required)))


@dataclass(frozen=True)
class Settings:
    max_upload_mb: int
    max_video_seconds: int
    cors_origins: tuple[str, ...]
    nvidia_api_key: str | None
    nvidia_nim_model: str

    @classmethod
    def from_env(
        cls,
        dotenv_path: str | os.PathLike[str] | None = DEFAULT_DOTENV_PATH,
    ) -> "Settings":
        dotenv = _read_dotenv(dotenv_path)
        cors_origins = _config_value("API_CORS_ORIGINS", dotenv, DEFAULT_CORS_ORIGINS)
        nvidia_api_key = _config_value("NVIDIA_API_KEY", dotenv, "")
        nvidia_nim_model = _config_value(
            "NVIDIA_NIM_MODEL",
            dotenv,
            "qwen/qwen3-next-80b-a3b-instruct",
        )
        return cls(
            max_upload_mb=int(_config_value("API_MAX_UPLOAD_MB", dotenv, "100")),
            max_video_seconds=int(_config_value("API_MAX_VIDEO_SECONDS", dotenv, "60")),
            cors_origins=_parse_cors_origins(cors_origins),
            nvidia_api_key=nvidia_api_key or None,
            nvidia_nim_model=nvidia_nim_model,
        )

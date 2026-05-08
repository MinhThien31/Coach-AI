"""Env-backed configuration for the API. Read once at startup."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    max_upload_mb: int
    max_video_seconds: int
    cors_origins: list[str]
    nvidia_api_key: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            max_upload_mb=int(os.getenv("API_MAX_UPLOAD_MB", "100")),
            max_video_seconds=int(os.getenv("API_MAX_VIDEO_SECONDS", "60")),
            cors_origins=[
                o.strip()
                for o in os.getenv("API_CORS_ORIGINS", "*").split(",")
                if o.strip()
            ],
            nvidia_api_key=os.getenv("NVIDIA_API_KEY") or None,
        )

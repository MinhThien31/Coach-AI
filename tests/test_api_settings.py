"""Tests for api.settings env-backed config."""
import os

import pytest

from api.settings import Settings


def test_defaults_when_no_env(monkeypatch):
    for var in ("API_MAX_UPLOAD_MB", "API_MAX_VIDEO_SECONDS",
                "API_CORS_ORIGINS", "NVIDIA_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    s = Settings.from_env()
    assert s.max_upload_mb == 100
    assert s.max_video_seconds == 60
    assert s.cors_origins == ["*"]
    assert s.nvidia_api_key is None


def test_overrides_from_env(monkeypatch):
    monkeypatch.setenv("API_MAX_UPLOAD_MB", "25")
    monkeypatch.setenv("API_MAX_VIDEO_SECONDS", "30")
    monkeypatch.setenv("API_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    s = Settings.from_env()
    assert s.max_upload_mb == 25
    assert s.max_video_seconds == 30
    assert s.cors_origins == ["http://localhost:3000", "http://localhost:5173"]
    assert s.nvidia_api_key == "nvapi-test"


def test_invalid_int_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("API_MAX_UPLOAD_MB", "not-a-number")
    with pytest.raises(ValueError):
        Settings.from_env()

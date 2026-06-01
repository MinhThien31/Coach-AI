"""Tests for api.settings env-backed config."""
import os

import pytest

from api.settings import DEFAULT_CORS_ORIGINS, Settings


def test_defaults_when_no_env(monkeypatch):
    for var in ("API_MAX_UPLOAD_MB", "API_MAX_VIDEO_SECONDS",
                "API_CORS_ORIGINS", "NVIDIA_API_KEY", "NVIDIA_NIM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    s = Settings.from_env(dotenv_path=None)
    assert s.max_upload_mb == 100
    assert s.max_video_seconds == 60
    assert s.cors_origins == tuple(DEFAULT_CORS_ORIGINS.split(","))
    assert s.nvidia_api_key is None
    assert s.nvidia_nim_model == "qwen/qwen3-next-80b-a3b-instruct"


def test_overrides_from_env(monkeypatch):
    monkeypatch.setenv("API_MAX_UPLOAD_MB", "25")
    monkeypatch.setenv("API_MAX_VIDEO_SECONDS", "30")
    monkeypatch.setenv("API_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("NVIDIA_NIM_MODEL", "meta/llama-3.3-70b-instruct")
    s = Settings.from_env(dotenv_path=None)
    assert s.max_upload_mb == 25
    assert s.max_video_seconds == 30
    assert s.cors_origins[:2] == ("http://localhost:3000", "http://localhost:5173")
    assert "https://www.minhthien.io.vn" in s.cors_origins
    assert s.nvidia_api_key == "nvapi-test"
    assert s.nvidia_nim_model == "meta/llama-3.3-70b-instruct"


def test_invalid_int_raises(monkeypatch):
    monkeypatch.setenv("API_MAX_UPLOAD_MB", "not-a-number")
    with pytest.raises(ValueError):
        Settings.from_env(dotenv_path=None)


def test_empty_cors_origins_env(monkeypatch):
    monkeypatch.setenv("API_CORS_ORIGINS", "")
    s = Settings.from_env(dotenv_path=None)
    assert s.cors_origins == ()


def test_loads_dotenv_when_env_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("API_MAX_UPLOAD_MB", raising=False)
    monkeypatch.delenv("API_MAX_VIDEO_SECONDS", raising=False)
    monkeypatch.delenv("API_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_NIM_MODEL", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join([
            "API_MAX_UPLOAD_MB=55",
            "API_MAX_VIDEO_SECONDS=45",
            "API_CORS_ORIGINS=http://localhost:8080",
            "NVIDIA_API_KEY=nvapi-from-dotenv",
            "NVIDIA_NIM_MODEL=deepseek-ai/deepseek-r1",
        ]),
        encoding="utf-8",
    )

    s = Settings.from_env(dotenv_path=dotenv)

    assert s.max_upload_mb == 55
    assert s.max_video_seconds == 45
    assert s.cors_origins[0] == "http://localhost:8080"
    assert "https://www.minhthien.io.vn" in s.cors_origins
    assert s.nvidia_api_key == "nvapi-from-dotenv"
    assert s.nvidia_nim_model == "deepseek-ai/deepseek-r1"


def test_env_overrides_dotenv(monkeypatch, tmp_path):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-from-env")
    monkeypatch.setenv("NVIDIA_NIM_MODEL", "model-from-env")
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join([
            "NVIDIA_API_KEY=nvapi-from-dotenv",
            "NVIDIA_NIM_MODEL=model-from-dotenv",
        ]),
        encoding="utf-8",
    )

    s = Settings.from_env(dotenv_path=dotenv)

    assert s.nvidia_api_key == "nvapi-from-env"
    assert s.nvidia_nim_model == "model-from-env"

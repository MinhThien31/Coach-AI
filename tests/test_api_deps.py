"""Tests for the FastAPI dependency providers."""
import asyncio

from fastapi import FastAPI, Request

from api.deps import (
    get_default_analyzer,
    get_enriched_analyzer,
    get_lock,
    get_settings,
)
from api.settings import Settings


def test_providers_read_from_app_state():
    app = FastAPI()
    app.state.settings = Settings.from_env()
    app.state.analyzer_default = "DEF"
    app.state.analyzer_enriched = "ENR"
    app.state.lock = asyncio.Lock()

    fake_request = Request({"type": "http", "app": app})

    assert get_settings(fake_request) is app.state.settings
    assert get_default_analyzer(fake_request) is app.state.analyzer_default
    assert get_enriched_analyzer(fake_request) is app.state.analyzer_enriched
    assert get_lock(fake_request) is app.state.lock


def test_enriched_analyzer_returns_none_when_unset():
    app = FastAPI()
    app.state.analyzer_enriched = None
    fake_request = Request({"type": "http", "app": app})
    assert get_enriched_analyzer(fake_request) is None

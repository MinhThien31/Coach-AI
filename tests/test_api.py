"""Fast, mocked-analyzer unit tests for the HTTP API.

Lifespan is intentionally NOT triggered (no `with TestClient(app):`) so
MediaPipe is never loaded. Each test pre-populates app.state with the
fakes it needs.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.settings import Settings


@pytest.fixture
def client() -> TestClient:
    app.state.settings = Settings.from_env()
    app.state.analyzer_default = MagicMock()
    app.state.analyzer_enriched = None
    app.state.lock = asyncio.Lock()
    return TestClient(app)


def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

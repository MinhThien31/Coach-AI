"""FastAPI dependency providers. Tests override these to inject fakes."""
from __future__ import annotations

import asyncio

from fastapi import Request

from sport_companion_ai import VideoAnalyzer

from api.settings import Settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_default_analyzer(request: Request) -> VideoAnalyzer:
    return request.app.state.analyzer_default


def get_enriched_analyzer(request: Request) -> VideoAnalyzer | None:
    return request.app.state.analyzer_enriched


def get_lock(request: Request) -> asyncio.Lock:
    return request.app.state.lock

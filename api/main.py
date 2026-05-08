"""FastAPI app entrypoint. Configured via env vars (see api.settings)."""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from sport_companion_ai import VideoAnalyzer
from sport_companion_ai.feedback.nim import NvidiaNimEnricher
from sport_companion_ai.feedback.template import TemplateEnricher
from sport_companion_ai.pose.extractor import MediaPipeExtractor

from api.errors import register_exception_handlers
from api.routes import router as api_router
from api.settings import Settings

log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    extractor = MediaPipeExtractor()
    app.state.settings = settings
    app.state.analyzer_default = VideoAnalyzer(
        pose_extractor=extractor, enricher=TemplateEnricher(),
    )
    if settings.nvidia_api_key:
        app.state.analyzer_enriched = VideoAnalyzer(
            pose_extractor=extractor,
            enricher=NvidiaNimEnricher(api_key=settings.nvidia_api_key),
        )
    else:
        app.state.analyzer_enriched = None
    app.state.lock = asyncio.Lock()
    log.info("api ready (enrichment=%s)", bool(app.state.analyzer_enriched))
    yield


app = FastAPI(
    title="Sport Companion AI API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — origins read from env. Default `*` for local dev; tighten in prod.
_origins_at_import = Settings.from_env().cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins_at_import,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.include_router(api_router)


@app.middleware("http")
async def request_logger(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    log.info(
        "%s %s -> %d (%.0fms)",
        request.method, request.url.path, response.status_code, elapsed_ms,
    )
    return response


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}

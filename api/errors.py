"""Map Phase 1 exceptions to HTTP responses with stable error codes."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from sport_companion_ai.errors import (
    PoseExtractionError,
    UnsupportedExerciseError,
    VideoReadError,
)

log = logging.getLogger("api.errors")


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(UnsupportedExerciseError)
    async def _unsupported(request: Request, exc: UnsupportedExerciseError):
        return JSONResponse(
            status_code=400,
            content={"error": "unsupported_exercise", "detail": str(exc)},
        )

    @app.exception_handler(VideoReadError)
    async def _video_read(request: Request, exc: VideoReadError):
        return JSONResponse(
            status_code=400,
            content={"error": "video_read_failed", "detail": str(exc)},
        )

    @app.exception_handler(PoseExtractionError)
    async def _pose(request: Request, exc: PoseExtractionError):
        log.exception("pose extraction failed")
        return JSONResponse(
            status_code=500,
            content={"error": "pose_extraction_failed"},
        )

    @app.exception_handler(Exception)
    async def _generic(request: Request, exc: Exception):
        log.exception("unhandled exception")
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error"},
        )

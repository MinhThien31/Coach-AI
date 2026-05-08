"""HTTP route handlers for /analyze, /exercises, etc."""
from __future__ import annotations

import asyncio
import os
import tempfile

import cv2
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

# Import side-effect: registers all rule classes into EXERCISE_REGISTRY.
import sport_companion_ai.exercises  # noqa: F401
from sport_companion_ai import VideoAnalyzer
from sport_companion_ai.exercises.base import EXERCISE_REGISTRY
from sport_companion_ai.report import AnalysisReport, AnalysisWarning

from api.deps import (
    get_default_analyzer,
    get_enriched_analyzer,
    get_lock,
    get_settings,
)
from api.schemas import SkeletonOutputLiteral
from api.settings import Settings

router = APIRouter()


@router.get("/exercises", tags=["meta"])
async def list_exercises() -> dict[str, list[str]]:
    return {"exercises": sorted(EXERCISE_REGISTRY.keys())}


def _stream_to_temp(upload: UploadFile, max_bytes: int) -> str | None:
    """Stream `upload` into a NamedTemporaryFile in 1 MiB chunks. Return the
    temp file path on success, or `None` if the upload exceeds `max_bytes`
    (in which case the temp file is unlinked before returning)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    written = 0
    try:
        while chunk := upload.file.read(1 << 20):
            written += len(chunk)
            if written > max_bytes:
                tmp.close()
                os.unlink(tmp.name)
                return None
            tmp.write(chunk)
    finally:
        tmp.close()
    return tmp.name


def _video_duration_seconds(path: str) -> float:
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        return (n / fps) if fps > 0 else 0.0
    finally:
        cap.release()


@router.post("/analyze", tags=["analysis"], response_model=AnalysisReport)
async def analyze(
    video: UploadFile = File(...),
    exercise: str = Form(...),
    skeleton_output: SkeletonOutputLiteral = Form("keyframes"),
    enrich: bool = Form(False),
    settings: Settings = Depends(get_settings),
    analyzer_default: VideoAnalyzer = Depends(get_default_analyzer),
    analyzer_enriched: VideoAnalyzer | None = Depends(get_enriched_analyzer),
    lock: asyncio.Lock = Depends(get_lock),
) -> AnalysisReport:
    max_bytes = settings.max_upload_mb * (1 << 20)
    tmp_path = _stream_to_temp(video, max_bytes)
    if tmp_path is None:
        return JSONResponse(
            status_code=413,
            content={"error": "video_too_large",
                     "detail": f"max {settings.max_upload_mb} MB"},
        )
    try:
        duration_s = _video_duration_seconds(tmp_path)
        if duration_s > settings.max_video_seconds:
            return JSONResponse(
                status_code=413,
                content={"error": "video_too_long",
                         "detail": f"max {settings.max_video_seconds}s"},
            )

        # Pick the right analyzer. If enrichment was requested but the server
        # has no key, fall back to default and inject ENRICHMENT_FAILED so the
        # contract matches the in-process no-key path.
        manual_warning: AnalysisWarning | None = None
        if enrich and analyzer_enriched is not None:
            chosen = analyzer_enriched
        else:
            chosen = analyzer_default
            if enrich and analyzer_enriched is None:
                manual_warning = AnalysisWarning(
                    code="ENRICHMENT_FAILED",
                    message_vi="Server không cấu hình NVIDIA_API_KEY",
                )

        async with lock:
            report = await run_in_threadpool(
                chosen.analyze,
                tmp_path,
                exercise=exercise,
                skeleton_output=skeleton_output,
            )

        if manual_warning is not None:
            report.warnings.append(manual_warning)
        return report
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass

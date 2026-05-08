"""Fast, mocked-analyzer unit tests for the HTTP API.

Lifespan is intentionally NOT triggered (no `with TestClient(app):`) so
MediaPipe is never loaded. Each test pre-populates app.state with the
fakes it needs.
"""
from __future__ import annotations

import asyncio
import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.settings import Settings
from sport_companion_ai.errors import UnsupportedExerciseError, VideoReadError
from sport_companion_ai.report import (
    AnalysisReport, SkeletonSchema, VideoMeta,
)


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


def test_exercises_lists_all_registered_rules(client):
    r = client.get("/exercises")
    assert r.status_code == 200
    data = r.json()
    assert set(data["exercises"]) == {
        "squat", "deadlift", "bench_press", "push_up", "bicep_curl",
    }


def _fake_report(exercise: str = "squat") -> AnalysisReport:
    return AnalysisReport(
        exercise=exercise,
        pose_model="mediapipe-blazepose-full",
        video=VideoMeta(width=720, height=1280, fps=30, duration_ms=5000),
        skeleton_schema=SkeletonSchema(keypoint_names=[], edges=[]),
        total_reps=3,
        passed_reps=2,
        avg_score=72.0,
    )


def test_analyze_happy_path_returns_report_json(client):
    fake_report = _fake_report()
    client.app.state.analyzer_default.analyze.return_value = fake_report

    files = {"video": ("squat.mp4", io.BytesIO(b"\x00\x00\x00 ftypisom" + b"x" * 1024), "video/mp4")}
    data = {"exercise": "squat"}
    r = client.post("/analyze", files=files, data=data)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["exercise"] == "squat"
    assert body["total_reps"] == 3
    assert body["passed_reps"] == 2
    assert body["avg_score"] == 72.0
    # analyzer.analyze was called with a path string + the exercise name
    args, kwargs = client.app.state.analyzer_default.analyze.call_args
    assert kwargs["exercise"] == "squat"
    assert isinstance(args[0], str) and args[0].endswith(".mp4")
    # the temp file should have been cleaned up
    assert not Path(args[0]).exists()


def test_analyze_missing_exercise_returns_422(client):
    files = {"video": ("squat.mp4", io.BytesIO(b"x" * 1024), "video/mp4")}
    r = client.post("/analyze", files=files)  # no `data`
    assert r.status_code == 422


def test_analyze_unknown_exercise_returns_400(client):
    client.app.state.analyzer_default.analyze.side_effect = UnsupportedExerciseError(
        "Unknown exercise: 'flying'",
    )
    files = {"video": ("x.mp4", io.BytesIO(b"x" * 1024), "video/mp4")}
    data = {"exercise": "flying"}
    r = client.post("/analyze", files=files, data=data)
    assert r.status_code == 400
    assert r.json()["error"] == "unsupported_exercise"


def test_analyze_corrupt_video_returns_400(client):
    client.app.state.analyzer_default.analyze.side_effect = VideoReadError(
        "could not open video",
    )
    files = {"video": ("x.mp4", io.BytesIO(b"x" * 1024), "video/mp4")}
    data = {"exercise": "squat"}
    r = client.post("/analyze", files=files, data=data)
    assert r.status_code == 400
    assert r.json()["error"] == "video_read_failed"


def test_analyze_invalid_skeleton_output_returns_422(client):
    files = {"video": ("x.mp4", io.BytesIO(b"x" * 1024), "video/mp4")}
    data = {"exercise": "squat", "skeleton_output": "bogus"}
    r = client.post("/analyze", files=files, data=data)
    assert r.status_code == 422

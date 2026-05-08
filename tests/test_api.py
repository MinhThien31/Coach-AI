"""Fast, mocked-analyzer unit tests for the HTTP API.

Lifespan is intentionally NOT triggered (no `with TestClient(app):`) so
MediaPipe is never loaded. Each test pre-populates app.state with the
fakes it needs.
"""
from __future__ import annotations

import asyncio
import io
import threading
import time as _time
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def test_analyze_too_large_returns_413(client):
    # Settings default is 100 MB — set to 1 MB on this client only.
    client.app.state.settings = Settings(
        max_upload_mb=1,
        max_video_seconds=60,
        cors_origins=("*",),
        nvidia_api_key=None,
    )
    big = io.BytesIO(b"x" * (2 * (1 << 20)))  # 2 MiB > 1 MiB limit
    files = {"video": ("big.mp4", big, "video/mp4")}
    data = {"exercise": "squat"}
    r = client.post("/analyze", files=files, data=data)
    assert r.status_code == 413
    assert r.json()["error"] == "video_too_large"


def test_analyze_too_long_returns_413(client):
    # cv2 returns a 120s duration -> over default 60s limit.
    fake_report = _fake_report()
    client.app.state.analyzer_default.analyze.return_value = fake_report
    with patch("api.routes._video_duration_seconds", return_value=120.0):
        files = {"video": ("long.mp4", io.BytesIO(b"x" * 1024), "video/mp4")}
        data = {"exercise": "squat"}
        r = client.post("/analyze", files=files, data=data)
    assert r.status_code == 413
    assert r.json()["error"] == "video_too_long"
    # analyzer must NOT have been called when duration check fails
    client.app.state.analyzer_default.analyze.assert_not_called()


@pytest.mark.parametrize("mode", ["full", "sampled", "keyframes", "none"])
def test_analyze_passes_skeleton_output_through(client, mode):
    client.app.state.analyzer_default.analyze.return_value = _fake_report()
    files = {"video": ("x.mp4", io.BytesIO(b"x" * 1024), "video/mp4")}
    data = {"exercise": "squat", "skeleton_output": mode}
    r = client.post("/analyze", files=files, data=data)
    assert r.status_code == 200
    _, kwargs = client.app.state.analyzer_default.analyze.call_args
    assert kwargs["skeleton_output"] == mode


def test_enrich_true_uses_enriched_analyzer_when_present(client):
    enriched = MagicMock()
    enriched.analyze.return_value = _fake_report()
    client.app.state.analyzer_enriched = enriched

    files = {"video": ("x.mp4", io.BytesIO(b"x" * 1024), "video/mp4")}
    data = {"exercise": "squat", "enrich": "true"}
    r = client.post("/analyze", files=files, data=data)
    assert r.status_code == 200
    enriched.analyze.assert_called_once()
    client.app.state.analyzer_default.analyze.assert_not_called()


def test_enrich_true_without_key_falls_back_with_warning(client):
    # analyzer_enriched stays None (set by the fixture).
    fake = _fake_report()
    client.app.state.analyzer_default.analyze.return_value = fake
    files = {"video": ("x.mp4", io.BytesIO(b"x" * 1024), "video/mp4")}
    data = {"exercise": "squat", "enrich": "true"}
    r = client.post("/analyze", files=files, data=data)
    assert r.status_code == 200
    body = r.json()
    codes = {w["code"] for w in body["warnings"]}
    assert "ENRICHMENT_FAILED" in codes


def test_analyze_serializes_concurrent_requests(client):
    in_flight = 0
    max_in_flight = 0
    lock_for_counter = threading.Lock()

    def slow_analyze(*args, **kwargs):
        nonlocal in_flight, max_in_flight
        with lock_for_counter:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        _time.sleep(0.2)
        with lock_for_counter:
            in_flight -= 1
        return _fake_report()

    client.app.state.analyzer_default.analyze.side_effect = slow_analyze

    # Both threads must share the same event-loop portal so that asyncio.Lock
    # serialises them. Inject a shared portal directly (bypasses lifespan so
    # MediaPipe is never loaded) and recreate the lock inside that loop.
    import anyio.from_thread as _af

    with _af.start_blocking_portal(backend="asyncio") as portal:
        # Replace the asyncio.Lock with one created in *this* event loop.
        client.app.state.lock = portal.call(asyncio.Lock)
        # Point the client at this shared portal (no lifespan triggered).
        client.portal = portal

        def fire():
            files = {"video": ("x.mp4", io.BytesIO(b"x" * 1024), "video/mp4")}
            data = {"exercise": "squat"}
            return client.post("/analyze", files=files, data=data)

        t1 = threading.Thread(target=fire)
        t2 = threading.Thread(target=fire)
        t1.start(); t2.start()
        t1.join(); t2.join()

    assert max_in_flight == 1, (
        f"expected serialized execution but max_in_flight={max_in_flight}"
    )

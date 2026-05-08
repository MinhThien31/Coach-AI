"""Tests for api.errors handler registration."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sport_companion_ai.errors import (
    PoseExtractionError,
    UnsupportedExerciseError,
    VideoReadError,
)

from api.errors import register_exception_handlers


def _build_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/raise-unsupported")
    def _u():
        raise UnsupportedExerciseError("nope")

    @app.get("/raise-video")
    def _v():
        raise VideoReadError("bad")

    @app.get("/raise-pose")
    def _p():
        raise PoseExtractionError("oops")

    @app.get("/raise-other")
    def _o():
        raise RuntimeError("boom")

    return app


def test_unsupported_exercise_maps_to_400():
    client = TestClient(_build_app())
    r = client.get("/raise-unsupported")
    assert r.status_code == 400
    assert r.json() == {"error": "unsupported_exercise", "detail": "nope"}


def test_video_read_maps_to_400():
    client = TestClient(_build_app())
    r = client.get("/raise-video")
    assert r.status_code == 400
    assert r.json() == {
        "error": "video_read_failed",
        "detail": "cannot read uploaded file as video",
    }


def test_pose_extraction_maps_to_500():
    client = TestClient(_build_app())
    r = client.get("/raise-pose")
    assert r.status_code == 500
    assert r.json() == {"error": "pose_extraction_failed"}


def test_generic_exception_maps_to_500():
    # raise_server_exceptions=False is required: by default TestClient re-raises
    # uncaught exceptions before Starlette dispatches the catch-all handler.
    client = TestClient(_build_app(), raise_server_exceptions=False)
    r = client.get("/raise-other")
    assert r.status_code == 500
    assert r.json() == {"error": "internal_error"}

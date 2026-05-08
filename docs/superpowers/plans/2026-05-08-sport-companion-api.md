# Sport Companion AI — Phase 2 HTTP API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the Phase 1 `VideoAnalyzer` in a FastAPI HTTP service so external clients can submit a workout video via multipart upload and get back the same `AnalysisReport` JSON they would get from `from sport_companion_ai import VideoAnalyzer`.

**Architecture:** New `api/` package in the same repo, importing `sport_companion_ai` as a library. Synchronous request/response; multipart MP4 upload streamed to a temp file; single uvicorn worker; per-worker `asyncio.Lock` to serialize MediaPipe calls. Two `VideoAnalyzer`s share one `MediaPipeExtractor` and differ only by enricher (template vs. NIM); the handler picks based on the `enrich` form field. No auth, no persistence, no rate limiting in v0.

**Tech Stack:** Python 3.13, FastAPI ≥ 0.110, Uvicorn, python-multipart, httpx (already a main dep, used by FastAPI's TestClient), Pydantic v2 (already used by Phase 1), pytest.

**Spec:** `docs/superpowers/specs/2026-05-08-sport-companion-api-design.md`

---

## File Structure

**Create:**
- `api/__init__.py` — empty package marker
- `api/settings.py` — env-backed `Settings` dataclass
- `api/errors.py` — `exception_handlers` mapping Phase 1 exceptions to HTTP
- `api/schemas.py` — small request DTOs (`AnalyzeForm`)
- `api/deps.py` — FastAPI dependency providers (so tests can override)
- `api/main.py` — `FastAPI` app, lifespan, middleware, exception handler registration
- `api/routes.py` — `/health`, `/exercises`, `/analyze`
- `tests/test_api.py` — fast, mocked-analyzer unit tests
- `tests/integration/test_api.py` — fixture-video integration + NIM smoke

**Modify:**
- `pyproject.toml` — add `[project.optional-dependencies].api`
- `Dockerfile` — drop ENTRYPOINT, copy `api/`, install `[api]`, default `CMD` runs uvicorn
- `README.md` — add an "HTTP API quickstart" section

**Untouched:** every file under `sport_companion_ai/` (Phase 1 code is a stable dependency, not a thing to refactor).

---

## Task 1: Add API extras to pyproject and install

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `[api]` optional-dependencies group**

In `pyproject.toml`, replace the existing `[project.optional-dependencies]` block:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-mock>=3.14",
    "pytest-cov>=5.0",
    "pyyaml>=6.0",
]
api = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "python-multipart>=0.0.9",
]
```

`httpx` is already a top-level dependency, so FastAPI's `TestClient` can be used by Phase 1 tests too without a dev addition.

- [ ] **Step 2: Install both extras locally**

```bash
pip install -e ".[dev,api]"
```

Expected: pip resolves and installs FastAPI, uvicorn, python-multipart. No conflicts.

- [ ] **Step 3: Verify FastAPI is importable**

```bash
python -c "import fastapi; print(fastapi.__version__)"
```

Expected: prints a version `>= 0.110`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build(api): add fastapi/uvicorn/python-multipart [api] extras"
```

---

## Task 2: Create empty api package

**Files:**
- Create: `api/__init__.py`

- [ ] **Step 1: Create the package marker file**

```bash
mkdir -p api
: > api/__init__.py
```

- [ ] **Step 2: Verify it imports**

```bash
python -c "import api; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add api/__init__.py
git commit -m "feat(api): scaffold api package"
```

---

## Task 3: Settings module (TDD)

**Files:**
- Create: `api/settings.py`
- Test: `tests/test_api_settings.py`

The settings module reads four env vars once at import time (or via a factory). Defaults match the spec's "Configuration" table.

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_settings.py`:

```python
"""Tests for api.settings env-backed config."""
import os

import pytest

from api.settings import Settings


def test_defaults_when_no_env(monkeypatch):
    for var in ("API_MAX_UPLOAD_MB", "API_MAX_VIDEO_SECONDS",
                "API_CORS_ORIGINS", "NVIDIA_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    s = Settings.from_env()
    assert s.max_upload_mb == 100
    assert s.max_video_seconds == 60
    assert s.cors_origins == ["*"]
    assert s.nvidia_api_key is None


def test_overrides_from_env(monkeypatch):
    monkeypatch.setenv("API_MAX_UPLOAD_MB", "25")
    monkeypatch.setenv("API_MAX_VIDEO_SECONDS", "30")
    monkeypatch.setenv("API_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    s = Settings.from_env()
    assert s.max_upload_mb == 25
    assert s.max_video_seconds == 30
    assert s.cors_origins == ["http://localhost:3000", "http://localhost:5173"]
    assert s.nvidia_api_key == "nvapi-test"


def test_invalid_int_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("API_MAX_UPLOAD_MB", "not-a-number")
    with pytest.raises(ValueError):
        Settings.from_env()
```

- [ ] **Step 2: Run the tests, see them fail**

```bash
pytest tests/test_api_settings.py -v
```

Expected: FAIL — `ModuleNotFoundError: api.settings`.

- [ ] **Step 3: Implement `api/settings.py`**

```python
"""Env-backed configuration for the API. Read once at startup."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    max_upload_mb: int
    max_video_seconds: int
    cors_origins: list[str]
    nvidia_api_key: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            max_upload_mb=int(os.getenv("API_MAX_UPLOAD_MB", "100")),
            max_video_seconds=int(os.getenv("API_MAX_VIDEO_SECONDS", "60")),
            cors_origins=[
                o.strip()
                for o in os.getenv("API_CORS_ORIGINS", "*").split(",")
                if o.strip()
            ],
            nvidia_api_key=os.getenv("NVIDIA_API_KEY") or None,
        )
```

- [ ] **Step 4: Run the tests, see them pass**

```bash
pytest tests/test_api_settings.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add api/settings.py tests/test_api_settings.py
git commit -m "feat(api): env-backed Settings dataclass"
```

---

## Task 4: Exception → HTTP mapping module (TDD)

**Files:**
- Create: `api/errors.py`
- Test: `tests/test_api_errors.py`

`api/errors.py` exports a `register_exception_handlers(app)` function that wires Phase 1 exceptions to JSON responses with the codes from the spec.

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_errors.py`:

```python
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
    assert r.json() == {"error": "video_read_failed", "detail": "bad"}


def test_pose_extraction_maps_to_500():
    client = TestClient(_build_app())
    r = client.get("/raise-pose")
    assert r.status_code == 500
    assert r.json() == {"error": "pose_extraction_failed"}


def test_generic_exception_maps_to_500():
    client = TestClient(_build_app(), raise_server_exceptions=False)
    r = client.get("/raise-other")
    assert r.status_code == 500
    assert r.json() == {"error": "internal_error"}
```

- [ ] **Step 2: Run the tests, see them fail**

```bash
pytest tests/test_api_errors.py -v
```

Expected: FAIL — `ModuleNotFoundError: api.errors`.

- [ ] **Step 3: Implement `api/errors.py`**

```python
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
```

- [ ] **Step 4: Run the tests, see them pass**

```bash
pytest tests/test_api_errors.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add api/errors.py tests/test_api_errors.py
git commit -m "feat(api): exception handlers map Phase 1 errors to HTTP"
```

---

## Task 5: Dependency providers (TDD)

**Files:**
- Create: `api/deps.py`
- Test: `tests/test_api_deps.py`

Routes use `Depends(...)` to fetch the analyzer/lock from `app.state`. This indirection is what lets unit tests skip the heavy lifespan and inject mocks.

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_deps.py`:

```python
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
    assert get_default_analyzer(fake_request) == "DEF"
    assert get_enriched_analyzer(fake_request) == "ENR"
    assert get_lock(fake_request) is app.state.lock


def test_enriched_analyzer_returns_none_when_unset():
    app = FastAPI()
    app.state.analyzer_enriched = None
    fake_request = Request({"type": "http", "app": app})
    assert get_enriched_analyzer(fake_request) is None
```

- [ ] **Step 2: Run the tests, see them fail**

```bash
pytest tests/test_api_deps.py -v
```

Expected: FAIL — `ModuleNotFoundError: api.deps`.

- [ ] **Step 3: Implement `api/deps.py`**

```python
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
```

- [ ] **Step 4: Run the tests, see them pass**

```bash
pytest tests/test_api_deps.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add api/deps.py tests/test_api_deps.py
git commit -m "feat(api): dependency providers for analyzer/settings/lock"
```

---

## Task 6: FastAPI app, health endpoint, lifespan (TDD)

**Files:**
- Create: `api/main.py`
- Test: `tests/test_api.py` (new file — first test goes in here)

Lifespan loads the heavy MediaPipe analyzer once and stores everything on `app.state`. Tests bypass lifespan by using `TestClient(app)` without a `with` block and pre-populating `app.state` directly.

- [ ] **Step 1: Write the failing test**

Create `tests/test_api.py`:

```python
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
```

- [ ] **Step 2: Run the test, see it fail**

```bash
pytest tests/test_api.py::test_health_returns_ok -v
```

Expected: FAIL — `ModuleNotFoundError: api.main`.

- [ ] **Step 3: Implement `api/main.py`**

```python
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
```

- [ ] **Step 4: Run the test, see it pass**

```bash
pytest tests/test_api.py::test_health_returns_ok -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add api/main.py tests/test_api.py
git commit -m "feat(api): FastAPI app, lifespan with shared MediaPipe extractor, /health"
```

---

## Task 7: `/exercises` endpoint (TDD)

**Files:**
- Create: `api/routes.py`
- Modify: `api/main.py` (register the router)
- Modify: `tests/test_api.py` (add tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_api.py`:

```python
def test_exercises_lists_all_registered_rules(client):
    r = client.get("/exercises")
    assert r.status_code == 200
    data = r.json()
    assert set(data["exercises"]) == {
        "squat", "deadlift", "bench_press", "push_up", "bicep_curl",
    }
```

- [ ] **Step 2: Run the test, see it fail**

```bash
pytest tests/test_api.py::test_exercises_lists_all_registered_rules -v
```

Expected: FAIL — 404 (route not registered).

- [ ] **Step 3: Create `api/routes.py` with `/exercises`**

```python
"""HTTP route handlers for /analyze, /exercises (etc)."""
from __future__ import annotations

from fastapi import APIRouter

# Import side-effect: registers all rule classes into EXERCISE_REGISTRY.
import sport_companion_ai.exercises  # noqa: F401
from sport_companion_ai.exercises.base import EXERCISE_REGISTRY

router = APIRouter()


@router.get("/exercises", tags=["meta"])
async def list_exercises() -> dict[str, list[str]]:
    return {"exercises": sorted(EXERCISE_REGISTRY.keys())}
```

- [ ] **Step 4: Wire the router into `api/main.py`**

Add right above the `@app.get("/health")` decorator (and remove or keep `/health` here — keep it):

```python
from api.routes import router as api_router

app.include_router(api_router)
```

- [ ] **Step 5: Run the test, see it pass**

```bash
pytest tests/test_api.py::test_exercises_lists_all_registered_rules -v
```

Expected: PASS. (`sorted(...)` makes the comparison order-agnostic; the test uses a set equality, so either approach is fine.)

- [ ] **Step 6: Commit**

```bash
git add api/routes.py api/main.py tests/test_api.py
git commit -m "feat(api): GET /exercises lists registered rule names"
```

---

## Task 8: `/analyze` happy path with mocked analyzer (TDD)

**Files:**
- Create: `api/schemas.py`
- Modify: `api/routes.py`
- Modify: `tests/test_api.py`

> **Convention for Tasks 8–12:** imports shown at the top of each test snippet belong at the top of `tests/test_api.py` (standard Python module layout). Test functions get appended at the bottom of the file. Don't paste imports literally between existing test bodies.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_api.py`:

```python
import io
from pathlib import Path

from sport_companion_ai.report import (
    AnalysisReport, SkeletonSchema, VideoMeta,
)


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
```

- [ ] **Step 2: Run the test, see it fail**

```bash
pytest tests/test_api.py::test_analyze_happy_path_returns_report_json -v
```

Expected: FAIL — 404 (`/analyze` not registered) or 405.

- [ ] **Step 3: Create `api/schemas.py`**

```python
"""API-specific request DTOs. Response uses Phase 1's AnalysisReport directly."""
from __future__ import annotations

from typing import Literal

SkeletonOutputLiteral = Literal["full", "sampled", "keyframes", "none"]
```

- [ ] **Step 4: Implement `/analyze` in `api/routes.py`**

Replace `api/routes.py` with:

```python
"""HTTP route handlers for /analyze, /exercises, etc."""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
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


def _stream_to_temp(upload: UploadFile, max_bytes: int) -> str:
    """Stream `upload` into a NamedTemporaryFile. Raise HTTPException(413)
    mid-stream if size exceeds max_bytes. Return the temp file path on disk."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    written = 0
    try:
        while chunk := upload.file.read(1 << 20):  # 1 MiB chunks
            written += len(chunk)
            if written > max_bytes:
                tmp.close()
                os.unlink(tmp.name)
                raise HTTPException(
                    status_code=413,
                    detail={"error": "video_too_large",
                            "detail": f"max {max_bytes // (1 << 20)} MB"},
                )
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
```

- [ ] **Step 5: Run the test, see it pass**

```bash
pytest tests/test_api.py::test_analyze_happy_path_returns_report_json -v
```

Expected: PASS. The mock returns the fake report; response is its JSON.

- [ ] **Step 6: Commit**

```bash
git add api/schemas.py api/routes.py tests/test_api.py
git commit -m "feat(api): POST /analyze multipart upload + sync analysis"
```

---

## Task 9: Validation errors — bad exercise, missing field, bad video (TDD)

**Files:**
- Modify: `tests/test_api.py`
- Modify: `api/routes.py` (only if a test fails — most should pass already)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_api.py`:

```python
from sport_companion_ai.errors import UnsupportedExerciseError, VideoReadError


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
```

- [ ] **Step 2: Run the tests**

```bash
pytest tests/test_api.py -v -k "analyze and (missing or unknown or corrupt or invalid)"
```

Expected: all PASS — the exception handlers from Task 4 already cover bad exercise / bad video, and FastAPI's Pydantic validation produces 422 for missing/invalid form fields. If anything fails, fix the route, not the tests.

- [ ] **Step 3: Commit**

```bash
git add tests/test_api.py
git commit -m "test(api): validation error mapping for /analyze"
```

---

## Task 10: Upload-size and duration limits (TDD)

**Files:**
- Modify: `tests/test_api.py`
- Modify: `api/routes.py` (only if needed)

The size guard runs before the analyzer; duration runs after the file lands on disk.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_api.py`:

```python
from unittest.mock import patch


def test_analyze_too_large_returns_413(client):
    # Settings default is 100 MB — set to 1 MB on this client only.
    client.app.state.settings = Settings(
        max_upload_mb=1,
        max_video_seconds=60,
        cors_origins=["*"],
        nvidia_api_key=None,
    )
    big = io.BytesIO(b"x" * (2 * (1 << 20)))  # 2 MiB > 1 MiB limit
    files = {"video": ("big.mp4", big, "video/mp4")}
    data = {"exercise": "squat"}
    r = client.post("/analyze", files=files, data=data)
    assert r.status_code == 413
    assert r.json()["detail"]["error"] == "video_too_large"


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
```

- [ ] **Step 2: Run the tests**

```bash
pytest tests/test_api.py -v -k "too_large or too_long"
```

Expected: PASS. Implementation from Task 8 already enforces both limits.

- [ ] **Step 3: Commit**

```bash
git add tests/test_api.py
git commit -m "test(api): 413 for over-size and over-duration uploads"
```

---

## Task 11: `skeleton_output` and `enrich` parameter wiring (TDD)

**Files:**
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_api.py`:

```python
import pytest


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
```

- [ ] **Step 2: Run the tests**

```bash
pytest tests/test_api.py -v -k "skeleton_output or enrich"
```

Expected: PASS. Wiring from Task 8 already covers the three cases.

- [ ] **Step 3: Commit**

```bash
git add tests/test_api.py
git commit -m "test(api): skeleton_output passthrough and enrich toggle"
```

---

## Task 12: Concurrency lock test (TDD)

**Files:**
- Modify: `tests/test_api.py`

Verify two concurrent `POST /analyze` requests serialize through `app.state.lock` so MediaPipe is only called by one at a time.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_api.py`:

```python
import threading
import time as _time


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
```

- [ ] **Step 2: Run the test**

```bash
pytest tests/test_api.py::test_analyze_serializes_concurrent_requests -v
```

Expected: PASS. The `async with lock` in the route serializes requests within one worker.

- [ ] **Step 3: Commit**

```bash
git add tests/test_api.py
git commit -m "test(api): concurrency lock serializes /analyze requests"
```

---

## Task 13: Integration test against fixture videos (real analyzer)

**Files:**
- Create: `tests/integration/test_api.py`

Reuses Phase 1's fixture videos and `manifest.yaml`. Drives the API with the real `VideoAnalyzer` by triggering the lifespan via `with TestClient(app):`.

- [ ] **Step 1: Write the integration test**

Create `tests/integration/test_api.py`:

```python
"""Integration tests for the HTTP API against real fixture videos.

Marked `integration` so PR CI skips them. They share the Phase 1
fixture set and manifest (tests/fixtures/).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def load_manifest():
    with open(FIXTURES_DIR / "manifest.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def api_client():
    # Real lifespan -> loads MediaPipe once for the whole module.
    from api.main import app
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("fixture_id", list(load_manifest().keys()))
def test_analyze_endpoint_against_fixture(api_client, fixture_id):
    spec = load_manifest()[fixture_id]
    video_path = FIXTURES_DIR / spec["path"]
    if not video_path.exists():
        pytest.skip(f"fixture missing: {video_path}")

    with open(video_path, "rb") as fh:
        files = {"video": (video_path.name, fh, "video/mp4")}
        data = {"exercise": spec["exercise"], "skeleton_output": "keyframes"}
        r = api_client.post("/analyze", files=files, data=data)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["exercise"] == spec["exercise"]
    rmin, rmax = spec["expected_reps"]["min"], spec["expected_reps"]["max"]
    assert rmin <= body["total_reps"] <= rmax


@pytest.mark.requires_nim_key
def test_analyze_endpoint_with_real_enrichment(api_client):
    if not os.getenv("NVIDIA_API_KEY"):
        pytest.skip("NVIDIA_API_KEY not set")

    fixture = next(
        (k for k, v in load_manifest().items() if v["exercise"] == "bicep_curl"),
        None,
    )
    assert fixture, "expected at least one bicep_curl fixture"
    spec = load_manifest()[fixture]
    video_path = FIXTURES_DIR / spec["path"]
    if not video_path.exists():
        pytest.skip(f"fixture missing: {video_path}")

    with open(video_path, "rb") as fh:
        files = {"video": (video_path.name, fh, "video/mp4")}
        data = {"exercise": spec["exercise"], "enrich": "true"}
        r = api_client.post("/analyze", files=files, data=data)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["enriched"] is True
    assert body["session_summary"] and len(body["session_summary"]) > 0
```

- [ ] **Step 2: Run the integration tests (with fixtures present)**

```bash
pytest tests/integration/test_api.py -m integration -v
```

Expected: PASS for each fixture present (others skipped).

- [ ] **Step 3: Run the NIM smoke test (optional, requires key)**

```bash
NVIDIA_API_KEY=$YOUR_KEY pytest tests/integration/test_api.py -m requires_nim_key -v
```

Expected: PASS if a fixture is present and the key is valid; skipped otherwise.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_api.py
git commit -m "test(api): fixture-video integration + NIM enrichment smoke"
```

---

## Task 14: Dockerfile, README, manual smoke

**Files:**
- Modify: `Dockerfile`
- Modify: `README.md`

- [ ] **Step 1: Replace `Dockerfile` contents**

```dockerfile
FROM python:3.13-slim

# System libs needed by opencv + mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY sport_companion_ai/ ./sport_companion_ai/
COPY api/ ./api/
COPY examples/ ./examples/

RUN pip install --no-cache-dir -e ".[api]"

EXPOSE 8000
# Default: HTTP API. CLI demo still callable with an explicit override:
#   docker run --rm -v "$PWD":/data <image> \
#     python examples/analyze_squat.py /data/squat.mp4 --exercise squat
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

- [ ] **Step 2: Append an "HTTP API quickstart" section to `README.md`**

After the existing "Quick start (Docker)" section, add:

````markdown
## Quick start (HTTP API — Phase 2)

```bash
pip install -e ".[api]"
uvicorn api.main:app --reload
# OpenAPI: http://localhost:8000/docs
```

Submit a video:

```bash
curl -F "video=@squat.mp4" -F "exercise=squat" \
     http://localhost:8000/analyze | jq
```

With LLM enrichment (server reads `NVIDIA_API_KEY` from env):

```bash
curl -F "video=@squat.mp4" -F "exercise=squat" -F "enrich=true" \
     http://localhost:8000/analyze | jq .session_summary
```

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/analyze` | Submit video, get `AnalysisReport` JSON |
| `GET`  | `/exercises` | List supported exercise names |
| `GET`  | `/health` | Liveness check |
| `GET`  | `/docs`, `/openapi.json` | FastAPI auto-docs |

Configuration (env vars):

| Var | Default | Purpose |
|---|---|---|
| `API_MAX_UPLOAD_MB` | `100` | Reject uploads bigger than this (413) |
| `API_MAX_VIDEO_SECONDS` | `60` | Reject videos longer than this (413) |
| `API_CORS_ORIGINS` | `*` | Comma-separated origins |
| `NVIDIA_API_KEY` | unset | If set, `enrich=true` runs through NIM |

### Docker

The default image now runs the API. The Phase 1 ENTRYPOINT is removed —
running the CLI demo requires an explicit command:

```bash
docker build -t sport-companion-ai .
docker run --rm -p 8000:8000 sport-companion-ai           # API
docker run --rm -v "$PWD":/data sport-companion-ai \
  python examples/analyze_squat.py /data/squat.mp4 --exercise squat   # CLI
```
````

- [ ] **Step 3: Manual smoke (start server, hit /health, hit /exercises)**

In one terminal:

```bash
uvicorn api.main:app --port 8001
```

In another:

```bash
curl -s http://localhost:8001/health
# {"status":"ok"}
curl -s http://localhost:8001/exercises | jq
# {"exercises":["bench_press","bicep_curl","deadlift","push_up","squat"]}
```

Stop the server (Ctrl-C).

- [ ] **Step 4: Run the full unit test suite**

```bash
pytest -v
```

Expected: all unit tests pass; integration / requires_nim_key are skipped by default per `pyproject.toml` markers.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile README.md
git commit -m "docs(api): Phase 2 quickstart + Dockerfile defaults to uvicorn"
```

---

## Self-review checklist (run before handing off)

After all tasks are done:

- [ ] `pytest -v` — green
- [ ] `pytest -m integration -v` — green (with fixtures)
- [ ] `pytest -m requires_nim_key -v` — green (with `NVIDIA_API_KEY`)
- [ ] `uvicorn api.main:app` — starts, `/health` returns 200, `/docs` renders
- [ ] `docker build .` succeeds; container runs API on port 8000
- [ ] README quickstart commands actually work as written

Phase 3 next steps live in the spec under "Next Steps for Phase 3" (groups A–E). They are not part of this plan.

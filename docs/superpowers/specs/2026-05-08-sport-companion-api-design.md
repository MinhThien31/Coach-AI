# Sport Companion AI — Phase 2 HTTP API Design

**Status**: Approved — ready for implementation planning
**Date**: 2026-05-08
**Phase**: Phase 2 (Productize)
**Predecessor**: `2026-05-08-sport-companion-prototype-design.md` (Phase 1)

## Context & Goals

Phase 1 shipped a pure-Python library that takes a workout video and returns a structured form-evaluation report (5 exercises, MediaPipe pose, rule-based evaluation, optional NVIDIA NIM enrichment). It is consumable as `from sport_companion_ai import VideoAnalyzer` and via a CLI demo.

Phase 2 wraps that library in an HTTP service so an external client (web frontend, mobile app, internal tools) can call it without writing Python.

**Concrete scope:**
- FastAPI HTTP service in a new `api/` package, in the same repo as Phase 1.
- Runs locally first (single host, single Docker container). No auth, no rate limiting, no multi-tenancy.
- Multipart upload of an MP4 file. Synchronous request/response. JSON output only — no rendered overlay images or video.
- Same exercise coverage as Phase 1 (5 exercises). No new ML, no new exercises.
- Reuses the existing Phase 1 `VideoAnalyzer`, `AnalysisReport`, `NvidiaNimEnricher`, exception hierarchy, and warning catalog without modification.

**Out of scope for Phase 2** (see "Next Steps for Phase 3" at the end):
- Auth, API keys, rate limits, multi-tenancy
- Async / job-based execution for long videos
- Real-time / streaming mode
- Server-side rendered overlay images or annotated MP4
- Persistence (DB, object storage)
- New pose models (YOLO11-pose), new ML (Fitness-AQA), new exercises
- Multi-person / form comparison
- A frontend (the API serves whatever client we or others build later)

## Non-goals (explicit)

- **Not a sidecar.** The API is a standalone deployable service. It imports the `sport_companion_ai` package as a library; the package does not import anything from `api/`. (Aligns with the user's saved preference: cross-language systems → independent microservices, not sidecars.)
- **Not a separate repo.** Same repo as Phase 1 to avoid premature publish/version overhead. The `api/` package is structured so it can be extracted into its own repo later without rewrites.
- **Not async/streaming.** Synchronous request/response only in v0. Long-video and live-coaching needs are deferred to Phase 3.
- **Not multi-worker by default.** MediaPipe `PoseLandmarker` is not thread-safe and the pipeline is CPU-bound. v0 ships with one worker; scaling guidance is documented, not automated.

## Architecture

A single Python repo with two consumption modes; Phase 2 adds the second:

```
┌────────────────────────────────────────────────────────────┐
│  Mode 1: In-process import (Phase 1, unchanged)            │
│    from sport_companion_ai import VideoAnalyzer            │
│    report = VideoAnalyzer().analyze("squat.mp4", "squat")  │
├────────────────────────────────────────────────────────────┤
│  Mode 2: HTTP API (Phase 2, NEW)                           │
│    POST /analyze   (multipart: video + exercise + opts)    │
│    GET  /health                                            │
│    GET  /exercises                                         │
│    GET  /docs, /openapi.json                               │
└────────────────────────────────────────────────────────────┘
```

The API is a thin adapter:

```
HTTP request
   │
   ├─► (1) FastAPI parses multipart, streams upload to NamedTemporaryFile,
   │       enforces size and duration limits.
   │
   ├─► (2) Acquire single asyncio.Lock (serialize requests within one worker
   │       because MediaPipe is not thread-safe).
   │
   ├─► (3) run_in_threadpool(app.state.analyzer.analyze, tmp_path, exercise,
   │                         skeleton_output=..., enricher=...)
   │
   ├─► (4) Catch SportCompanionError subclasses → map to HTTP via
   │       exception_handlers (see Error Mapping table).
   │
   ├─► (5) Return AnalysisReport.model_dump() as JSON (200).
   │
   └─► (6) finally: os.unlink(tmp_path).
```

### Layout (additions only — Phase 1 code untouched)

```
sport-companion-ai/
├── api/                              # NEW — Phase 2
│   ├── __init__.py
│   ├── main.py                       # FastAPI app, lifespan, middleware
│   ├── routes.py                     # /analyze, /health, /exercises
│   ├── schemas.py                    # API-specific request/response models
│   ├── errors.py                     # exception_handlers mapping
│   └── settings.py                   # env-based config
├── tests/
│   ├── test_api.py                   # NEW — mocked analyzer, fast
│   └── test_api_integration.py       # NEW — real analyzer + fixture videos
├── Dockerfile                        # MODIFIED — default CMD becomes uvicorn
├── pyproject.toml                    # MODIFIED — add [api] extras
└── README.md                         # MODIFIED — add API quickstart section
```

### Module boundaries

- `api/routes.py` knows only "HTTP → analyzer call → JSON". No exercise or pose knowledge.
- `api/errors.py` maps Phase 1 exceptions to HTTP status codes. No new exception types.
- `api/schemas.py` holds request models (multipart fields are parsed, not Pydantic-validated as a single model). The response uses Phase 1's `AnalysisReport` directly via `model_dump()`.
- `api/settings.py` reads env vars once at startup.
- `sport_companion_ai/` (Phase 1) is unchanged. The API package depends on it; the reverse dependency does not exist.

## Endpoints

### `POST /analyze`

Multipart form-data fields:

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `video` | file (MP4) | yes | — | Streamed to temp file. Rejected if upload exceeds `API_MAX_UPLOAD_MB`. |
| `exercise` | str | yes | — | One of `squat`, `deadlift`, `bench_press`, `push_up`, `bicep_curl` (matches `EXERCISE_REGISTRY` keys). Anything else → 400. |
| `skeleton_output` | str | no | `keyframes` | One of `full`, `sampled`, `keyframes`, `none`. Invalid → 422. |
| `enrich` | bool | no | `false` | If `true` and the server has `NVIDIA_API_KEY`, the report runs through `NvidiaNimEnricher`. If `true` but no key, the request still succeeds with a `ENRICHMENT_FAILED` warning in the report (does not 500). |

**Response 200**: full `AnalysisReport` JSON, identical to Phase 1's `model_dump()`. Schema is documented in the Phase 1 spec; this design does not redefine it.

### `GET /health`

```json
{ "status": "ok" }
```

Returns 200 once the lifespan startup has loaded the analyzer (the model file `pose_landmarker_full.task` is downloaded on first run). Returns 503 if the analyzer failed to initialize.

### `GET /exercises`

```json
{ "exercises": ["squat", "deadlift", "bench_press", "push_up", "bicep_curl"] }
```

Drives a frontend dropdown. The list is the keys of `EXERCISE_REGISTRY` (populated when `sport_companion_ai.exercises` is imported) — adding an exercise to the package automatically extends this list. Order is registration order, not stable; clients should not rely on it.

### `GET /docs`, `GET /openapi.json`, `GET /redoc`

Auto-generated by FastAPI. Tagged: `analysis` (the `/analyze` endpoint), `meta` (`/health`, `/exercises`).

## Error Mapping

Only Phase 1 exception types are used; the API maps them to HTTP via `exception_handlers`.

| Source | HTTP | Body shape |
|---|---|---|
| `UnsupportedExerciseError` | 400 | `{"error":"unsupported_exercise","detail":"<msg>"}` |
| `VideoReadError` | 400 | `{"error":"video_read_failed","detail":"<msg>"}` |
| Pydantic / multipart validation | 422 | FastAPI default (`detail: [...]`) |
| Upload exceeds `API_MAX_UPLOAD_MB` | 413 | `{"error":"video_too_large","detail":"max <N> MB"}` |
| Video duration exceeds `API_MAX_VIDEO_SECONDS` | 413 | `{"error":"video_too_long","detail":"max <N>s"}` |
| `PoseExtractionError` | 500 | `{"error":"pose_extraction_failed"}` (also logged) |
| Any other unhandled exception | 500 | `{"error":"internal_error"}` (also logged) |

`EnricherError` is **not** raised through the HTTP boundary. Phase 1's `NvidiaNimEnricher` already absorbs failures into a `ENRICHMENT_FAILED` warning on the report; that contract is preserved.

## Concurrency, Lifecycle, Resources

### Concurrency model

- **One worker by default.** `uvicorn --workers 1`. To scale, operators run more workers (each loads its own MediaPipe model into RAM, ~150 MB).
- **Within a worker, requests serialize** through a single `asyncio.Lock`. MediaPipe `PoseLandmarker` is not documented as thread-safe, and the pipeline is CPU-bound, so true concurrency inside one worker would not help anyway.
- The `/analyze` handler is `async def` but the heavy work runs via `fastapi.concurrency.run_in_threadpool` so the event loop stays free for `/health` and `/exercises` while `/analyze` is processing.

### Lifespan

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    app.state.settings = Settings()                      # env → typed config
    extractor = MediaPipeExtractor()                     # heavy; load once
    app.state.analyzer_default = VideoAnalyzer(
        pose_extractor=extractor, enricher=TemplateEnricher(),
    )
    key = os.getenv("NVIDIA_API_KEY")
    app.state.analyzer_enriched = (
        VideoAnalyzer(pose_extractor=extractor, enricher=NvidiaNimEnricher(api_key=key))
        if key else None
    )
    app.state.lock = asyncio.Lock()
    yield
```

Two analyzers share a single `MediaPipeExtractor` (the heavy resource). One uses the no-op `TemplateEnricher`, the other (only constructed if `NVIDIA_API_KEY` is set) uses `NvidiaNimEnricher`. The handler picks based on the `enrich` form field. If `enrich=true` but `analyzer_enriched is None`, the default analyzer runs and the response gets a manually-appended `ENRICHMENT_FAILED` warning so the server-no-key path matches the in-process no-key path semantically.

### Upload handling

- The video field is streamed to `NamedTemporaryFile(suffix=".mp4", delete=False)` chunk by chunk — never loaded fully into RAM.
- During streaming, a running byte counter enforces `API_MAX_UPLOAD_MB`; exceeding it aborts the write and returns 413.
- After the file is on disk, `cv2.VideoCapture` reads metadata to check `duration <= API_MAX_VIDEO_SECONDS`; over → 413.
- A `try / finally` around the analyze call always `os.unlink`s the temp file, even on error.

### Configuration (env vars)

| Var | Default | Purpose |
|---|---|---|
| `API_MAX_UPLOAD_MB` | `100` | Reject upload over this size (return 413 mid-stream). |
| `API_MAX_VIDEO_SECONDS` | `60` | Reject videos longer than this (sync timeout safety). |
| `API_CORS_ORIGINS` | `*` | Comma-separated origins. `*` for local dev; tighten in any non-local deploy. |
| `NVIDIA_API_KEY` | unset | Same semantics as Phase 1 CLI: present → enrichment available. |

`api/settings.py` reads these once at startup into a typed `Settings` object.

## Packaging

### `pyproject.toml`

```toml
[project.optional-dependencies]
api = [
  "fastapi>=0.110",
  "uvicorn[standard]>=0.27",
  "python-multipart>=0.0.9",
]
# dev extras unchanged — httpx is already a main dependency, so FastAPI's
# TestClient (which uses httpx) needs no addition.
```

Phase 1 users (`pip install -e .`) get no API dependencies. Phase 2 users run `pip install -e ".[api]"`.

### Dockerfile (modified)

One image, two entrypoints. The default is the API; CLI demo is still runnable by overriding `CMD`.

```dockerfile
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY sport_companion_ai/ ./sport_companion_ai/
COPY api/ ./api/
COPY examples/ ./examples/

RUN pip install --no-cache-dir -e ".[api]"

EXPOSE 8000
# No ENTRYPOINT (Phase 1 used one for the CLI). API is the default.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

The Phase 1 ENTRYPOINT (`python examples/analyze_squat.py`) is removed; CLI usage now requires an explicit command override:

```bash
docker run --rm -v "$PWD":/data sport-companion-ai \
  python examples/analyze_squat.py /data/squat.mp4 --exercise squat
```

This is a breaking change for any user that depended on the Phase 1 image's bare-flag invocation. Document it in the README API section.

### Middleware

- `CORSMiddleware` configured from `API_CORS_ORIGINS`.
- A small request-logging middleware: log method, path, status, elapsed ms (no bodies, no upload contents).

## Testing Strategy

### Tier 1 — API unit tests (default in PR CI)

`tests/test_api.py`. Uses `fastapi.testclient.TestClient`. Overrides `app.state.analyzer` with a `MagicMock` returning a fixed `AnalysisReport`. Goal: prove the **HTTP layer** is correct, independent of the pipeline.

| Case | Assert |
|---|---|
| `GET /health` | 200, body `{"status":"ok"}` |
| `GET /exercises` | 200, list contains all 5 names |
| `POST /analyze` happy path (mock analyzer) | 200, body matches `AnalysisReport` schema |
| `POST /analyze` missing `exercise` | 422 |
| `POST /analyze` invalid `exercise` | 400, error code `unsupported_exercise` |
| `POST /analyze` upload over limit | 413, error code `video_too_large` |
| `POST /analyze` non-video / corrupt MP4 | 400, error code `video_read_failed` |
| `POST /analyze` `enrich=true` but no key on server | 200, report has `ENRICHMENT_FAILED` warning |
| `POST /analyze` for each `skeleton_output` mode | 200, `frames[]` size matches mode |
| Two concurrent `POST /analyze` requests | both 200; mock sees them serialize via lock |

### Tier 2 — Integration tests (real analyzer, fixture videos)

`tests/test_api_integration.py`, marker `@pytest.mark.integration`. Reuses the existing Phase 1 fixture videos and `tests/fixtures/manifest.yaml`.

| Case | Assert |
|---|---|
| Upload `squat_good_*.mp4` | 200; `total_reps > 0`; `avg_score` within manifest range |
| Upload a known-bad squat fixture | 200; rep with expected issue code (e.g., `SQUAT_DEPTH_INSUFFICIENT`) present |

### Tier 3 — NIM smoke test

`tests/test_api_integration.py::test_enrichment_real`, marker `@pytest.mark.requires_nim_key`. Skipped without `NVIDIA_API_KEY`. Asserts `enriched=true` and a non-empty `session_summary` after a real call.

### CI matrix (extends Phase 1)

```
PR check        : pytest -m "not integration and not requires_nim_key"
Nightly / main  : pytest -m integration
Manual / nightly: pytest -m requires_nim_key
```

### Coverage targets

| Module | Target |
|---|---|
| `api/routes.py`, `api/errors.py` | ≥ 90% (mock-driven) |
| `api/main.py`, `api/settings.py` | ≥ 70% |

Phase 1 modules' coverage targets are unchanged.

## Phase 2 Deliverables

1. `api/` package with FastAPI app, routes, error mapping, settings.
2. `tests/test_api.py` (mocked) and `tests/test_api_integration.py` (fixture-backed + NIM smoke).
3. `pyproject.toml` extended with `[api]` and `httpx` in `[dev]`.
4. `Dockerfile` updated; default `CMD` runs the API; CLI still callable.
5. README extended with an "HTTP API quickstart" section: `pip install -e ".[api]"`, `uvicorn api.main:app --reload`, `curl` example, Docker example, env-var reference.
6. OpenAPI auto-published at `/docs` and `/openapi.json`.

## Estimated Effort (Phase 2)

| Workstream | Days |
|---|---|
| `api/` package (routes, error map, settings, lifespan) | 1–2 |
| Upload handling, limits, multipart streaming | 0.5–1 |
| Tests (mocked + integration + NIM smoke) | 1–2 |
| Dockerfile + pyproject + README + OpenAPI polish | 0.5–1 |
| **Total** | **~3–6 days (1 dev full-time)** |

## Next Steps for Phase 3

Phase 2 deliberately stops at "local-first JSON API". The deferred items below are grouped so each can become its own brainstorm → spec → plan cycle. None of them require changes to Phase 2's HTTP contract; all are additive.

### A. Production-readiness for the API itself

| Item | Triggered when |
|---|---|
| **Auth** — API key middleware first (header `X-API-Key`), JWT later if the product gets a user model | Any deploy outside of `localhost` |
| **Rate limiting / quotas** — per-key request and per-key per-day video minutes | Multi-tenant or public exposure |
| **Async / job-based mode** — `POST /analyze` returns `202 + job_id`; `GET /jobs/{id}` for status; persistence in SQLite/Redis; FastAPI `BackgroundTasks` or a separate worker | Videos longer than ~60s become common, or sync responses start hitting client/proxy timeouts |
| **Persistence** — store reports (Postgres) and uploaded videos (S3-compatible) keyed by `job_id`; expose `GET /jobs/{id}/report` and `GET /jobs/{id}/video` | Reports need to be re-fetched, shared, or audited |
| **Multi-worker / GPU** — document `--workers N` sizing; investigate whether MediaPipe Tasks supports GPU; revisit when YOLO11-pose lands (which does benefit from GPU) | Throughput becomes the bottleneck |
| **Observability** — structured JSON logs, request IDs, Prometheus metrics endpoint, basic SLO dashboard | First non-local deploy |

### B. Evaluation quality (independent of API)

These improvements drop into the existing `PoseExtractor` and `ExerciseRule` interfaces with no API change.

| Item | Notes |
|---|---|
| **YOLO11-pose backend** | Alternative `PoseExtractor`. Better on side-view, multi-person, occlusion. Trade-offs: 2D-only by default, AGPL-3.0 license unless commercial. Recommend benchmarking on real user videos against MediaPipe before committing. |
| **Fitness-AQA integration** | Squat-only initially. Plug in via the reserved `aqa/` package and `AqaClient` Protocol. Surface its score as a complementary `aqa_score` field on the squat report; rules still drive `issue.code`. Risk: pretrained weights for production use may not be publicly available. |
| **Beyond 5 exercises** | OHP, row, pull-up, lunge, RDL — each is one new file under `exercises/`, no shared-code changes. Drives content depth more than architecture. |
| **Multi-person handling** | Today's `TOO_MANY_PEOPLE` warning picks the largest centered bbox. Phase 3 could either reject multi-person videos cleanly, or analyze each person separately and return a list. |

### C. Real-time / streaming mode

Architectural shift from batch to stream. Phase 1 reserved an `analyze_stream(frame_iter)` interface but did not implement it.

| Item | Notes |
|---|---|
| **WebSocket endpoint** `/analyze/stream` | Client pushes frame chunks; server pushes incremental rep evaluations and warnings. |
| **Per-rep streaming evaluation** | `RepDetector` needs to become incremental (state machine across frames) instead of operating on a complete angle series. |
| **Live-coaching feedback** | Lower-latency LLM (or template-only) chosen over the heavyweight enricher used in batch mode. |

### D. Form comparison

Compare a user's rep against a reference "good" rep (canned, or another user's). Architecturally this is a new endpoint plus a new comparison module that consumes two `AnalysisReport`s. Independent of A–C.

### E. Frontend

Out of scope for the backend project but worth flagging: the API's JSON output (especially the per-rep `keyframes` and skeleton `frames[]`) is the contract any frontend will rely on. Don't break it lightly in Phase 3 — version the API (`/v1/analyze`) the moment a real frontend exists.

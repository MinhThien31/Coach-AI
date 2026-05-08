# Phase 2 HTTP API — Manual QA Test Plan

**Date:** 2026-05-08
**Feature:** Sport Companion AI Phase 2 HTTP API
**Spec:** `docs/superpowers/specs/2026-05-08-sport-companion-api-design.md`
**Plan:** `docs/superpowers/plans/2026-05-08-sport-companion-api.md`

## Goal

Complement the automated suite (103 unit + 8 integration + 1 NIM smoke) with a structured QA pass that exercises the running uvicorn process end-to-end. Focus on:
- Behavior the tests can't easily cover (real HTTP layer, OpenAPI docs render, response headers, error message wording in Vietnamese, real CORS preflight).
- Verifying the contract as a third-party developer would experience it (curl-driven).

## Preconditions

- Server: `uvicorn api.main:app --port 8001 --workers 1`
- Default settings (`API_MAX_UPLOAD_MB=100`, `API_MAX_VIDEO_SECONDS=60`, `API_CORS_ORIGINS=*`)
- `NVIDIA_API_KEY` loaded from `.env` (verified present)
- Fixtures in `tests/fixtures/videos/` (8 MP4s confirmed)

## Phases & Cases

### Phase A — Meta endpoints

| TC | Description | Steps | Expected | Priority | Type |
|---|---|---|---|---|---|
| A1 | Health check | `GET /health` | 200, body `{"status":"ok"}` | P0 | API |
| A2 | Exercises list | `GET /exercises` | 200, `{"exercises":[...]}` chứa cả 5 tên `bench_press,bicep_curl,deadlift,push_up,squat` | P0 | API |
| A3 | OpenAPI schema | `GET /openapi.json` | 200, JSON valid; `paths` chứa `/analyze`, `/health`, `/exercises`; `info.title` = "Sport Companion AI API" | P0 | API |
| A4 | Swagger UI render | `GET /docs` | 200, HTML response, body chứa `swagger-ui` | P1 | API |
| A5 | ReDoc render | `GET /redoc` | 200, HTML response | P2 | API |
| A6 | 404 unknown path | `GET /not-a-real-route` | 404, FastAPI default body | P1 | API |

### Phase B — `/analyze` happy paths (5 exercises)

| TC | Fixture | Exercise | Expected | Priority | Type |
|---|---|---|---|---|---|
| B1 | `squat_emptybar.mp4` | `squat` | 200, `total_reps>=1`, response shape khớp `AnalysisReport` | P0 | API |
| B2 | `deadlift_man.mp4` | `deadlift` | 200, `total_reps>=1` | P0 | API |
| B3 | `bench_woman.mp4` | `bench_press` | 200, `total_reps>=1`, `passed_reps>=1` (manifest range 80-100 score) | P0 | API |
| B4 | `pushup_incline.mp4` | `push_up` | 200, `total_reps>=3` | P0 | API |
| B5 | `curl_dumbbell.mp4` | `bicep_curl` | 200, `total_reps>=4` | P0 | API |

For each: verify response has `exercise`, `pose_model`, `video.fps`, `video.duration_ms`, `skeleton_schema.keypoint_names`, `reps[].score`, `reps[].keyframes`, `warnings`. Default `skeleton_output=keyframes` → `frames[]` chứa per-rep start/peak/end.

### Phase C — `/analyze` error paths

| TC | Setup | Expected | Priority | Type |
|---|---|---|---|---|
| C1 | POST không có `exercise` field | 422 Pydantic validation | P0 | API |
| C2 | POST không có `video` field | 422 Pydantic validation | P0 | API |
| C3 | `exercise=flying` (unknown) | 400, body `{"error":"unsupported_exercise","detail":"Unknown exercise: 'flying'"}` | P0 | API |
| C4 | Upload non-video bytes (e.g., text file đặt extension `.mp4`) | 400, body `{"error":"video_read_failed",...}` | P0 | API |
| C5 | `skeleton_output=invalid_mode` | 422 | P1 | API |
| C6 | `enrich=not-a-bool` | 422 | P1 | API |

### Phase D — Limits

| TC | Setup | Expected | Priority | Type |
|---|---|---|---|---|
| D1 | Upload 200MB random bytes (>default 100MB) | 413, `{"error":"video_too_large","detail":"max 100 MB"}` flat shape | P0 | API |
| D2 | Upload nominally video-shaped but >60s duration | 413, `{"error":"video_too_long","detail":"max 60s"}` flat shape | P1 | API |

D2 is harder to set up cleanly (need a real >60s video). If no fixture is >60s, skip with explicit reason.

### Phase E — Skeleton output modes

Use `squat_emptybar.mp4` (smallest fixture). For each mode, compare `len(frames)`:

| TC | Mode | Expected | Priority | Type |
|---|---|---|---|---|
| E1 | `full` | `len(frames)` = total frames in video (~all) | P1 | API |
| E2 | `sampled` | `len(frames)` ≪ full, > 0 | P1 | API |
| E3 | `keyframes` (default) | `len(frames)` = sum of per-rep start/peak/end + issue refs | P0 | API |
| E4 | `none` | `len(frames)` == 0 | P1 | API |

### Phase F — Enrichment toggle

| TC | Setup | Expected | Priority | Type |
|---|---|---|---|---|
| F1 | `enrich=true` (with NVIDIA_API_KEY in env) | 200, `enriched=true`, `session_summary` non-empty (Vietnamese text) | P0 | API |
| F2 | `enrich=true` after temporarily unsetting key (server-side) | 200, `enriched=false`, `warnings[].code` contains `ENRICHMENT_FAILED` | P1 | API |
| F3 | `enrich=false` (default) | 200, `enriched=false`, `session_summary=null` | P0 | API |

F2 requires either restarting the server without the key, or testing it via `tests/test_api.py::test_enrich_true_without_key_falls_back_with_warning` (already covered in unit suite). Will reference unit test rather than restarting server.

### Phase G — CORS preflight

| TC | Setup | Expected | Priority | Type |
|---|---|---|---|---|
| G1 | `OPTIONS /analyze` with `Origin: http://localhost:3000` and `Access-Control-Request-Method: POST` | 200, response includes `Access-Control-Allow-Origin: *`, `Access-Control-Allow-Methods` includes POST | P1 | API |

### Phase H — Concurrency smoke

| TC | Setup | Expected | Priority | Type |
|---|---|---|---|---|
| H1 | Two concurrent `POST /analyze` (same fixture) via parallel curl | Both return 200 with valid reports; total wall time ≈ 2× sequential (proves serialization through lock) | P2 | API |

H1 is also covered by `tests/test_api.py::test_analyze_serializes_concurrent_requests`. The manual version verifies it under real uvicorn (not TestClient).

## Execution Strategy

Group tests into 3 parallel batches that can run concurrently against the same server (asyncio.Lock serializes /analyze calls internally; meta endpoints are non-conflicting):

- **Batch 1**: Phase A (meta) + Phase G (CORS)
- **Batch 2**: Phase B (5 happy paths) + Phase E (skeleton modes)
- **Batch 3**: Phase C (errors) + Phase D (limits) + Phase F (enrichment) + Phase H (concurrency smoke)

Each batch produces a partial results doc; merge into final `results.md`.

## Out of Scope

- UI/browser testing (no frontend yet; deferred to Phase 3)
- Auth/rate-limit testing (intentionally absent in v0)
- Multi-worker / load testing (single worker by design)
- Performance benchmarking
- Real video uploads from a non-fixture source

## Severity Tags for Findings

- **HIGH** — blocks shipping; spec violation or contract break
- **MEDIUM** — works but degraded UX or undocumented edge case
- **LOW** — cosmetic / minor inconsistency

## Language

Product surfaces Vietnamese error messages (`message_vi`) and Vietnamese session summaries when enriched. Bug reports cite VN strings verbatim.

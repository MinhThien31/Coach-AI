# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Sport Companion AI — Phase 1/2 prototype that ingests an MP4 of a gym lift and returns a structured form-evaluation `AnalysisReport` (Vietnamese coaching strings). Pose extraction is **MediaPipe BlazePose Full via the Tasks API**; supported exercises are squat, deadlift, bench press, push-up, bicep curl. Phase 2 wraps the library with a FastAPI HTTP service.

## Common commands

```bash
# install (editable, with dev + api extras)
pip install -e ".[dev,api]"

# default: fast unit tests only — integration/NIM marks are excluded by addopts
pytest

# integration suite (needs MP4s in tests/fixtures/videos/; see manifest.yaml for source URLs)
pytest -m integration

# tests that hit the real NVIDIA NIM API
NVIDIA_API_KEY=nvapi-... pytest -m requires_nim_key

# run a single test
pytest tests/test_analyzer.py::test_name -xvs

# run the HTTP API locally
uvicorn api.main:app --reload   # OpenAPI at http://localhost:8000/docs

# run the CLI demo
python examples/analyze_squat.py path/to/clip.mp4 --exercise squat [--enrich-with-nim]

# Docker (default CMD = uvicorn API on :8000)
docker build -t sport-companion-ai .
docker run --rm -p 8000:8000 sport-companion-ai
```

`pyproject.toml` sets `addopts = "-m 'not integration and not requires_nim_key'"` — those marks are opt-in. The integration test runner has a `pytest_sessionfinish` hook that writes `tests/integration/_artifacts/results.md` with a pass/fail table and per-rep peak skeleton PNGs; that directory is gitignored.

On first pose extraction, `MediaPipeExtractor` downloads `pose_landmarker_full.task` (~17 MB) into `~/.cache/sport_companion_ai/`. Tests can bypass the download by passing `model_path=` to the extractor.

## Architecture — the part you can't infer from the file tree

### Pipeline shape (`VideoAnalyzer.analyze`)

```
read_video → PoseExtractor.extract_batch → ExerciseRule.detect_reps
   → ExerciseRule.evaluate_rep (per rep) → detect_warnings
   → select_frames_for_output → FeedbackEnricher.enrich → AnalysisReport
```

Frames flow as `list[Frame]` end-to-end. `Frame.skeleton` may be `None` (no detection that frame); rule code must handle that — use `math.isnan` and skip, never raise.

### Rule registry — import-as-side-effect

`sport_companion_ai/exercises/__init__.py` imports every rule module purely for the `@register_rule` decorator side effect, populating `EXERCISE_REGISTRY`. `analyzer.py` and `api/routes.py` both `import sport_companion_ai.exercises  # noqa: F401` so the lookup `ExerciseRule.get(name)` works.

**When adding a new exercise**: drop a module under `exercises/`, decorate the class with `@register_rule`, and add it to the import list in `exercises/__init__.py`. Forgetting that last step is silent — `/exercises` simply omits it.

### Threshold contract

Rule classes pin form-evaluation thresholds as class constants (e.g. `SquatRule.DEPTH_TARGET = 95.0`). Comments on these constants (`"# MUST match spec exactly"`) are load-bearing — the spec at `docs/superpowers/specs/2026-05-08-sport-companion-prototype-design.md` is the source of truth, and the integration tests in `tests/fixtures/manifest.yaml` assert behavior tied to those exact numbers. Don't tune thresholds to make a test pass; update the spec first.

### Issue codes are an API

Each rule emits stable `Issue.code` strings (e.g. `SQUAT_KNEE_VALGUS`, `PUSHUP_HIP_SAG`) plus Vietnamese `message_vi` and `recommendation`. Frontends and the integration manifest (`required_issues_present` / `required_issues_absent`) match on these codes. **Renaming a code is a breaking change** — preserve old codes or update the manifest in the same commit.

### Vietnamese strings

`message_vi` and `recommendation` ship verbatim to clients; the NIM enricher prompt is also Vietnamese. When editing or asserting on these strings, cite the exact Vietnamese text — don't translate.

### Enricher contract — soft-fail invariant

`FeedbackEnricher.enrich(report) -> AnalysisReport` is a `Protocol`. Implementations may set `report.session_summary` and `report.enriched = True`, but must **never** mutate `score`, `passed`, `metrics`, or `issue.code`. On failure, append an `AnalysisWarning(code="ENRICHMENT_FAILED", ...)` and return the report unchanged. `NvidiaNimEnricher` follows this — it catches every `httpx`/parsing error in `_call_with_retry` and degrades silently. The HTTP route at `api/routes.py:104-113` mirrors this contract: when `enrich=true` is requested but the server has no key, it falls back to the default analyzer and injects the same warning so in-process and over-HTTP behavior match.

### API request handling

`api/routes.py:_stream_to_temp` writes uploads to a NamedTemporaryFile in 1 MiB chunks, enforcing `API_MAX_UPLOAD_MB` mid-stream (returns 413 without buffering the whole file). Then `_video_duration_seconds` opens the file with OpenCV to enforce `API_MAX_VIDEO_SECONDS` (also 413). The temp file is unlinked in `finally`.

The route serializes analyses through `app.state.lock` (an `asyncio.Lock`) and dispatches the actual analyzer call via `run_in_threadpool` — MediaPipe is GIL-bound, so concurrent uploads queue rather than thrash. `lifespan` constructs one `MediaPipeExtractor` and reuses it across `analyzer_default` (always present) and `analyzer_enriched` (only if `NVIDIA_API_KEY` is set at startup).

### Error mapping

`api/errors.register_exception_handlers` maps the library exception hierarchy from `sport_companion_ai/errors.py` to stable HTTP error codes:

| Exception | HTTP | `error` field |
|---|---|---|
| `UnsupportedExerciseError` | 400 | `unsupported_exercise` |
| `VideoReadError` | 400 | `video_read_failed` (detail is sanitized — never echo `exc` to the client; it can contain filesystem paths) |
| `PoseExtractionError` | 500 | `pose_extraction_failed` |
| anything else | 500 | `internal_error` |

The path-leak fix is in commit `87a7ff5` — preserve that pattern when adding new error handlers.

### Output schema modes (`SkeletonOutputMode`)

`frames[]` in the report is shaped by `skeleton_output`:
- `full` — every frame (large payload)
- `sampled` — ~5 fps subsample
- `keyframes` (default) — only per-rep start/peak/end + frames referenced by `Issue.frame_indices`
- `none` — empty list

`select_frames_for_output` is the single source of truth; the API exposes the same literal values via `SkeletonOutputLiteral` in `api/schemas.py`.

## Repo conventions

- Python ≥ 3.12 (`pyproject.toml` sets `requires-python`). Modern syntax — `X | None`, `list[T]`, no `typing.Optional` / `typing.List`.
- Pydantic v2 (`model_dump_json`, `BaseModel`, `Field(ge=..., le=...)`).
- Two installable packages from one source tree: `sport_companion_ai` (the library) and `api` (the FastAPI app). `pyproject.toml`'s `[tool.hatch.build.targets.wheel] packages` lists both.
- Specs and plans live in `docs/superpowers/specs/` and `docs/superpowers/plans/`. Manual QA runs go under `docs/superpowers/test-runs/<YYYY-MM-DD>-<feature>/` with `test-plan.md` + `results.md`. Read the spec before changing rule thresholds, schema fields, or error contracts — those docs are the contract.
- `tests/fixtures/videos/*.mp4` is gitignored. To restore fixtures, follow `tests/fixtures/manifest.yaml` — every entry records the Pexels URL and a `yt-dlp` recipe at the top of the file.

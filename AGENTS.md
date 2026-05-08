# Repository Guidelines

## Project Structure & Module Organization

This repository contains a Python 3.12 gym-video form evaluator. Core library code lives in `sport_companion_ai/`: `pose/` handles video/keypoint extraction, `exercises/` contains per-exercise rule classes, `feedback/` enriches reports, and `viz/` provides rendering helpers. The FastAPI service is in `api/`, with routes, settings, schemas, dependencies, and error mapping separated by file. CLI/demo entry points live in `examples/`. Tests are under `tests/`, with slower video-backed cases in `tests/integration/` and fixture metadata in `tests/fixtures/manifest.yaml`. Product specs, plans, and QA evidence live under `docs/superpowers/`.

## Build, Test, and Development Commands

- `pip install -e ".[dev,api]"`: install the package in editable mode with test and API dependencies.
- `pytest`: run the default fast suite; `pyproject.toml` excludes `integration` and `requires_nim_key` markers by default.
- `pytest -m integration`: run video fixture integration tests; requires MP4s restored under `tests/fixtures/videos/`.
- `NVIDIA_API_KEY=nvapi-... pytest -m requires_nim_key`: run tests that call NVIDIA NIM.
- `uvicorn api.main:app --reload`: start the HTTP API locally; OpenAPI is at `http://localhost:8000/docs`.
- `python examples/analyze_squat.py path/to/clip.mp4 --exercise squat`: run the CLI analyzer.
- `docker build -t sport-companion-ai . && docker run --rm -p 8000:8000 sport-companion-ai`: build and run the API container.

## Coding Style & Naming Conventions

Use Python 3.12 idioms: `X | None`, `list[T]`, Pydantic v2 APIs, and explicit type hints for public interfaces. Follow 4-space indentation and small-module style. Test functions use `test_...`; helper functions may be private with a leading underscore. Exercise rule modules should register classes with `@register_rule` and be imported from `sport_companion_ai/exercises/__init__.py`.

## Testing Guidelines

Add focused unit tests beside the behavior you change. Use mocks or synthetic `Frame` objects for fast analyzer/rule tests, and reserve real videos for `tests/integration/`. Preserve marker intent: default `pytest` must stay fast and offline. When changing thresholds, issue codes, schemas, or API error contracts, update tests and the relevant spec/manifest together.

## Commit & Pull Request Guidelines

Recent history uses Conventional Commit style such as `fix(api): ...`, `test(api): ...`, and `docs(qa): ...`; keep that pattern. Pull requests should include a short problem/solution summary, linked issue or spec when applicable, test commands run, and API/schema impact notes. Include screenshots or generated artifacts only for visualization or QA changes.

## Security & Configuration Tips

Keep secrets in `.env` and copy from `.env.example` when needed. Never commit `NVIDIA_API_KEY`, downloaded MediaPipe models, fixture videos, coverage files, or generated integration artifacts. API errors must avoid leaking filesystem paths or raw internal exceptions.

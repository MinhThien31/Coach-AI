# Sport Companion AI

Phase 1 prototype: take a gym video, get a structured form-evaluation report.

## What it does

- Reads an MP4 video
- Extracts skeleton keypoints with MediaPipe BlazePose (Tasks API, downloads `pose_landmarker_full.task` on first use)
- Detects reps and evaluates form for: **squat, deadlift, bench press, push-up, bicep curl**
- Returns a JSON report with per-rep score, issues (Vietnamese), metrics, and optional skeleton frames for visualization
- Optional: enrich with a natural-language coaching summary via the **NVIDIA NIM** API

See `docs/superpowers/specs/2026-05-08-sport-companion-prototype-design.md` for the full design.

## Install

```bash
pip install -e ".[dev]"
```

## Quick start (Python)

```python
from sport_companion_ai import VideoAnalyzer

analyzer = VideoAnalyzer()
report = analyzer.analyze("squat.mp4", exercise="squat")
print(report.model_dump_json(indent=2))
```

With NVIDIA NIM enrichment:

```python
import os
from sport_companion_ai import VideoAnalyzer
from sport_companion_ai.feedback.nim import NvidiaNimEnricher

analyzer = VideoAnalyzer(
    enricher=NvidiaNimEnricher(api_key=os.environ["NVIDIA_API_KEY"]),
)
report = analyzer.analyze("squat.mp4", exercise="squat")
print(report.session_summary)
```

## Quick start (CLI)

```bash
python examples/analyze_squat.py squat.mp4 --exercise squat --skeleton keyframes
```

## Quick start (Docker)

```bash
docker build -t sport-companion-ai .
docker run --rm -v "$PWD":/data sport-companion-ai /data/squat.mp4 --exercise squat
```

## Output schema

Top-level fields:

| Field | Description |
|---|---|
| `exercise` | which rule was applied |
| `total_reps`, `passed_reps`, `avg_score` | aggregate over reps |
| `reps[]` | per-rep score, passed flag, issues, metrics, keyframe indices |
| `frames[]` | skeleton keypoints (size depends on `skeleton_output`) |
| `skeleton_schema` | keypoint names + edges, for FE rendering |
| `warnings[]` | soft issues (low confidence, no reps, low fps, ...) |
| `session_summary` | LLM-generated string, only when enricher succeeded |
| `enriched` | true iff an LLM enricher modified the report |

Skeleton output modes:

| Mode | Frames returned |
|---|---|
| `full` | every frame |
| `sampled` | ~5 fps subsample |
| `keyframes` (default) | per-rep start/peak/end + issue frames |
| `none` | omitted |

## Run tests

```bash
pytest                                       # fast unit tests (default)
pytest -m integration                        # requires fixture videos in tests/fixtures/videos/
pytest -m requires_nim_key                   # requires NVIDIA_API_KEY env var
```

## Project layout

See `sport_companion_ai/` for source. Each subpackage has one responsibility:
- `pose/` — video read, keypoint extraction
- `exercises/` — per-exercise rule classes
- `feedback/` — enrichment plug-ins
- `viz/` — dev rendering helpers

## Phase 2 (planned)

- Real-time / streaming mode
- Fitness-AQA pretrained-model integration
- YOLO11-pose backend
- REST API wrapper
- Beyond 5 exercises

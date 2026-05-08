# Sport Companion AI — Prototype Design

**Status**: Approved — ready for implementation planning
**Date**: 2026-05-08
**Phase**: Prototype (Phase 1)

## Context & Goals

Build a Python library prototype that takes a gym workout video as input, extracts skeleton/joint information, and evaluates whether the exercise was performed correctly. The library is the foundation for a real product, so the architecture must be clean, extensible, and ready to grow.

**Concrete scope:**
- 5 exercises: squat, deadlift, bench press, push-up, bicep curl
- Batch input (uploaded video file). Real-time streaming reserved for Phase 2.
- Pure-Python package, importable directly. Optional FastAPI wrapper deferred to Phase 2.
- Hybrid evaluation: rule-based per exercise (deterministic core) + optional LLM enrichment (NVIDIA NIM) for natural-language coaching feedback.

**Out of scope for this prototype** (architecture leaves hooks for them):
- Real-time/streaming mode
- Fitness-AQA pretrained model integration (squat AQA score)
- YOLO11-pose backend (MediaPipe is the default)
- REST API service
- Beyond 5 exercises
- Multi-person / form comparison

## Non-goals (explicit)

- Not a Java library. Original ask was "Java backend" but user confirmed the language was a soft constraint; pure Python is simpler given the entire ML/CV stack is Python-native.
- Not a sidecar architecture. Pose estimation runs in-process inside the Python package, not as a separate co-deployed sidecar service.
- Not a generic action classifier. The user picks the exercise (squat/deadlift/...); the library does not auto-detect which exercise is being performed in the prototype.

## Architecture

Single Python package `sport-companion-ai`, two consumption modes:

```
┌──────────────────────────────────────────────────────────────┐
│  Mode 1: In-process import (primary, prototype)              │
│    from sport_companion_ai import VideoAnalyzer              │
│    report = VideoAnalyzer().analyze("squat.mp4", "squat")    │
├──────────────────────────────────────────────────────────────┤
│  Mode 2: REST API (FastAPI wrapper, Phase 2)                 │
│    POST /analyze  { video, exercise }                        │
│    → JSON AnalysisReport                                     │
└──────────────────────────────────────────────────────────────┘
```

External dependency (optional, opt-in): NVIDIA NIM API for LLM enrichment (`build.nvidia.com`). Falls back gracefully to template-based feedback if unavailable.

### Package structure

```
sport-companion-ai/
├── pyproject.toml
├── sport_companion_ai/
│   ├── __init__.py
│   ├── analyzer.py            # VideoAnalyzer — entry point, orchestrate
│   ├── pose/
│   │   ├── extractor.py       # PoseExtractor Protocol + MediaPipeExtractor
│   │   └── schema.py          # Keypoint, Skeleton, Frame
│   ├── exercises/
│   │   ├── base.py            # ExerciseRule ABC
│   │   ├── squat.py
│   │   ├── deadlift.py
│   │   ├── bench.py
│   │   ├── pushup.py
│   │   └── bicep_curl.py
│   ├── geometry.py            # Pure functions: angles, vectors
│   ├── rep_detector.py        # Peak-detection rep finder (shared)
│   ├── feedback/
│   │   ├── enricher.py        # FeedbackEnricher Protocol
│   │   ├── template.py        # TemplateEnricher (default, hardcoded VN strings)
│   │   └── nim.py             # NvidiaNimEnricher (opt-in, NVIDIA NIM API)
│   ├── viz/                    # Optional dev/test rendering
│   │   └── render.py          # render_skeleton_png, render_overlay
│   ├── aqa/                    # Phase 2 hook (Fitness-AQA client)
│   ├── report.py              # AnalysisReport dataclass + JSON serialization
│   └── errors.py              # Exception hierarchy
├── api/                         # Phase 2 stub — empty in Phase 1; FastAPI wrapper added later
├── tests/
│   ├── fixtures/videos/
│   │   ├── squat_good_5reps.mp4
│   │   ├── squat_shallow_2reps.mp4
│   │   └── ...
│   ├── fixtures/manifest.yaml  # expected ranges per fixture
│   ├── test_geometry.py
│   ├── test_rep_detector.py
│   ├── test_squat_rule.py
│   ├── test_deadlift_rule.py
│   ├── test_bench_rule.py
│   ├── test_pushup_rule.py
│   ├── test_bicep_curl_rule.py
│   ├── test_analyzer.py
│   └── test_nim_enricher.py
└── examples/
    └── analyze_squat.py        # CLI demo
```

### Module boundaries

- `pose/` knows only "video → frames with skeletons". Does not know about exercises.
- `exercises/` knows only "frames with skeletons → reps + correctness". Does not know about pose models.
- `feedback/` knows only "structured report → enriched report". Does not know about exercises or pose.
- `analyzer.py` orchestrates. Contains no domain logic.
- `geometry.py` and `rep_detector.py` are pure (no state, no IO). Trivially testable.

## Data Flow

```
analyze("squat.mp4", exercise="squat", skeleton_output="keyframes")
   │
   ├─► (1) Read video → raw frames (cv2.VideoCapture)
   │
   ├─► (2) PoseExtractor.extract_batch(frames) → List[Frame]
   │       Frame { index, timestamp_ms, skeleton: Skeleton | None }
   │       Skeleton { keypoints: dict[KeypointName, Keypoint] }
   │       Keypoint { x, y, z, visibility }   # x,y normalized [0,1]
   │
   ├─► (3) ExerciseRule.detect_reps(frames) → List[Rep]
   │       Rep { index, start_idx, peak_idx, end_idx, primary_angle_series }
   │       (peak detection on primary angle: knee for squat, elbow for curl)
   │
   ├─► (4) For each rep: ExerciseRule.evaluate_rep(rep, frames) → RepEvaluation
   │       RepEvaluation { rep_index, score, passed, inconclusive,
   │                       issues: List[Issue], metrics: dict, keyframes }
   │
   ├─► (5) Aggregate → AnalysisReport
   │
   └─► (6) FeedbackEnricher.enrich(report) → AnalysisReport (with session_summary)
```

## Output Schema

```json
{
  "exercise": "squat",
  "version": "0.1.0",
  "pose_model": "mediapipe-blazepose-full",
  "enriched": false,

  "video": { "width": 1080, "height": 1920, "fps": 30, "duration_ms": 6000 },

  "skeleton_schema": {
    "keypoint_names": ["nose", "left_shoulder", "right_shoulder", "left_elbow", "..."],
    "edges": [["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"], "..."],
    "coordinate_space": "normalized"
  },

  "frames": [
    {
      "index": 0,
      "timestamp_ms": 0,
      "keypoints": {
        "nose":          { "x": 0.50, "y": 0.30, "z": 0.0, "visibility": 0.99 },
        "left_shoulder": { "x": 0.45, "y": 0.40, "z": 0.0, "visibility": 0.97 }
      }
    }
  ],

  "total_reps": 5,
  "passed_reps": 4,
  "avg_score": 78,
  "session_summary": null,

  "reps": [
    {
      "rep_index": 0,
      "score": 85,
      "passed": true,
      "inconclusive": false,
      "issues": [],
      "metrics": {
        "min_knee_angle": 92,
        "max_knee_angle": 168,
        "back_angle_at_bottom": 42,
        "knee_valgus_ratio": 0.05,
        "rep_duration_ms": 1800
      },
      "keyframes": { "start": 12, "peak": 45, "end": 78 }
    },
    {
      "rep_index": 2,
      "score": 50,
      "passed": false,
      "inconclusive": false,
      "issues": [
        {
          "code": "SQUAT_DEPTH_INSUFFICIENT",
          "severity": "HIGH",
          "message_vi": "Hạ chưa đủ sâu, đầu gối chỉ gập 105° (cần ≤ 95°)",
          "frame_indices": [142, 158],
          "recommendation": "Hạ thấp hông hơn cho đến khi đùi song song mặt đất"
        }
      ],
      "metrics": { "min_knee_angle": 105, "max_knee_angle": 170, "back_angle_at_bottom": 38 },
      "keyframes": { "start": 130, "peak": 150, "end": 170 }
    }
  ],

  "warnings": []
}
```

### Skeleton output sampling modes

The `skeleton_output` argument controls how many frames appear in `frames[]`:

| Mode | Frames in output | Use case | Approx size (30s @ 30fps) |
|---|---|---|---|
| `full` | All ~900 frames | Dev/test, animation playback | ~150 KB |
| `sampled` | ~5 fps subsample | Production preview | ~25 KB |
| `keyframes` (default) | Per-rep start/peak/end + frames referenced by issues | Production report | ~5 KB |
| `none` | (omitted) | When client does not need to render | ~0 KB |

References from `issues[].frame_indices` and `reps[].keyframes` are integer indices into `frames[]`.

## Exercise Rule Design

### Rule contract

```python
class ExerciseRule(ABC):
    name: str                    # e.g., "squat"
    primary_angle: str           # angle used for rep detection
    rep_threshold_low: float     # bottom of rep
    rep_threshold_high: float    # top of rep

    @abstractmethod
    def detect_reps(self, frames: list[Frame]) -> list[Rep]: ...

    @abstractmethod
    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation: ...
```

### Squat rule (illustrative full implementation)

| Rule | Condition | Severity | Score penalty |
|---|---|---|---|
| `SQUAT_DEPTH_INSUFFICIENT` | `min_knee_angle > 95°` | HIGH (>110°) / MED | -25 / -10 |
| `SQUAT_BACK_TOO_VERTICAL` | `back_angle_at_bottom < 30°` | LOW | -5 |
| `SQUAT_FORWARD_LEAN` | `back_angle_at_bottom > 60°` | MED | -15 |
| `SQUAT_KNEE_VALGUS` | `knee_valgus_ratio > 0.15` | HIGH | -20 |
| `SQUAT_TOO_FAST` | `rep_duration_ms < 800` | LOW | -5 |

A rep `passes` when `score >= 70` and no `HIGH` severity issue.

The penalty values and thresholds in the table above are initial estimates from literature and coaching heuristics. They must be empirically tuned against fixture videos during implementation. Tuning lives in each rule's class (constants at the top of `exercises/squat.py` etc.) so it does not require touching shared code.

### Rule sketch for the other 4 exercises

| Exercise | Primary angle | Core rules |
|---|---|---|
| **Deadlift** | hip | Back rounding, bar path away from body, knees locking too early, hip-hinge vs squat-pattern |
| **Bench press** | elbow | Elbow flare (>75°), asymmetric hand spacing, bar path drift, no chest contact |
| **Push-up** | elbow | Hip sag, hip pike, elbow flare, partial ROM |
| **Bicep curl** | elbow | Shoulder swing/cheat, elbow drift forward, short ROM, fast eccentric |

For Phase 1, each non-squat exercise ships with 2-3 core rules (the most important form errors). Additional rules can be added incrementally without touching shared code.

### Rep detector (shared)

Pure function in `rep_detector.py`:

```python
def detect_reps_by_peaks(
    angle_series: list[float],
    low_thresh: float,
    high_thresh: float,
    min_rep_duration_ms: int = 500,
    fps: int = 30,
) -> list[Rep]:
    """
    Smooth angle series, find pattern: high → low → high (one rep).
    Uses scipy.signal.find_peaks. Filters out reps shorter than threshold (noise).
    """
```

Used by all five exercises with different angle/threshold inputs.

### Geometry helpers (pure)

```python
def angle_3pt(a: Point, b: Point, c: Point) -> float: ...
def knee_angle(skel: Skeleton) -> float: ...
def back_angle(skel: Skeleton) -> float: ...
def knee_valgus_ratio(skel: Skeleton) -> float: ...
def angle_with_vertical(p1: Point, p2: Point) -> float: ...
```

All pure functions on numeric inputs — testable with concrete numbers, no fixtures needed.

## Feedback Enrichment

Default = template (free, deterministic). Opt-in NVIDIA NIM enricher rewrites Vietnamese messages and adds a `session_summary`.

```python
class FeedbackEnricher(Protocol):
    def enrich(self, report: AnalysisReport) -> AnalysisReport: ...

class TemplateEnricher:
    """Default. Hardcoded Vietnamese strings already populated by ExerciseRule.
    Returns the report unchanged. Leaves `report.enriched = False` and
    `report.session_summary = None` (no LLM-generated summary)."""

class NvidiaNimEnricher:
    def __init__(self, api_key: str, model: str = "meta/llama-3.3-70b-instruct",
                 timeout_s: float = 10, max_retries: int = 1): ...
    def enrich(self, report: AnalysisReport) -> AnalysisReport:
        # Build prompt from report.metrics + report.issues (codes, not raw VN strings)
        # POST to https://integrate.api.nvidia.com/v1/chat/completions
        # Parse response → set report.session_summary, optionally rewrite issue.message_vi
        # report.enriched = True
```

**Invariants any enricher MUST NOT violate:**
- Does not modify `score`, `passed`, `passed_reps`, `metrics`, or issue `code`
- Only LLM-backed enrichers set `enriched = True` (after the LLM call succeeds and produces text). `TemplateEnricher` always leaves `enriched = False`.
- On failure: append `ENRICHMENT_FAILED` warning, return report unmodified, `enriched = False`.

## Error Handling

### Hard errors (raise)

```python
class SportCompanionError(Exception): ...
class VideoReadError(SportCompanionError): ...
class UnsupportedExerciseError(SportCompanionError): ...
class PoseExtractionError(SportCompanionError): ...
class EnricherError(SportCompanionError): ...
```

### Soft warnings (returned in `report.warnings[]`)

| Code | Trigger | Action |
|---|---|---|
| `LOW_POSE_CONFIDENCE` | >30% frames have avg visibility < 0.5 | Continue, warn |
| `PARTIAL_BODY_VISIBLE` | Any keypoint required by the active rule has visibility < 0.5 in >20% of rep frames | Skip dependent rules, mark affected reps `inconclusive` |
| `NO_REPS_DETECTED` | Zero reps detected | Empty report + warning |
| `TOO_MANY_PEOPLE` | >1 person in frame | Pick largest bbox closest to center |
| `VIDEO_TOO_SHORT` | Duration < 3s | Continue, warn |
| `VIDEO_TOO_LONG` | Duration > 5min | Continue, warn |
| `LOW_FPS` | Source fps < 15 | Continue, warn (timing accuracy degraded) |
| `ENRICHMENT_FAILED` | NIM call failed after retries | Fall back to template silently |

### Per-rep `inconclusive`

A rep with insufficient data sets `inconclusive = true`, `score = null`, `passed = null`, with `inconclusive_reason`. Other reps in the same video still evaluate normally.

### Resilience

- Pose extractor: retry once on batch failure, then `PoseExtractionError`
- NIM enricher: 10s timeout, 1 retry with exponential backoff, then fallback
- Video read: verify frame count > 0 before processing
- Memory: chunk-process videos > 60s (batch size 300 frames)

## Testing Strategy

### Tier 1 — Pure unit tests (~70%)

`geometry.py`, `rep_detector.py`, every `ExerciseRule.evaluate_rep` against fake skeletons. Helper `make_fake_squat_frames(min_knee_angle, back_angle, ...)` synthesizes skeletons with target angles, no video required.

### Tier 2 — Integration tests with fixture videos (~20%)

Real video fixtures stored in `tests/fixtures/videos/`. Each ≤ 15s, ≤ 5MB. Filename convention: `{exercise}_{quality}_{nreps}reps.mp4`. Expected outcomes recorded in `tests/fixtures/manifest.yaml` (avoid magic numbers in tests).

Marked `@pytest.mark.integration` so PR CI can skip when fixtures are unavailable; full suite runs nightly.

### Tier 3 — LLM enricher tests

- **Smoke test** (`@pytest.mark.requires_nim_key`): real NIM call, assert structural validity (non-empty summary, schema preserved). Skipped without `NVIDIA_API_KEY` env var.
- **Mocked test** (default): mock NIM HTTP client to test retry, fallback, and warning behavior deterministically.

### CI matrix

```
PR check         : pytest -m "not integration and not requires_nim_key"   # fast
Nightly / main   : pytest -m integration
Manual / nightly : pytest -m requires_nim_key  (with secret)
```

### Coverage targets

| Module | Target |
|---|---|
| `geometry.py`, `rep_detector.py` | 100% |
| `exercises/*.py` | ≥ 90%, ≥ 3 cases per rule (good / bad / inconclusive) |
| `analyzer.py`, `pose/` | ≥ 70% |
| `feedback/nim.py` | smoke + mocked |

## Phase 1 Deliverables

1. Python package with the structure above
2. 5 exercise rule files (squat full, others with 2-3 core rules each)
3. ≥ 5 video fixtures (1 per exercise) + `manifest.yaml`
4. Test suite: ~30-50 unit tests + ~5 integration tests
5. README with install/quickstart
6. CLI demo (`examples/analyze_squat.py`) running end-to-end
7. Dockerfile for the package
8. `viz/` rendering helper for dev inspection (PNG of skeleton on a frame)

## Phase 2 Hooks (already designed for)

- Real-time mode: `analyzer.analyze_stream(frame_iter)` method (interface allows it; not implemented)
- Fitness-AQA: `aqa/` directory reserved; `AqaClient` Protocol can plug into squat rule
- YOLO11-pose: `PoseExtractor` Protocol allows swap with no rule changes
- REST API: `api/` directory is reserved (empty in Phase 1, populated in Phase 2 with a FastAPI wrapper around `VideoAnalyzer`)
- Additional exercises: each exercise is an isolated file under `exercises/`; registry pattern allows dynamic registration
- LLM-driven training plans: enricher result already structured; downstream analytics can consume the JSON history

## Estimated Effort (Phase 1)

| Workstream | Days |
|---|---|
| Pose pipeline + analyzer skeleton | 1-2 |
| Squat rule (full) + rep detector + geometry | 2-3 |
| 4 remaining exercises (2-3 rules each) | 2-3 |
| Report serialization + sampling + viz | 1 |
| NIM enricher + fallback + warnings | 1 |
| Tests + fixtures | 2-3 |
| Docker + docs + demo | 1 |
| **Total** | **~10-14 days (1 dev full-time)** |

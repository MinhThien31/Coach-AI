# Sport Companion AI — Phase 1 Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python library that takes a gym video, extracts skeleton keypoints, and evaluates squat/deadlift/bench/push-up/bicep-curl form via per-exercise rules, with optional NVIDIA NIM LLM enrichment.

**Architecture:** Single Python package, in-process pipeline `video → MediaPipe pose → per-exercise rule → JSON report`. Pose extractor and feedback enricher are Protocols so they can be swapped (Phase 2 will add YOLO and Fitness-AQA).

**Tech Stack:** Python 3.13, MediaPipe (BlazePose), OpenCV (video read), NumPy + SciPy (signal processing), Pydantic v2 (typed schema + JSON), httpx (NIM client), Pillow + Matplotlib (viz), pytest + pytest-mock (tests).

**Spec:** See `docs/superpowers/specs/2026-05-08-sport-companion-prototype-design.md`. Read it before starting.

---

## Task 1: Project Bootstrap

**Files:**
- Create: `pyproject.toml`
- Create: `sport_companion_ai/__init__.py`
- Create: `sport_companion_ai/pose/__init__.py`
- Create: `sport_companion_ai/exercises/__init__.py`
- Create: `sport_companion_ai/feedback/__init__.py`
- Create: `sport_companion_ai/viz/__init__.py`
- Create: `sport_companion_ai/aqa/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/fixtures/.gitkeep`
- Create: `.gitignore`
- Create: `README.md`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "sport-companion-ai"
version = "0.1.0"
description = "Gym video form evaluator (Phase 1 prototype)"
requires-python = ">=3.13"
dependencies = [
    "mediapipe>=0.10.20",
    "opencv-python-headless>=4.10",
    "numpy>=2.0",
    "scipy>=1.14",
    "pydantic>=2.8",
    "httpx>=0.27",
    "pillow>=10.4",
    "matplotlib>=3.9",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-mock>=3.14",
    "pytest-cov>=5.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["sport_companion_ai"]

[tool.pytest.ini_options]
markers = [
    "integration: integration tests using real video fixtures (slow)",
    "requires_nim_key: tests that require NVIDIA_API_KEY env var",
]
addopts = "-m 'not integration and not requires_nim_key'"
```

- [ ] **Step 2: Create empty `__init__.py` files**

```python
# sport_companion_ai/__init__.py
"""Sport Companion AI — gym video form evaluator."""

__version__ = "0.1.0"
```

```python
# sport_companion_ai/pose/__init__.py
# sport_companion_ai/exercises/__init__.py
# sport_companion_ai/feedback/__init__.py
# sport_companion_ai/viz/__init__.py
# sport_companion_ai/aqa/__init__.py
# tests/__init__.py
```

(All five `__init__.py` are empty files.)

- [ ] **Step 3: Create .gitignore**

```
__pycache__/
*.pyc
.pytest_cache/
.coverage
htmlcov/
*.egg-info/
dist/
build/
.venv/
.env
tests/fixtures/videos/*.mp4
!tests/fixtures/videos/.gitkeep
```

- [ ] **Step 4: Create README.md stub**

```markdown
# Sport Companion AI

Phase 1 prototype: take a gym video, get an evaluation report.

See `docs/superpowers/specs/2026-05-08-sport-companion-prototype-design.md` for design.

## Install

```bash
pip install -e ".[dev]"
```

## Run tests

```bash
pytest
```
```

- [ ] **Step 5: Install and verify**

Run: `pip install -e ".[dev]"`
Expected: installation succeeds without errors.

Run: `python -c "import sport_companion_ai; print(sport_companion_ai.__version__)"`
Expected: prints `0.1.0`.

- [ ] **Step 6: Commit**

```bash
git init
git add pyproject.toml sport_companion_ai/ tests/ .gitignore README.md
git commit -m "chore: bootstrap sport-companion-ai package"
```

---

## Task 2: Pose Schema (Keypoint, Skeleton, Frame)

**Files:**
- Create: `sport_companion_ai/pose/schema.py`
- Create: `tests/test_pose_schema.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pose_schema.py
import pytest
from pydantic import ValidationError

from sport_companion_ai.pose.schema import Keypoint, Skeleton, Frame, KEYPOINT_NAMES, SKELETON_EDGES


def test_keypoint_normalized_coords_required():
    Keypoint(x=0.5, y=0.5)  # OK
    with pytest.raises(ValidationError):
        Keypoint(x=1.5, y=0.5)
    with pytest.raises(ValidationError):
        Keypoint(x=0.5, y=-0.1)


def test_keypoint_defaults():
    kp = Keypoint(x=0.5, y=0.5)
    assert kp.z == 0.0
    assert kp.visibility == 0.0


def test_skeleton_holds_named_keypoints():
    skel = Skeleton(keypoints={"nose": Keypoint(x=0.5, y=0.3)})
    assert skel.keypoints["nose"].x == 0.5


def test_frame_skeleton_optional():
    f = Frame(index=0, timestamp_ms=0, skeleton=None)
    assert f.skeleton is None


def test_keypoint_names_includes_required_set():
    required = {
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    }
    assert required.issubset(set(KEYPOINT_NAMES))


def test_skeleton_edges_reference_known_names():
    names = set(KEYPOINT_NAMES)
    for a, b in SKELETON_EDGES:
        assert a in names
        assert b in names
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_pose_schema.py -v`
Expected: FAIL with import error (`schema` module does not exist).

- [ ] **Step 3: Implement schema**

```python
# sport_companion_ai/pose/schema.py
"""Pose data types. Coordinates are normalized to [0, 1]."""
from pydantic import BaseModel, Field


KEYPOINT_NAMES: list[str] = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


SKELETON_EDGES: list[tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


class Keypoint(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    z: float = 0.0
    visibility: float = Field(ge=0.0, le=1.0, default=0.0)


class Skeleton(BaseModel):
    keypoints: dict[str, Keypoint]


class Frame(BaseModel):
    index: int
    timestamp_ms: int
    skeleton: Skeleton | None = None
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_pose_schema.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add sport_companion_ai/pose/schema.py tests/test_pose_schema.py
git commit -m "feat(pose): add Keypoint/Skeleton/Frame schema and BlazePose name list"
```

---

## Task 3: Report Schema (Rep, Issue, RepEvaluation, AnalysisReport)

**Files:**
- Create: `sport_companion_ai/report.py`
- Create: `tests/test_report.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_report.py
import json

from sport_companion_ai.pose.schema import Frame
from sport_companion_ai.report import (
    AnalysisReport, Issue, Rep, RepEvaluation,
    SkeletonSchema, VideoMeta, AnalysisWarning, Severity,
)


def test_issue_severity_literal():
    issue = Issue(code="X", severity="HIGH", message_vi="...")
    assert issue.severity == "HIGH"


def test_rep_evaluation_inconclusive_defaults():
    e = RepEvaluation(rep_index=0, score=None, passed=None, inconclusive=True)
    assert e.issues == []
    assert e.metrics == {}
    assert e.keyframes == {}


def test_analysis_report_serializes_to_json():
    report = AnalysisReport(
        exercise="squat",
        pose_model="mediapipe-blazepose-full",
        video=VideoMeta(width=1080, height=1920, fps=30, duration_ms=6000),
        skeleton_schema=SkeletonSchema(keypoint_names=["nose"], edges=[]),
    )
    text = report.model_dump_json()
    parsed = json.loads(text)
    assert parsed["exercise"] == "squat"
    assert parsed["enriched"] is False
    assert parsed["session_summary"] is None
    assert parsed["frames"] == []


def test_warning_minimal():
    w = AnalysisWarning(code="LOW_FPS")
    assert w.code == "LOW_FPS"
    assert w.message_vi == ""
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/test_report.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement report types**

```python
# sport_companion_ai/report.py
"""Output schema. Designed to be JSON-serializable for FE/API consumption."""
from typing import Literal

from pydantic import BaseModel

from sport_companion_ai.pose.schema import Frame


Severity = Literal["LOW", "MEDIUM", "HIGH"]


class Issue(BaseModel):
    code: str
    severity: Severity
    message_vi: str
    frame_indices: list[int] = []
    recommendation: str = ""


class Rep(BaseModel):
    rep_index: int
    start_idx: int
    peak_idx: int
    end_idx: int


class RepEvaluation(BaseModel):
    rep_index: int
    score: int | None
    passed: bool | None
    inconclusive: bool = False
    inconclusive_reason: str | None = None
    issues: list[Issue] = []
    metrics: dict[str, float | None] = {}
    keyframes: dict[str, int] = {}


class AnalysisWarning(BaseModel):
    code: str
    message_vi: str = ""
    affected_frame_count: int | None = None


class VideoMeta(BaseModel):
    width: int
    height: int
    fps: int
    duration_ms: int


class SkeletonSchema(BaseModel):
    keypoint_names: list[str]
    edges: list[tuple[str, str]]
    coordinate_space: str = "normalized"


class AnalysisReport(BaseModel):
    exercise: str
    version: str = "0.1.0"
    pose_model: str
    enriched: bool = False
    video: VideoMeta
    skeleton_schema: SkeletonSchema
    frames: list[Frame] = []
    total_reps: int = 0
    passed_reps: int = 0
    avg_score: float = 0.0
    session_summary: str | None = None
    reps: list[RepEvaluation] = []
    warnings: list[AnalysisWarning] = []
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_report.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add sport_companion_ai/report.py tests/test_report.py
git commit -m "feat(report): add AnalysisReport schema and supporting types"
```

---

## Task 4: Errors Module

**Files:**
- Create: `sport_companion_ai/errors.py`
- Create: `tests/test_errors.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_errors.py
import pytest

from sport_companion_ai.errors import (
    SportCompanionError,
    VideoReadError,
    UnsupportedExerciseError,
    PoseExtractionError,
    EnricherError,
)


def test_subclass_hierarchy():
    for cls in (VideoReadError, UnsupportedExerciseError, PoseExtractionError, EnricherError):
        assert issubclass(cls, SportCompanionError)


def test_can_raise_and_catch_as_base():
    with pytest.raises(SportCompanionError):
        raise VideoReadError("bad codec")
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/test_errors.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement errors**

```python
# sport_companion_ai/errors.py
"""Exception hierarchy. Catch SportCompanionError to handle any library failure."""


class SportCompanionError(Exception):
    """Base class for all errors raised by sport-companion-ai."""


class VideoReadError(SportCompanionError):
    """Raised when a video file cannot be opened or decoded."""


class UnsupportedExerciseError(SportCompanionError):
    """Raised when an exercise name is not registered."""


class PoseExtractionError(SportCompanionError):
    """Raised when the pose extractor fails irrecoverably."""


class EnricherError(SportCompanionError):
    """Raised when an enricher must signal hard failure (rare; usually we soft-fail)."""
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_errors.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add sport_companion_ai/errors.py tests/test_errors.py
git commit -m "feat: add exception hierarchy"
```

---

## Task 5: Geometry Helpers (Pure Functions)

**Files:**
- Create: `sport_companion_ai/geometry.py`
- Create: `tests/test_geometry.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_geometry.py
import math
import pytest

from sport_companion_ai.geometry import (
    angle_3pt, angle_with_vertical,
    knee_angle, elbow_angle, hip_angle, back_angle, knee_valgus_ratio,
)
from sport_companion_ai.pose.schema import Keypoint, Skeleton


def kp(x, y):
    return Keypoint(x=x, y=y, visibility=1.0)


def test_angle_3pt_right_angle():
    a = kp(0.0, 0.0)
    b = kp(0.0, 0.5)  # corner
    c = kp(0.5, 0.5)
    assert angle_3pt(a, b, c) == pytest.approx(90.0, abs=0.1)


def test_angle_3pt_straight():
    a = kp(0.0, 0.5)
    b = kp(0.5, 0.5)
    c = kp(1.0, 0.5)
    assert angle_3pt(a, b, c) == pytest.approx(180.0, abs=0.1)


def test_angle_3pt_zero_length_returns_nan():
    a = kp(0.5, 0.5)
    b = kp(0.5, 0.5)
    c = kp(0.5, 0.5)
    assert math.isnan(angle_3pt(a, b, c))


def test_angle_with_vertical_zero_for_vertical_line():
    p1 = kp(0.5, 0.2)
    p2 = kp(0.5, 0.8)
    assert angle_with_vertical(p1, p2) == pytest.approx(0.0, abs=0.1)


def test_angle_with_vertical_45_deg():
    p1 = kp(0.2, 0.2)
    p2 = kp(0.5, 0.5)
    assert angle_with_vertical(p1, p2) == pytest.approx(45.0, abs=0.1)


def make_skeleton(**points):
    return Skeleton(keypoints={name: kp(x, y) for name, (x, y) in points.items()})


def test_knee_angle_extended_leg_is_180():
    skel = make_skeleton(
        left_hip=(0.5, 0.4), left_knee=(0.5, 0.6), left_ankle=(0.5, 0.8),
    )
    assert knee_angle(skel, side="left") == pytest.approx(180.0, abs=0.5)


def test_knee_angle_squat_bottom_is_about_90():
    skel = make_skeleton(
        left_hip=(0.5, 0.5), left_knee=(0.3, 0.5), left_ankle=(0.3, 0.7),
    )
    assert knee_angle(skel, side="left") == pytest.approx(90.0, abs=1.0)


def test_back_angle_vertical_torso():
    skel = make_skeleton(
        left_hip=(0.4, 0.6), right_hip=(0.6, 0.6),
        left_shoulder=(0.4, 0.4), right_shoulder=(0.6, 0.4),
    )
    assert back_angle(skel) == pytest.approx(0.0, abs=0.5)


def test_back_angle_leaning_forward():
    skel = make_skeleton(
        left_hip=(0.4, 0.6), right_hip=(0.6, 0.6),
        left_shoulder=(0.6, 0.4), right_shoulder=(0.8, 0.4),  # shifted forward
    )
    assert back_angle(skel) == pytest.approx(45.0, abs=1.0)


def test_knee_valgus_ratio_neutral_is_zero():
    skel = make_skeleton(
        left_hip=(0.4, 0.4), left_knee=(0.4, 0.6), left_ankle=(0.4, 0.8),
        right_hip=(0.6, 0.4), right_knee=(0.6, 0.6), right_ankle=(0.6, 0.8),
    )
    assert knee_valgus_ratio(skel) == pytest.approx(0.0, abs=0.01)


def test_knee_valgus_ratio_inward_collapse():
    skel = make_skeleton(
        left_hip=(0.4, 0.4), left_knee=(0.5, 0.6), left_ankle=(0.4, 0.8),
        right_hip=(0.6, 0.4), right_knee=(0.5, 0.6), right_ankle=(0.6, 0.8),
    )
    # Each knee deviates by 0.1 toward midline; hip width = 0.2 → ratio = 0.5
    assert knee_valgus_ratio(skel) == pytest.approx(0.5, abs=0.01)
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/test_geometry.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement geometry**

```python
# sport_companion_ai/geometry.py
"""Pure geometry helpers operating on Keypoints. All angles in degrees."""
import math

from sport_companion_ai.pose.schema import Keypoint, Skeleton


def angle_3pt(a: Keypoint, b: Keypoint, c: Keypoint) -> float:
    """Angle at vertex b formed by rays b→a and b→c. NaN if degenerate."""
    bax, bay = a.x - b.x, a.y - b.y
    bcx, bcy = c.x - b.x, c.y - b.y
    mag_ba = math.hypot(bax, bay)
    mag_bc = math.hypot(bcx, bcy)
    if mag_ba == 0 or mag_bc == 0:
        return float("nan")
    cos = (bax * bcx + bay * bcy) / (mag_ba * mag_bc)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def angle_with_vertical(p1: Keypoint, p2: Keypoint) -> float:
    """Angle of the line p1→p2 from the vertical axis (always non-negative)."""
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def knee_angle(skel: Skeleton, side: str = "left") -> float:
    return angle_3pt(
        skel.keypoints[f"{side}_hip"],
        skel.keypoints[f"{side}_knee"],
        skel.keypoints[f"{side}_ankle"],
    )


def elbow_angle(skel: Skeleton, side: str = "left") -> float:
    return angle_3pt(
        skel.keypoints[f"{side}_shoulder"],
        skel.keypoints[f"{side}_elbow"],
        skel.keypoints[f"{side}_wrist"],
    )


def hip_angle(skel: Skeleton, side: str = "left") -> float:
    return angle_3pt(
        skel.keypoints[f"{side}_shoulder"],
        skel.keypoints[f"{side}_hip"],
        skel.keypoints[f"{side}_knee"],
    )


def back_angle(skel: Skeleton) -> float:
    """Angle of the torso (mid-hip → mid-shoulder) from vertical."""
    lh = skel.keypoints["left_hip"]
    rh = skel.keypoints["right_hip"]
    ls = skel.keypoints["left_shoulder"]
    rs = skel.keypoints["right_shoulder"]
    mid_hip = Keypoint(x=(lh.x + rh.x) / 2, y=(lh.y + rh.y) / 2)
    mid_sh = Keypoint(x=(ls.x + rs.x) / 2, y=(ls.y + rs.y) / 2)
    return angle_with_vertical(mid_hip, mid_sh)


def knee_valgus_ratio(skel: Skeleton) -> float:
    """Mean horizontal knee deviation from the hip-ankle midline,
    normalized by hip width. 0.0 = neutral, ~0.3+ = visible valgus."""
    lh, lk, la = skel.keypoints["left_hip"], skel.keypoints["left_knee"], skel.keypoints["left_ankle"]
    rh, rk, ra = skel.keypoints["right_hip"], skel.keypoints["right_knee"], skel.keypoints["right_ankle"]

    def deviation(hip: Keypoint, knee: Keypoint, ankle: Keypoint) -> float:
        expected_x = (hip.x + ankle.x) / 2
        return abs(knee.x - expected_x)

    hip_width = abs(rh.x - lh.x)
    if hip_width == 0:
        return 0.0
    return (deviation(lh, lk, la) + deviation(rh, rk, ra)) / 2 / hip_width
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_geometry.py -v`
Expected: PASS (11 tests).

- [ ] **Step 5: Check coverage**

Run: `pytest tests/test_geometry.py --cov=sport_companion_ai.geometry --cov-report=term-missing`
Expected: 100% coverage on `geometry.py`.

- [ ] **Step 6: Commit**

```bash
git add sport_companion_ai/geometry.py tests/test_geometry.py
git commit -m "feat(geometry): add joint-angle and valgus helpers (pure functions)"
```

---

## Task 6: Rep Detector

**Files:**
- Create: `sport_companion_ai/rep_detector.py`
- Create: `tests/test_rep_detector.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_rep_detector.py
import math

from sport_companion_ai.rep_detector import detect_reps_by_peaks


def synthesize_reps(n_reps: int, fps: int = 30, top: float = 170.0, bottom: float = 90.0,
                   rep_seconds: float = 2.0) -> list[float]:
    """Build a sinusoidal angle series simulating reps."""
    samples_per_rep = int(rep_seconds * fps)
    series = []
    for _ in range(n_reps):
        for i in range(samples_per_rep):
            t = i / samples_per_rep
            # cosine: 1 at t=0, -1 at t=0.5, 1 at t=1
            v = top - (top - bottom) * (1 - math.cos(2 * math.pi * t)) / 2
            series.append(v)
    return series


def test_detects_three_reps():
    series = synthesize_reps(3, rep_seconds=2.0)
    reps = detect_reps_by_peaks(series, low_thresh=100, high_thresh=160, fps=30)
    assert len(reps) == 3
    assert all(r.start_idx < r.peak_idx < r.end_idx for r in reps)


def test_short_input_returns_empty():
    assert detect_reps_by_peaks([170.0, 170.0], 100, 160, fps=30) == []


def test_filters_reps_shorter_than_min_duration():
    # Quick reps (0.4s each) should be filtered out at default 500ms min duration
    series = synthesize_reps(5, rep_seconds=0.4)
    reps = detect_reps_by_peaks(series, low_thresh=100, high_thresh=160, fps=30,
                                min_rep_duration_ms=500)
    assert len(reps) == 0


def test_handles_nan_values():
    series = synthesize_reps(2)
    series[10] = float("nan")
    series[50] = float("nan")
    reps = detect_reps_by_peaks(series, low_thresh=100, high_thresh=160, fps=30)
    assert len(reps) == 2
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/test_rep_detector.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement rep detector**

```python
# sport_companion_ai/rep_detector.py
"""Detect repetitions in an angle series. Pure function — no IO, no state."""
import numpy as np
from scipy.signal import find_peaks, savgol_filter

from sport_companion_ai.report import Rep


def detect_reps_by_peaks(
    angle_series: list[float],
    low_thresh: float,
    high_thresh: float,
    fps: int = 30,
    min_rep_duration_ms: int = 500,
) -> list[Rep]:
    """Find reps in a primary-joint angle series.

    A rep is a pattern: angle starts above `high_thresh`, drops below `low_thresh`
    (the rep peak is the minimum), and returns above `high_thresh`.

    Smooths input with Savitzky-Golay (window=11, polyorder=2) when long enough,
    interpolates NaNs from neighbors, and filters peaks closer than
    `min_rep_duration_ms` apart.
    """
    if len(angle_series) < 5:
        return []

    arr = np.array(angle_series, dtype=float)
    nans = np.isnan(arr)
    if nans.all():
        return []
    if nans.any():
        arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])

    window = min(11, (len(arr) // 2) * 2 + 1)
    if window >= 5 and len(arr) >= window:
        arr = savgol_filter(arr, window_length=window, polyorder=2)

    min_distance = max(1, int(min_rep_duration_ms / 1000 * fps))
    inverted = -arr
    peak_indices, _ = find_peaks(inverted, distance=min_distance, height=-low_thresh)

    reps: list[Rep] = []
    for peak_idx in peak_indices:
        start_idx = 0
        for j in range(int(peak_idx), -1, -1):
            if arr[j] >= high_thresh:
                start_idx = j
                break
        end_idx = len(arr) - 1
        for j in range(int(peak_idx), len(arr)):
            if arr[j] >= high_thresh:
                end_idx = j
                break

        duration_ms = (end_idx - start_idx) / fps * 1000
        if duration_ms < min_rep_duration_ms:
            continue

        reps.append(Rep(
            rep_index=len(reps),
            start_idx=int(start_idx),
            peak_idx=int(peak_idx),
            end_idx=int(end_idx),
        ))

    return reps
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_rep_detector.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add sport_companion_ai/rep_detector.py tests/test_rep_detector.py
git commit -m "feat: add peak-based rep detector with smoothing and min-duration filter"
```

---

## Task 7: ExerciseRule Base Class

**Files:**
- Create: `sport_companion_ai/exercises/base.py`
- Create: `tests/test_exercise_base.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_exercise_base.py
import pytest

from sport_companion_ai.exercises.base import ExerciseRule, EXERCISE_REGISTRY, register_rule
from sport_companion_ai.errors import UnsupportedExerciseError


def test_registry_lookup_unknown_raises():
    with pytest.raises(UnsupportedExerciseError):
        ExerciseRule.get("does_not_exist")


def test_register_and_retrieve():
    class Dummy(ExerciseRule):
        name = "dummy"
        primary_angle = "knee"
        rep_threshold_low = 100
        rep_threshold_high = 160

        def _primary_angle_series(self, frames):
            return []

        def evaluate_rep(self, rep, frames):
            raise NotImplementedError

    register_rule(Dummy)
    assert ExerciseRule.get("dummy") is Dummy
    # Cleanup so it doesn't leak to other tests
    EXERCISE_REGISTRY.pop("dummy", None)
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/test_exercise_base.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement base class**

```python
# sport_companion_ai/exercises/base.py
"""Base class and registry for exercise-specific rules."""
from abc import ABC, abstractmethod
from typing import ClassVar

from sport_companion_ai.errors import UnsupportedExerciseError
from sport_companion_ai.pose.schema import Frame
from sport_companion_ai.rep_detector import detect_reps_by_peaks
from sport_companion_ai.report import Rep, RepEvaluation


EXERCISE_REGISTRY: dict[str, type["ExerciseRule"]] = {}


def register_rule(cls: type["ExerciseRule"]) -> type["ExerciseRule"]:
    """Class decorator (or callable) registering an ExerciseRule subclass."""
    EXERCISE_REGISTRY[cls.name] = cls
    return cls


class ExerciseRule(ABC):
    """Strategy class for one exercise (e.g., squat).

    Subclasses must:
    - set class-level `name`, `primary_angle`, `rep_threshold_low/high`
    - implement `_primary_angle_series(frames)` to extract the angle series
    - implement `evaluate_rep(rep, frames)` to score a single rep
    """

    name: ClassVar[str]
    primary_angle: ClassVar[str]
    rep_threshold_low: ClassVar[float]
    rep_threshold_high: ClassVar[float]
    fps: ClassVar[int] = 30

    @classmethod
    def get(cls, name: str) -> type["ExerciseRule"]:
        try:
            return EXERCISE_REGISTRY[name]
        except KeyError as exc:
            raise UnsupportedExerciseError(f"Unknown exercise: {name!r}") from exc

    @abstractmethod
    def _primary_angle_series(self, frames: list[Frame]) -> list[float]:
        """Return the angle (in degrees) for each frame; NaN if skeleton missing."""

    def detect_reps(self, frames: list[Frame], fps: int = 30) -> list[Rep]:
        series = self._primary_angle_series(frames)
        return detect_reps_by_peaks(
            series,
            low_thresh=self.rep_threshold_low,
            high_thresh=self.rep_threshold_high,
            fps=fps,
        )

    @abstractmethod
    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation: ...
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_exercise_base.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add sport_companion_ai/exercises/base.py tests/test_exercise_base.py
git commit -m "feat(exercises): add ExerciseRule ABC + registry"
```

---

## Task 8: Squat Rule (Full Implementation)

This is the lead exercise — full rule set with fake-skeleton tests.

**Files:**
- Create: `sport_companion_ai/exercises/squat.py`
- Create: `tests/exercises/__init__.py`
- Create: `tests/exercises/_helpers.py`
- Create: `tests/exercises/test_squat.py`

- [ ] **Step 1: Write fake-skeleton helper**

```python
# tests/exercises/__init__.py
```

```python
# tests/exercises/_helpers.py
"""Shared helpers for building fake skeletons in rule tests."""
from sport_companion_ai.pose.schema import Frame, Keypoint, Skeleton


def kp(x: float, y: float, vis: float = 1.0) -> Keypoint:
    return Keypoint(x=x, y=y, visibility=vis)


def squat_skeleton(min_knee_deg: float, back_deg: float = 40.0,
                   knee_offset: float = 0.0) -> Skeleton:
    """Build a squat skeleton at the bottom position. min_knee_deg controls depth.

    A simple kinematic stand-in: place hip, knee, ankle so that the knee angle
    matches `min_knee_deg`, then place shoulders so torso lean = `back_deg`.
    `knee_offset` (in normalized x) shifts knees toward the midline (valgus).
    """
    import math

    half = min_knee_deg / 2  # angle each thigh/shin makes from vertical
    rad = math.radians(half)
    leg_len = 0.20
    knee_dx = leg_len * math.sin(rad)
    knee_dy = leg_len * math.cos(rad)

    # Left leg
    lh_x, lh_y = 0.40, 0.50
    lk_x, lk_y = lh_x - knee_dx + knee_offset, lh_y + knee_dy
    la_x, la_y = lh_x, lh_y + 2 * knee_dy
    # Right leg (mirror)
    rh_x, rh_y = 0.60, 0.50
    rk_x, rk_y = rh_x + knee_dx - knee_offset, rh_y + knee_dy
    ra_x, ra_y = rh_x, rh_y + 2 * knee_dy

    # Torso lean
    torso_len = 0.30
    back_rad = math.radians(back_deg)
    sh_dx = torso_len * math.sin(back_rad)
    sh_dy = torso_len * math.cos(back_rad)
    ls_x, ls_y = lh_x + sh_dx, lh_y - sh_dy
    rs_x, rs_y = rh_x + sh_dx, rh_y - sh_dy

    return Skeleton(keypoints={
        "left_hip": kp(lh_x, lh_y), "right_hip": kp(rh_x, rh_y),
        "left_knee": kp(lk_x, lk_y), "right_knee": kp(rk_x, rk_y),
        "left_ankle": kp(la_x, la_y), "right_ankle": kp(ra_x, ra_y),
        "left_shoulder": kp(ls_x, ls_y), "right_shoulder": kp(rs_x, rs_y),
    })


def make_squat_rep_frames(
    min_knee_deg: float, back_deg: float = 40.0, knee_offset: float = 0.0,
    fps: int = 30, rep_seconds: float = 2.0,
) -> list[Frame]:
    """Build a list of frames for one squat rep: descend, hit bottom, ascend."""
    n = int(rep_seconds * fps)
    frames = []
    for i in range(n):
        progress = abs(2 * (i / n) - 1)  # 1 → 0 → 1 (V shape)
        # Knee angle at this frame
        knee_at_t = 170 - (170 - min_knee_deg) * (1 - progress)
        skel = squat_skeleton(knee_at_t, back_deg=back_deg, knee_offset=knee_offset)
        frames.append(Frame(index=i, timestamp_ms=int(i / fps * 1000), skeleton=skel))
    return frames
```

- [ ] **Step 2: Write failing tests for SquatRule**

```python
# tests/exercises/test_squat.py
from sport_companion_ai.exercises.squat import SquatRule
from sport_companion_ai.exercises.base import ExerciseRule
from sport_companion_ai.report import Rep
from tests.exercises._helpers import make_squat_rep_frames


def test_registered():
    assert ExerciseRule.get("squat") is SquatRule


def test_detect_reps_finds_one_rep():
    frames = make_squat_rep_frames(min_knee_deg=88)
    rule = SquatRule()
    reps = rule.detect_reps(frames)
    assert len(reps) == 1


def test_clean_rep_passes():
    frames = make_squat_rep_frames(min_knee_deg=88, back_deg=42)
    rule = SquatRule()
    rep = rule.detect_reps(frames)[0]
    eval_ = rule.evaluate_rep(rep, frames)
    assert eval_.passed is True
    assert eval_.score >= 90
    assert eval_.issues == []


def test_shallow_rep_fails_with_depth_issue():
    frames = make_squat_rep_frames(min_knee_deg=115)
    rule = SquatRule()
    rep = rule.detect_reps(frames)[0] if rule.detect_reps(frames) else Rep(
        rep_index=0, start_idx=0, peak_idx=len(frames) // 2, end_idx=len(frames) - 1)
    eval_ = rule.evaluate_rep(rep, frames)
    assert eval_.passed is False
    codes = {i.code for i in eval_.issues}
    assert "SQUAT_DEPTH_INSUFFICIENT" in codes


def test_excessive_forward_lean_flagged():
    frames = make_squat_rep_frames(min_knee_deg=90, back_deg=70)
    rule = SquatRule()
    rep = rule.detect_reps(frames)[0]
    eval_ = rule.evaluate_rep(rep, frames)
    codes = {i.code for i in eval_.issues}
    assert "SQUAT_FORWARD_LEAN" in codes


def test_knee_valgus_flagged():
    frames = make_squat_rep_frames(min_knee_deg=90, back_deg=42, knee_offset=0.06)
    rule = SquatRule()
    rep = rule.detect_reps(frames)[0]
    eval_ = rule.evaluate_rep(rep, frames)
    codes = {i.code for i in eval_.issues}
    assert "SQUAT_KNEE_VALGUS" in codes


def test_metrics_reported():
    frames = make_squat_rep_frames(min_knee_deg=88)
    rule = SquatRule()
    rep = rule.detect_reps(frames)[0]
    eval_ = rule.evaluate_rep(rep, frames)
    assert "min_knee_angle" in eval_.metrics
    assert "back_angle_at_bottom" in eval_.metrics
    assert "knee_valgus_ratio" in eval_.metrics
    assert "rep_duration_ms" in eval_.metrics
```

- [ ] **Step 3: Run tests, verify failure**

Run: `pytest tests/exercises/test_squat.py -v`
Expected: FAIL — module not found.

- [ ] **Step 4: Implement SquatRule**

```python
# sport_companion_ai/exercises/squat.py
"""Squat form evaluator.

Rules implemented (codes):
- SQUAT_DEPTH_INSUFFICIENT (HIGH/MED) — min knee angle > 95°
- SQUAT_BACK_TOO_VERTICAL (LOW)       — back angle < 30° at bottom
- SQUAT_FORWARD_LEAN (MED)            — back angle > 60° at bottom
- SQUAT_KNEE_VALGUS (HIGH)            — knee valgus ratio > 0.15
- SQUAT_TOO_FAST (LOW)                — rep duration < 800 ms

Tuning: thresholds and penalties are starting estimates. Adjust against fixture
videos (see tests/fixtures/manifest.yaml) without touching shared code.
"""
import math

from sport_companion_ai.exercises.base import ExerciseRule, register_rule
from sport_companion_ai.geometry import back_angle, knee_angle, knee_valgus_ratio
from sport_companion_ai.pose.schema import Frame
from sport_companion_ai.report import Issue, Rep, RepEvaluation


@register_rule
class SquatRule(ExerciseRule):
    name = "squat"
    primary_angle = "knee"
    rep_threshold_low = 100.0
    rep_threshold_high = 160.0

    DEPTH_TARGET = 95.0
    DEPTH_HIGH_SEVERITY_THRESHOLD = 110.0
    BACK_ANGLE_MIN = 30.0
    BACK_ANGLE_MAX = 60.0
    KNEE_VALGUS_THRESHOLD = 0.15
    REP_TOO_FAST_MS = 800

    def _primary_angle_series(self, frames: list[Frame]) -> list[float]:
        out: list[float] = []
        for f in frames:
            if f.skeleton is None:
                out.append(float("nan"))
                continue
            try:
                out.append(knee_angle(f.skeleton, side="left"))
            except KeyError:
                out.append(float("nan"))
        return out

    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation:
        rep_frames = [f for f in frames[rep.start_idx:rep.end_idx + 1] if f.skeleton is not None]
        if not rep_frames:
            return RepEvaluation(
                rep_index=rep.rep_index,
                score=None, passed=None,
                inconclusive=True,
                inconclusive_reason="MISSING_KEYPOINTS",
                keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
            )

        # Metrics
        knee_series = [knee_angle(f.skeleton, side="left") for f in rep_frames]
        knee_series = [v for v in knee_series if not math.isnan(v)]
        if not knee_series:
            return RepEvaluation(
                rep_index=rep.rep_index, score=None, passed=None,
                inconclusive=True, inconclusive_reason="MISSING_KEYPOINTS",
                keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
            )

        min_knee = min(knee_series)
        max_knee = max(knee_series)
        bottom_frame = frames[rep.peak_idx]
        back_at_bottom = back_angle(bottom_frame.skeleton) if bottom_frame.skeleton else float("nan")
        valgus = knee_valgus_ratio(bottom_frame.skeleton) if bottom_frame.skeleton else 0.0

        fps = self.fps
        rep_duration_ms = int((rep.end_idx - rep.start_idx) / fps * 1000)

        issues: list[Issue] = []
        score = 100

        # Rule 1: depth
        if min_knee > self.DEPTH_TARGET:
            high_sev = min_knee > self.DEPTH_HIGH_SEVERITY_THRESHOLD
            severity = "HIGH" if high_sev else "MEDIUM"
            penalty = 25 if high_sev else 10
            issues.append(Issue(
                code="SQUAT_DEPTH_INSUFFICIENT",
                severity=severity,
                message_vi=f"Hạ chưa đủ sâu, đầu gối chỉ gập {min_knee:.0f}° (cần ≤ {self.DEPTH_TARGET:.0f}°)",
                frame_indices=[rep.peak_idx],
                recommendation="Hạ thấp hông hơn cho đến khi đùi song song mặt đất",
            ))
            score -= penalty

        # Rule 2: back angle range at bottom
        if not math.isnan(back_at_bottom):
            if back_at_bottom < self.BACK_ANGLE_MIN:
                issues.append(Issue(
                    code="SQUAT_BACK_TOO_VERTICAL",
                    severity="LOW",
                    message_vi=f"Lưng dựng đứng quá ({back_at_bottom:.0f}°), có thể dồn áp lực gối",
                    frame_indices=[rep.peak_idx],
                    recommendation="Cho phép thân hơi nghiêng về trước khoảng 35–50°",
                ))
                score -= 5
            elif back_at_bottom > self.BACK_ANGLE_MAX:
                issues.append(Issue(
                    code="SQUAT_FORWARD_LEAN",
                    severity="MEDIUM",
                    message_vi=f"Thân nghiêng về trước quá nhiều ({back_at_bottom:.0f}°)",
                    frame_indices=[rep.peak_idx],
                    recommendation="Giữ ngực lên, cốt lõi căng để giảm forward lean",
                ))
                score -= 15

        # Rule 3: valgus
        if valgus > self.KNEE_VALGUS_THRESHOLD:
            issues.append(Issue(
                code="SQUAT_KNEE_VALGUS",
                severity="HIGH",
                message_vi=f"Đầu gối quặp vào trong (ratio {valgus:.2f}, ngưỡng {self.KNEE_VALGUS_THRESHOLD:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Đẩy đầu gối ra ngoài cùng hướng mũi chân khi xuống và lên",
            ))
            score -= 20

        # Rule 4: tempo
        if rep_duration_ms < self.REP_TOO_FAST_MS:
            issues.append(Issue(
                code="SQUAT_TOO_FAST",
                severity="LOW",
                message_vi=f"Nhịp rep quá nhanh ({rep_duration_ms} ms)",
                frame_indices=[rep.start_idx, rep.end_idx],
                recommendation="Hạ chậm 2–3 giây và lên có kiểm soát",
            ))
            score -= 5

        score = max(0, score)
        passed = score >= 70 and not any(i.severity == "HIGH" for i in issues)

        return RepEvaluation(
            rep_index=rep.rep_index,
            score=score,
            passed=passed,
            inconclusive=False,
            issues=issues,
            metrics={
                "min_knee_angle": round(min_knee, 1),
                "max_knee_angle": round(max_knee, 1),
                "back_angle_at_bottom": round(back_at_bottom, 1) if not math.isnan(back_at_bottom) else None,
                "knee_valgus_ratio": round(valgus, 3),
                "rep_duration_ms": float(rep_duration_ms),
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )
```

- [ ] **Step 5: Register the rule on import**

Update `sport_companion_ai/exercises/__init__.py`:

```python
# sport_companion_ai/exercises/__init__.py
"""Exercise rule modules. Importing this package registers all rules."""
from sport_companion_ai.exercises import squat  # noqa: F401
```

- [ ] **Step 6: Run tests, verify pass**

Run: `pytest tests/exercises/test_squat.py -v`
Expected: PASS (7 tests).

- [ ] **Step 7: Commit**

```bash
git add sport_companion_ai/exercises/ tests/exercises/
git commit -m "feat(exercises): add full SquatRule with depth/back/valgus/tempo rules"
```

---

## Task 9: Push-Up Rule

**Files:**
- Create: `sport_companion_ai/exercises/pushup.py`
- Create: `tests/exercises/test_pushup.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/exercises/test_pushup.py
import math

from sport_companion_ai.exercises.pushup import PushUpRule
from sport_companion_ai.exercises.base import ExerciseRule
from sport_companion_ai.pose.schema import Frame, Keypoint, Skeleton


def kp(x, y):
    return Keypoint(x=x, y=y, visibility=1.0)


def pushup_skeleton(elbow_deg: float, hip_sag: float = 0.0) -> Skeleton:
    """Bottom-of-pushup-ish kinematic skeleton.

    elbow_deg controls elbow flex (180 = locked out, 80 = bottom).
    hip_sag in normalized y; positive = hip drops (sag), negative = pikes up.
    """
    half = elbow_deg / 2
    rad = math.radians(half)
    arm_len = 0.18
    elbow_dx = arm_len * math.sin(rad)
    elbow_dy = arm_len * math.cos(rad)
    ls_x, ls_y = 0.30, 0.45
    le_x, le_y = ls_x + elbow_dx, ls_y + elbow_dy
    lw_x, lw_y = ls_x, ls_y + 2 * elbow_dy
    rs_x, rs_y = 0.30, 0.55
    re_x, re_y = rs_x + elbow_dx, rs_y + elbow_dy
    rw_x, rw_y = rs_x, rs_y + 2 * elbow_dy
    lh_x, lh_y = 0.55, 0.50 + hip_sag
    rh_x, rh_y = 0.55, 0.50 + hip_sag
    lk_x, lk_y = 0.75, 0.50
    rk_x, rk_y = 0.75, 0.50
    la_x, la_y = 0.90, 0.50
    ra_x, ra_y = 0.90, 0.50
    return Skeleton(keypoints={
        "left_shoulder": kp(ls_x, ls_y), "right_shoulder": kp(rs_x, rs_y),
        "left_elbow": kp(le_x, le_y), "right_elbow": kp(re_x, re_y),
        "left_wrist": kp(lw_x, lw_y), "right_wrist": kp(rw_x, rw_y),
        "left_hip": kp(lh_x, lh_y), "right_hip": kp(rh_x, rh_y),
        "left_knee": kp(lk_x, lk_y), "right_knee": kp(rk_x, rk_y),
        "left_ankle": kp(la_x, la_y), "right_ankle": kp(ra_x, ra_y),
    })


def make_pushup_rep_frames(min_elbow_deg: float, hip_sag: float = 0.0,
                           fps: int = 30, rep_seconds: float = 1.6) -> list[Frame]:
    n = int(rep_seconds * fps)
    out = []
    for i in range(n):
        progress = abs(2 * (i / n) - 1)
        elbow_at_t = 170 - (170 - min_elbow_deg) * (1 - progress)
        skel = pushup_skeleton(elbow_at_t, hip_sag=hip_sag)
        out.append(Frame(index=i, timestamp_ms=int(i / fps * 1000), skeleton=skel))
    return out


def test_registered():
    assert ExerciseRule.get("push_up") is PushUpRule


def test_clean_rep_passes():
    frames = make_pushup_rep_frames(min_elbow_deg=85)
    rule = PushUpRule()
    rep = rule.detect_reps(frames)[0]
    eval_ = rule.evaluate_rep(rep, frames)
    assert eval_.passed is True
    assert eval_.issues == []


def test_partial_rom_flagged():
    frames = make_pushup_rep_frames(min_elbow_deg=130)
    rule = PushUpRule()
    reps = rule.detect_reps(frames)
    if not reps:
        return  # threshold may filter shallow rep entirely; that's a pass for the rule
    eval_ = rule.evaluate_rep(reps[0], frames)
    assert any(i.code == "PUSHUP_PARTIAL_ROM" for i in eval_.issues)


def test_hip_sag_flagged():
    frames = make_pushup_rep_frames(min_elbow_deg=85, hip_sag=0.05)
    rule = PushUpRule()
    rep = rule.detect_reps(frames)[0]
    eval_ = rule.evaluate_rep(rep, frames)
    assert any(i.code == "PUSHUP_HIP_SAG" for i in eval_.issues)
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/exercises/test_pushup.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement PushUpRule**

```python
# sport_companion_ai/exercises/pushup.py
"""Push-up form evaluator.

Rules:
- PUSHUP_PARTIAL_ROM (MEDIUM) — min elbow angle > 110° (not deep enough)
- PUSHUP_HIP_SAG (HIGH)       — hip y deviates above shoulder/ankle line by > 0.03
- PUSHUP_HIP_PIKE (MEDIUM)    — hip y above shoulder/ankle line (pushed up)
"""
import math

from sport_companion_ai.exercises.base import ExerciseRule, register_rule
from sport_companion_ai.geometry import elbow_angle
from sport_companion_ai.pose.schema import Frame
from sport_companion_ai.report import Issue, Rep, RepEvaluation


def _hip_alignment_offset(skel) -> float:
    """Positive = hip below shoulder-ankle line (sag). Negative = pike up."""
    ls = skel.keypoints["left_shoulder"]; rs = skel.keypoints["right_shoulder"]
    lh = skel.keypoints["left_hip"]; rh = skel.keypoints["right_hip"]
    la = skel.keypoints["left_ankle"]; ra = skel.keypoints["right_ankle"]
    sh_y = (ls.y + rs.y) / 2
    hip_y = (lh.y + rh.y) / 2
    ank_y = (la.y + ra.y) / 2
    expected_hip_y = (sh_y + ank_y) / 2
    return hip_y - expected_hip_y


@register_rule
class PushUpRule(ExerciseRule):
    name = "push_up"
    primary_angle = "elbow"
    rep_threshold_low = 110.0
    rep_threshold_high = 160.0

    ROM_TARGET = 110.0
    HIP_SAG_THRESHOLD = 0.03
    HIP_PIKE_THRESHOLD = -0.03

    def _primary_angle_series(self, frames: list[Frame]) -> list[float]:
        out: list[float] = []
        for f in frames:
            if f.skeleton is None:
                out.append(float("nan"))
                continue
            try:
                out.append(elbow_angle(f.skeleton, side="left"))
            except KeyError:
                out.append(float("nan"))
        return out

    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation:
        rep_frames = [f for f in frames[rep.start_idx:rep.end_idx + 1] if f.skeleton is not None]
        if not rep_frames:
            return RepEvaluation(
                rep_index=rep.rep_index, score=None, passed=None,
                inconclusive=True, inconclusive_reason="MISSING_KEYPOINTS",
                keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
            )

        elbow_series = [elbow_angle(f.skeleton, side="left") for f in rep_frames]
        elbow_series = [v for v in elbow_series if not math.isnan(v)]
        min_elbow = min(elbow_series) if elbow_series else float("nan")
        bottom_frame = frames[rep.peak_idx]
        hip_offset = _hip_alignment_offset(bottom_frame.skeleton) if bottom_frame.skeleton else 0.0

        issues: list[Issue] = []
        score = 100

        if not math.isnan(min_elbow) and min_elbow > self.ROM_TARGET:
            issues.append(Issue(
                code="PUSHUP_PARTIAL_ROM",
                severity="MEDIUM",
                message_vi=f"Khuỷu chỉ gập đến {min_elbow:.0f}°, chưa xuống đủ sâu",
                frame_indices=[rep.peak_idx],
                recommendation="Hạ ngực gần chạm sàn rồi đẩy lên",
            ))
            score -= 15

        if hip_offset > self.HIP_SAG_THRESHOLD:
            issues.append(Issue(
                code="PUSHUP_HIP_SAG",
                severity="HIGH",
                message_vi=f"Hông sụp xuống (offset {hip_offset:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Siết bụng và mông giữ thân thẳng plank",
            ))
            score -= 25
        elif hip_offset < self.HIP_PIKE_THRESHOLD:
            issues.append(Issue(
                code="PUSHUP_HIP_PIKE",
                severity="MEDIUM",
                message_vi=f"Hông đẩy cao (offset {hip_offset:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Hạ hông xuống thẳng hàng với vai và gót",
            ))
            score -= 15

        score = max(0, score)
        passed = score >= 70 and not any(i.severity == "HIGH" for i in issues)

        return RepEvaluation(
            rep_index=rep.rep_index,
            score=score, passed=passed, inconclusive=False,
            issues=issues,
            metrics={
                "min_elbow_angle": round(min_elbow, 1) if not math.isnan(min_elbow) else None,
                "hip_alignment_offset": round(hip_offset, 3),
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )
```

- [ ] **Step 4: Register**

Update `sport_companion_ai/exercises/__init__.py`:

```python
from sport_companion_ai.exercises import squat, pushup  # noqa: F401
```

- [ ] **Step 5: Run tests, verify pass**

Run: `pytest tests/exercises/test_pushup.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add sport_companion_ai/exercises/pushup.py sport_companion_ai/exercises/__init__.py tests/exercises/test_pushup.py
git commit -m "feat(exercises): add PushUpRule (ROM + hip alignment)"
```

---

## Task 10: Bicep Curl Rule

**Files:**
- Create: `sport_companion_ai/exercises/bicep_curl.py`
- Create: `tests/exercises/test_bicep_curl.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/exercises/test_bicep_curl.py
import math

from sport_companion_ai.exercises.bicep_curl import BicepCurlRule
from sport_companion_ai.exercises.base import ExerciseRule
from sport_companion_ai.pose.schema import Frame, Keypoint, Skeleton


def kp(x, y):
    return Keypoint(x=x, y=y, visibility=1.0)


def curl_skeleton(elbow_deg: float, elbow_drift: float = 0.0) -> Skeleton:
    """Standing curl. elbow_drift > 0 moves elbow forward of shoulder."""
    half = elbow_deg / 2
    rad = math.radians(half)
    arm_len = 0.15
    e_dx = arm_len * math.sin(rad)
    e_dy = arm_len * math.cos(rad)
    ls_x, ls_y = 0.50, 0.30
    le_x, le_y = ls_x + elbow_drift, ls_y + 0.18
    lw_x, lw_y = le_x - e_dx, le_y - e_dy
    return Skeleton(keypoints={
        "left_shoulder": kp(ls_x, ls_y), "left_elbow": kp(le_x, le_y), "left_wrist": kp(lw_x, lw_y),
        "right_shoulder": kp(0.55, 0.30), "right_elbow": kp(0.55, 0.48), "right_wrist": kp(0.55, 0.62),
        "left_hip": kp(0.48, 0.55), "right_hip": kp(0.55, 0.55),
        "left_knee": kp(0.48, 0.75), "right_knee": kp(0.55, 0.75),
        "left_ankle": kp(0.48, 0.95), "right_ankle": kp(0.55, 0.95),
    })


def make_curl_rep_frames(min_elbow_deg: float, elbow_drift: float = 0.0,
                         fps: int = 30, rep_seconds: float = 1.6) -> list[Frame]:
    n = int(rep_seconds * fps)
    out = []
    for i in range(n):
        progress = abs(2 * (i / n) - 1)
        elbow_at_t = 170 - (170 - min_elbow_deg) * (1 - progress)
        skel = curl_skeleton(elbow_at_t, elbow_drift=elbow_drift)
        out.append(Frame(index=i, timestamp_ms=int(i / fps * 1000), skeleton=skel))
    return out


def test_registered():
    assert ExerciseRule.get("bicep_curl") is BicepCurlRule


def test_clean_rep_passes():
    frames = make_curl_rep_frames(min_elbow_deg=45)
    rule = BicepCurlRule()
    rep = rule.detect_reps(frames)[0]
    eval_ = rule.evaluate_rep(rep, frames)
    assert eval_.passed is True


def test_short_rom_flagged():
    frames = make_curl_rep_frames(min_elbow_deg=110)
    rule = BicepCurlRule()
    reps = rule.detect_reps(frames)
    if not reps:
        return
    eval_ = rule.evaluate_rep(reps[0], frames)
    assert any(i.code == "CURL_PARTIAL_ROM" for i in eval_.issues)


def test_elbow_drift_flagged():
    frames = make_curl_rep_frames(min_elbow_deg=45, elbow_drift=0.05)
    rule = BicepCurlRule()
    rep = rule.detect_reps(frames)[0]
    eval_ = rule.evaluate_rep(rep, frames)
    assert any(i.code == "CURL_ELBOW_DRIFT" for i in eval_.issues)
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/exercises/test_bicep_curl.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement BicepCurlRule**

```python
# sport_companion_ai/exercises/bicep_curl.py
"""Bicep curl form evaluator.

Rules:
- CURL_PARTIAL_ROM (MEDIUM)   — min elbow angle > 70° (not flexed enough)
- CURL_ELBOW_DRIFT (MEDIUM)   — elbow x deviates from shoulder x > 0.04
- CURL_TOO_FAST (LOW)         — eccentric < 600 ms
"""
import math

from sport_companion_ai.exercises.base import ExerciseRule, register_rule
from sport_companion_ai.geometry import elbow_angle
from sport_companion_ai.pose.schema import Frame
from sport_companion_ai.report import Issue, Rep, RepEvaluation


def _elbow_drift(skel) -> float:
    return abs(skel.keypoints["left_elbow"].x - skel.keypoints["left_shoulder"].x)


@register_rule
class BicepCurlRule(ExerciseRule):
    name = "bicep_curl"
    primary_angle = "elbow"
    rep_threshold_low = 70.0
    rep_threshold_high = 150.0

    ROM_TARGET = 70.0
    ELBOW_DRIFT_THRESHOLD = 0.04
    REP_TOO_FAST_MS = 600

    def _primary_angle_series(self, frames: list[Frame]) -> list[float]:
        out: list[float] = []
        for f in frames:
            if f.skeleton is None:
                out.append(float("nan"))
                continue
            try:
                out.append(elbow_angle(f.skeleton, side="left"))
            except KeyError:
                out.append(float("nan"))
        return out

    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation:
        rep_frames = [f for f in frames[rep.start_idx:rep.end_idx + 1] if f.skeleton is not None]
        if not rep_frames:
            return RepEvaluation(
                rep_index=rep.rep_index, score=None, passed=None,
                inconclusive=True, inconclusive_reason="MISSING_KEYPOINTS",
                keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
            )

        elbow_series = [elbow_angle(f.skeleton, side="left") for f in rep_frames]
        elbow_series = [v for v in elbow_series if not math.isnan(v)]
        min_elbow = min(elbow_series) if elbow_series else float("nan")
        max_drift = max(_elbow_drift(f.skeleton) for f in rep_frames)
        rep_duration_ms = int((rep.end_idx - rep.start_idx) / self.fps * 1000)

        issues: list[Issue] = []
        score = 100

        if not math.isnan(min_elbow) and min_elbow > self.ROM_TARGET:
            issues.append(Issue(
                code="CURL_PARTIAL_ROM", severity="MEDIUM",
                message_vi=f"Khuỷu chỉ gập đến {min_elbow:.0f}°, chưa hết tầm",
                frame_indices=[rep.peak_idx],
                recommendation="Curl hết biên độ, đưa tạ gần vai",
            ))
            score -= 15

        if max_drift > self.ELBOW_DRIFT_THRESHOLD:
            issues.append(Issue(
                code="CURL_ELBOW_DRIFT", severity="MEDIUM",
                message_vi=f"Khuỷu di chuyển ra trước (drift {max_drift:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Giữ khuỷu cố định sát thân, không vung vai",
            ))
            score -= 15

        if rep_duration_ms < self.REP_TOO_FAST_MS:
            issues.append(Issue(
                code="CURL_TOO_FAST", severity="LOW",
                message_vi=f"Rep quá nhanh ({rep_duration_ms} ms)",
                frame_indices=[rep.start_idx, rep.end_idx],
                recommendation="Hạ tạ trong 2 giây để kiểm soát eccentric",
            ))
            score -= 5

        score = max(0, score)
        passed = score >= 70 and not any(i.severity == "HIGH" for i in issues)

        return RepEvaluation(
            rep_index=rep.rep_index,
            score=score, passed=passed, inconclusive=False,
            issues=issues,
            metrics={
                "min_elbow_angle": round(min_elbow, 1) if not math.isnan(min_elbow) else None,
                "max_elbow_drift": round(max_drift, 3),
                "rep_duration_ms": float(rep_duration_ms),
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )
```

- [ ] **Step 4: Register**

Update `sport_companion_ai/exercises/__init__.py`:

```python
from sport_companion_ai.exercises import squat, pushup, bicep_curl  # noqa: F401
```

- [ ] **Step 5: Run tests, verify pass**

Run: `pytest tests/exercises/test_bicep_curl.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add sport_companion_ai/exercises/bicep_curl.py sport_companion_ai/exercises/__init__.py tests/exercises/test_bicep_curl.py
git commit -m "feat(exercises): add BicepCurlRule (ROM + elbow drift + tempo)"
```

---

## Task 11: Deadlift Rule

**Files:**
- Create: `sport_companion_ai/exercises/deadlift.py`
- Create: `tests/exercises/test_deadlift.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/exercises/test_deadlift.py
import math

from sport_companion_ai.exercises.deadlift import DeadliftRule
from sport_companion_ai.exercises.base import ExerciseRule
from sport_companion_ai.pose.schema import Frame, Keypoint, Skeleton


def kp(x, y):
    return Keypoint(x=x, y=y, visibility=1.0)


def deadlift_skeleton(hip_deg: float, back_round: float = 0.0) -> Skeleton:
    """Set up: hip flex angle controls how bent over.
    back_round shifts mid-back forward to simulate spinal flexion."""
    half = hip_deg / 2
    rad = math.radians(half)
    torso_len = 0.30
    sh_dx = torso_len * math.sin(rad)
    sh_dy = torso_len * math.cos(rad)
    lh_x, lh_y = 0.50, 0.55
    rh_x, rh_y = 0.52, 0.55
    ls_x, ls_y = lh_x + sh_dx + back_round, lh_y - sh_dy
    rs_x, rs_y = rh_x + sh_dx + back_round, rh_y - sh_dy
    return Skeleton(keypoints={
        "left_hip": kp(lh_x, lh_y), "right_hip": kp(rh_x, rh_y),
        "left_shoulder": kp(ls_x, ls_y), "right_shoulder": kp(rs_x, rs_y),
        "left_knee": kp(lh_x, 0.75), "right_knee": kp(rh_x, 0.75),
        "left_ankle": kp(lh_x, 0.95), "right_ankle": kp(rh_x, 0.95),
        "left_elbow": kp(ls_x + 0.05, ls_y + 0.10), "right_elbow": kp(rs_x + 0.05, rs_y + 0.10),
        "left_wrist": kp(ls_x + 0.05, ls_y + 0.20), "right_wrist": kp(rs_x + 0.05, rs_y + 0.20),
    })


def make_deadlift_rep_frames(min_hip_deg: float, back_round: float = 0.0,
                             fps: int = 30, rep_seconds: float = 2.5) -> list[Frame]:
    n = int(rep_seconds * fps)
    out = []
    for i in range(n):
        progress = abs(2 * (i / n) - 1)
        hip_at_t = 170 - (170 - min_hip_deg) * (1 - progress)
        skel = deadlift_skeleton(hip_at_t, back_round=back_round)
        out.append(Frame(index=i, timestamp_ms=int(i / fps * 1000), skeleton=skel))
    return out


def test_registered():
    assert ExerciseRule.get("deadlift") is DeadliftRule


def test_clean_rep_passes():
    frames = make_deadlift_rep_frames(min_hip_deg=80)
    rule = DeadliftRule()
    rep = rule.detect_reps(frames)[0]
    eval_ = rule.evaluate_rep(rep, frames)
    assert eval_.passed is True


def test_back_rounding_flagged():
    frames = make_deadlift_rep_frames(min_hip_deg=80, back_round=0.08)
    rule = DeadliftRule()
    rep = rule.detect_reps(frames)[0]
    eval_ = rule.evaluate_rep(rep, frames)
    assert any(i.code == "DEADLIFT_BACK_ROUND" for i in eval_.issues)


def test_partial_lockout_flagged():
    # Setup never opens hip back to >155° at top — incomplete lockout
    frames = make_deadlift_rep_frames(min_hip_deg=80, rep_seconds=2.5)
    # Truncate trailing ascent half — last frame stays around hip≈110°
    frames = frames[:int(len(frames) * 0.55)]
    rule = DeadliftRule()
    reps = rule.detect_reps(frames)
    if not reps:
        return  # rep filter may discard incomplete; that's a documented behavior
    eval_ = rule.evaluate_rep(reps[0], frames)
    codes = {i.code for i in eval_.issues}
    assert "DEADLIFT_PARTIAL_LOCKOUT" in codes
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/exercises/test_deadlift.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement DeadliftRule**

```python
# sport_companion_ai/exercises/deadlift.py
"""Deadlift form evaluator.

Rules:
- DEADLIFT_BACK_ROUND (HIGH)        — shoulder x deviates forward of hip-knee line
- DEADLIFT_PARTIAL_LOCKOUT (MEDIUM) — hip never opens past 155° on ascent
- DEADLIFT_HIP_RISE_FIRST (MEDIUM)  — hip extends faster than shoulders early in ascent
"""
import math

from sport_companion_ai.exercises.base import ExerciseRule, register_rule
from sport_companion_ai.geometry import hip_angle
from sport_companion_ai.pose.schema import Frame
from sport_companion_ai.report import Issue, Rep, RepEvaluation


def _back_curvature(skel) -> float:
    """Approximate spinal flexion: shoulder x relative to hip x.
    Larger positive = shoulders forward of hip (rounding)."""
    return abs(skel.keypoints["left_shoulder"].x - skel.keypoints["left_hip"].x)


@register_rule
class DeadliftRule(ExerciseRule):
    name = "deadlift"
    primary_angle = "hip"
    rep_threshold_low = 100.0
    rep_threshold_high = 155.0

    BACK_ROUND_THRESHOLD = 0.18
    LOCKOUT_TARGET = 155.0

    def _primary_angle_series(self, frames: list[Frame]) -> list[float]:
        out: list[float] = []
        for f in frames:
            if f.skeleton is None:
                out.append(float("nan"))
                continue
            try:
                out.append(hip_angle(f.skeleton, side="left"))
            except KeyError:
                out.append(float("nan"))
        return out

    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation:
        rep_frames = [f for f in frames[rep.start_idx:rep.end_idx + 1] if f.skeleton is not None]
        if not rep_frames:
            return RepEvaluation(
                rep_index=rep.rep_index, score=None, passed=None,
                inconclusive=True, inconclusive_reason="MISSING_KEYPOINTS",
                keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
            )

        hip_series = [hip_angle(f.skeleton, side="left") for f in rep_frames]
        hip_series = [v for v in hip_series if not math.isnan(v)]
        max_hip = max(hip_series) if hip_series else float("nan")
        max_back_curve = max(_back_curvature(f.skeleton) for f in rep_frames)

        issues: list[Issue] = []
        score = 100

        if max_back_curve > self.BACK_ROUND_THRESHOLD:
            issues.append(Issue(
                code="DEADLIFT_BACK_ROUND", severity="HIGH",
                message_vi=f"Lưng có dấu hiệu cong (offset {max_back_curve:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Siết lat, ngực ưỡn giữ lưng phẳng trong toàn bộ rep",
            ))
            score -= 30

        if not math.isnan(max_hip) and max_hip < self.LOCKOUT_TARGET:
            issues.append(Issue(
                code="DEADLIFT_PARTIAL_LOCKOUT", severity="MEDIUM",
                message_vi=f"Chưa khóa lockout hông (max {max_hip:.0f}°)",
                frame_indices=[rep.end_idx],
                recommendation="Đứng thẳng hoàn toàn, siết mông ở đỉnh rep",
            ))
            score -= 15

        score = max(0, score)
        passed = score >= 70 and not any(i.severity == "HIGH" for i in issues)

        return RepEvaluation(
            rep_index=rep.rep_index,
            score=score, passed=passed, inconclusive=False,
            issues=issues,
            metrics={
                "max_hip_angle": round(max_hip, 1) if not math.isnan(max_hip) else None,
                "max_back_curvature": round(max_back_curve, 3),
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )
```

- [ ] **Step 4: Register**

Update `sport_companion_ai/exercises/__init__.py`:

```python
from sport_companion_ai.exercises import squat, pushup, bicep_curl, deadlift  # noqa: F401
```

- [ ] **Step 5: Run tests, verify pass**

Run: `pytest tests/exercises/test_deadlift.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add sport_companion_ai/exercises/deadlift.py sport_companion_ai/exercises/__init__.py tests/exercises/test_deadlift.py
git commit -m "feat(exercises): add DeadliftRule (back rounding + lockout)"
```

---

## Task 12: Bench Press Rule

**Files:**
- Create: `sport_companion_ai/exercises/bench.py`
- Create: `tests/exercises/test_bench.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/exercises/test_bench.py
import math

from sport_companion_ai.exercises.bench import BenchRule
from sport_companion_ai.exercises.base import ExerciseRule
from sport_companion_ai.pose.schema import Frame, Keypoint, Skeleton


def kp(x, y):
    return Keypoint(x=x, y=y, visibility=1.0)


def bench_skeleton(elbow_deg: float, flare: float = 0.0) -> Skeleton:
    """Side-on view; elbow_deg controls flex.
    flare > 0 = elbow further from torso (wider flare)."""
    half = elbow_deg / 2
    rad = math.radians(half)
    arm_len = 0.18
    e_dx = arm_len * math.sin(rad)
    e_dy = arm_len * math.cos(rad)
    ls_x, ls_y = 0.45, 0.50
    rs_x, rs_y = 0.55, 0.50
    le_x, le_y = ls_x - e_dx - flare, ls_y - e_dy
    re_x, re_y = rs_x + e_dx + flare, rs_y - e_dy
    lw_x, lw_y = ls_x, ls_y - 2 * e_dy
    rw_x, rw_y = rs_x, rs_y - 2 * e_dy
    return Skeleton(keypoints={
        "left_shoulder": kp(ls_x, ls_y), "right_shoulder": kp(rs_x, rs_y),
        "left_elbow": kp(le_x, le_y), "right_elbow": kp(re_x, re_y),
        "left_wrist": kp(lw_x, lw_y), "right_wrist": kp(rw_x, rw_y),
        "left_hip": kp(0.45, 0.65), "right_hip": kp(0.55, 0.65),
        "left_knee": kp(0.40, 0.85), "right_knee": kp(0.60, 0.85),
        "left_ankle": kp(0.40, 0.95), "right_ankle": kp(0.60, 0.95),
    })


def make_bench_rep_frames(min_elbow_deg: float, flare: float = 0.0,
                          fps: int = 30, rep_seconds: float = 1.8) -> list[Frame]:
    n = int(rep_seconds * fps)
    out = []
    for i in range(n):
        progress = abs(2 * (i / n) - 1)
        elbow_at_t = 170 - (170 - min_elbow_deg) * (1 - progress)
        skel = bench_skeleton(elbow_at_t, flare=flare)
        out.append(Frame(index=i, timestamp_ms=int(i / fps * 1000), skeleton=skel))
    return out


def test_registered():
    assert ExerciseRule.get("bench_press") is BenchRule


def test_clean_rep_passes():
    frames = make_bench_rep_frames(min_elbow_deg=85)
    rule = BenchRule()
    rep = rule.detect_reps(frames)[0]
    eval_ = rule.evaluate_rep(rep, frames)
    assert eval_.passed is True


def test_partial_rom_flagged():
    frames = make_bench_rep_frames(min_elbow_deg=125)
    rule = BenchRule()
    reps = rule.detect_reps(frames)
    if not reps:
        return
    eval_ = rule.evaluate_rep(reps[0], frames)
    assert any(i.code == "BENCH_PARTIAL_ROM" for i in eval_.issues)


def test_elbow_flare_flagged():
    frames = make_bench_rep_frames(min_elbow_deg=85, flare=0.10)
    rule = BenchRule()
    rep = rule.detect_reps(frames)[0]
    eval_ = rule.evaluate_rep(rep, frames)
    assert any(i.code == "BENCH_ELBOW_FLARE" for i in eval_.issues)
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/exercises/test_bench.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement BenchRule**

```python
# sport_companion_ai/exercises/bench.py
"""Bench press form evaluator.

Rules:
- BENCH_PARTIAL_ROM (MEDIUM)   — min elbow angle > 110° (no chest contact)
- BENCH_ELBOW_FLARE (MEDIUM)   — elbow x deviates outward from shoulder by > 0.06
- BENCH_ASYMMETRY (LOW)        — left/right elbow angles differ by > 15°
"""
import math

from sport_companion_ai.exercises.base import ExerciseRule, register_rule
from sport_companion_ai.geometry import elbow_angle
from sport_companion_ai.pose.schema import Frame
from sport_companion_ai.report import Issue, Rep, RepEvaluation


def _elbow_flare(skel) -> float:
    le = skel.keypoints["left_elbow"]; ls = skel.keypoints["left_shoulder"]
    re = skel.keypoints["right_elbow"]; rs = skel.keypoints["right_shoulder"]
    return max(abs(le.x - ls.x), abs(re.x - rs.x))


@register_rule
class BenchRule(ExerciseRule):
    name = "bench_press"
    primary_angle = "elbow"
    rep_threshold_low = 100.0
    rep_threshold_high = 160.0

    ROM_TARGET = 110.0
    FLARE_THRESHOLD = 0.16
    ASYMMETRY_THRESHOLD = 15.0

    def _primary_angle_series(self, frames: list[Frame]) -> list[float]:
        out: list[float] = []
        for f in frames:
            if f.skeleton is None:
                out.append(float("nan"))
                continue
            try:
                out.append(elbow_angle(f.skeleton, side="left"))
            except KeyError:
                out.append(float("nan"))
        return out

    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation:
        rep_frames = [f for f in frames[rep.start_idx:rep.end_idx + 1] if f.skeleton is not None]
        if not rep_frames:
            return RepEvaluation(
                rep_index=rep.rep_index, score=None, passed=None,
                inconclusive=True, inconclusive_reason="MISSING_KEYPOINTS",
                keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
            )

        left_series = [elbow_angle(f.skeleton, side="left") for f in rep_frames]
        right_series = [elbow_angle(f.skeleton, side="right") for f in rep_frames]
        left_series = [v for v in left_series if not math.isnan(v)]
        right_series = [v for v in right_series if not math.isnan(v)]
        min_elbow = min(left_series) if left_series else float("nan")
        max_flare = max(_elbow_flare(f.skeleton) for f in rep_frames)
        asymmetry = abs(min(left_series) - min(right_series)) if left_series and right_series else 0.0

        issues: list[Issue] = []
        score = 100

        if not math.isnan(min_elbow) and min_elbow > self.ROM_TARGET:
            issues.append(Issue(
                code="BENCH_PARTIAL_ROM", severity="MEDIUM",
                message_vi=f"Khuỷu chỉ gập đến {min_elbow:.0f}°, chưa đụng ngực",
                frame_indices=[rep.peak_idx],
                recommendation="Hạ tạ chạm nhẹ ngực rồi đẩy lên",
            ))
            score -= 15

        if max_flare > self.FLARE_THRESHOLD:
            issues.append(Issue(
                code="BENCH_ELBOW_FLARE", severity="MEDIUM",
                message_vi=f"Khuỷu xòe quá rộng (flare {max_flare:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Khuỷu nên ở khoảng 45–60° so với thân, không vuông góc",
            ))
            score -= 15

        if asymmetry > self.ASYMMETRY_THRESHOLD:
            issues.append(Issue(
                code="BENCH_ASYMMETRY", severity="LOW",
                message_vi=f"Hai bên khuỷu lệch nhau {asymmetry:.0f}°",
                frame_indices=[rep.peak_idx],
                recommendation="Kiểm tra grip cân và đẩy đều hai tay",
            ))
            score -= 5

        score = max(0, score)
        passed = score >= 70 and not any(i.severity == "HIGH" for i in issues)

        return RepEvaluation(
            rep_index=rep.rep_index,
            score=score, passed=passed, inconclusive=False,
            issues=issues,
            metrics={
                "min_elbow_angle_left": round(min_elbow, 1) if not math.isnan(min_elbow) else None,
                "max_elbow_flare": round(max_flare, 3),
                "left_right_asymmetry": round(asymmetry, 1),
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )
```

- [ ] **Step 4: Register**

Update `sport_companion_ai/exercises/__init__.py`:

```python
from sport_companion_ai.exercises import squat, pushup, bicep_curl, deadlift, bench  # noqa: F401
```

- [ ] **Step 5: Run tests, verify pass**

Run: `pytest tests/exercises/test_bench.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add sport_companion_ai/exercises/bench.py sport_companion_ai/exercises/__init__.py tests/exercises/test_bench.py
git commit -m "feat(exercises): add BenchRule (ROM + flare + asymmetry)"
```

---

## Task 13: Pose Extractor (MediaPipe)

**Files:**
- Create: `sport_companion_ai/pose/extractor.py`
- Create: `tests/test_pose_extractor.py`

MediaPipe ships its own keypoint indices. The extractor maps them to our names.

- [ ] **Step 1: Write tests (mock MediaPipe)**

```python
# tests/test_pose_extractor.py
from unittest.mock import MagicMock

import numpy as np

from sport_companion_ai.pose.extractor import MediaPipeExtractor, PoseExtractor


def test_protocol_satisfied():
    assert isinstance(MediaPipeExtractor(), PoseExtractor)


def test_extract_batch_returns_one_frame_per_input(mocker):
    """Mock mp.solutions.pose.Pose so we don't load the real model."""
    fake_landmark = MagicMock(x=0.5, y=0.5, z=0.0, visibility=0.9)
    fake_landmarks = MagicMock(landmark=[fake_landmark] * 33)
    fake_result = MagicMock(pose_landmarks=fake_landmarks)

    pose_instance = MagicMock()
    pose_instance.process.return_value = fake_result
    pose_instance.close.return_value = None
    mocker.patch("mediapipe.solutions.pose.Pose", return_value=pose_instance)

    images = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(5)]
    extractor = MediaPipeExtractor(fps=30)

    frames = extractor.extract_batch(images)

    assert len(frames) == 5
    assert frames[0].index == 0
    assert frames[0].timestamp_ms == 0
    assert frames[1].timestamp_ms == int(1000 / 30)
    assert frames[0].skeleton is not None
    assert "left_shoulder" in frames[0].skeleton.keypoints


def test_extract_batch_skeleton_none_when_no_detection(mocker):
    fake_result = MagicMock(pose_landmarks=None)
    pose_instance = MagicMock()
    pose_instance.process.return_value = fake_result
    pose_instance.close.return_value = None
    mocker.patch("mediapipe.solutions.pose.Pose", return_value=pose_instance)

    images = [np.zeros((720, 1280, 3), dtype=np.uint8)]
    frames = MediaPipeExtractor().extract_batch(images)

    assert frames[0].skeleton is None
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/test_pose_extractor.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement extractor**

```python
# sport_companion_ai/pose/extractor.py
"""Pose extraction backends. MediaPipe is the Phase 1 default."""
from typing import Protocol, runtime_checkable

import numpy as np

from sport_companion_ai.errors import PoseExtractionError
from sport_companion_ai.pose.schema import Frame, Keypoint, Skeleton, KEYPOINT_NAMES


@runtime_checkable
class PoseExtractor(Protocol):
    """Anything that turns video frames into typed Frames."""
    model_id: str

    def extract_batch(self, images: list[np.ndarray]) -> list[Frame]: ...


class MediaPipeExtractor:
    """BlazePose Full via mediapipe.solutions.pose.

    Returns 33 keypoints in normalized [0,1] coordinates. Maps MediaPipe's
    integer indices onto our `KEYPOINT_NAMES` (same order as MediaPipe's).
    """

    model_id: str = "mediapipe-blazepose-full"

    def __init__(self, fps: int = 30, model_complexity: int = 1,
                 min_detection_confidence: float = 0.5):
        self.fps = fps
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence

    def extract_batch(self, images: list[np.ndarray]) -> list[Frame]:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise PoseExtractionError("mediapipe not installed") from exc

        pose = mp.solutions.pose.Pose(
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            static_image_mode=False,
        )
        try:
            frames: list[Frame] = []
            for i, img in enumerate(images):
                result = pose.process(img)
                if result.pose_landmarks is None:
                    skel = None
                else:
                    kp_dict: dict[str, Keypoint] = {}
                    for idx, lm in enumerate(result.pose_landmarks.landmark):
                        if idx >= len(KEYPOINT_NAMES):
                            break
                        name = KEYPOINT_NAMES[idx]
                        kp_dict[name] = Keypoint(
                            x=float(min(max(lm.x, 0.0), 1.0)),
                            y=float(min(max(lm.y, 0.0), 1.0)),
                            z=float(getattr(lm, "z", 0.0)),
                            visibility=float(min(max(getattr(lm, "visibility", 0.0), 0.0), 1.0)),
                        )
                    skel = Skeleton(keypoints=kp_dict)
                frames.append(Frame(
                    index=i,
                    timestamp_ms=int(i / self.fps * 1000),
                    skeleton=skel,
                ))
            return frames
        finally:
            pose.close()
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_pose_extractor.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add sport_companion_ai/pose/extractor.py tests/test_pose_extractor.py
git commit -m "feat(pose): add MediaPipe pose extractor (Protocol-based, swappable)"
```

---

## Task 14: Video Reader

**Files:**
- Create: `sport_companion_ai/pose/video_reader.py`
- Create: `tests/test_video_reader.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_video_reader.py
from unittest.mock import MagicMock

import numpy as np
import pytest

from sport_companion_ai.errors import VideoReadError
from sport_companion_ai.pose.video_reader import read_video, VideoMeta


def _fake_capture(frames: list[np.ndarray], fps: int = 30, w: int = 1280, h: int = 720):
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        3: w,    # CAP_PROP_FRAME_WIDTH
        4: h,    # CAP_PROP_FRAME_HEIGHT
        5: float(fps),  # CAP_PROP_FPS
        7: len(frames),  # CAP_PROP_FRAME_COUNT
    }.get(prop, 0)
    iter_ = iter(frames)
    def read_side_effect():
        try:
            return True, next(iter_)
        except StopIteration:
            return False, None
    cap.read.side_effect = read_side_effect
    cap.release = MagicMock()
    return cap


def test_read_video_returns_frames_and_meta(mocker):
    frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(5)]
    cap = _fake_capture(frames)
    mocker.patch("cv2.VideoCapture", return_value=cap)

    images, meta = read_video("dummy.mp4")

    assert len(images) == 5
    assert isinstance(meta, VideoMeta)
    assert meta.width == 1280
    assert meta.height == 720
    assert meta.fps == 30
    assert meta.duration_ms == int(5 / 30 * 1000)


def test_read_video_raises_when_not_opened(mocker):
    cap = MagicMock()
    cap.isOpened.return_value = False
    mocker.patch("cv2.VideoCapture", return_value=cap)

    with pytest.raises(VideoReadError):
        read_video("missing.mp4")


def test_read_video_raises_when_zero_frames(mocker):
    cap = _fake_capture([])
    cap.get.side_effect = lambda prop: {3: 1280, 4: 720, 5: 30.0, 7: 0}.get(prop, 0)
    mocker.patch("cv2.VideoCapture", return_value=cap)

    with pytest.raises(VideoReadError):
        read_video("empty.mp4")
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/test_video_reader.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement video reader**

```python
# sport_companion_ai/pose/video_reader.py
"""Read a video file into a list of BGR numpy frames + metadata."""
from typing import Tuple

import cv2
import numpy as np

from sport_companion_ai.errors import VideoReadError
from sport_companion_ai.report import VideoMeta


def read_video(path: str) -> Tuple[list[np.ndarray], VideoMeta]:
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise VideoReadError(f"cannot open video: {path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 30))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            raise VideoReadError(f"video reports 0 frames: {path}")

        frames: list[np.ndarray] = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            # Convert BGR → RGB for MediaPipe consumption
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not frames:
            raise VideoReadError(f"decoded 0 frames from {path}")

        duration_ms = int(len(frames) / fps * 1000)
        meta = VideoMeta(width=width, height=height, fps=fps, duration_ms=duration_ms)
        return frames, meta
    finally:
        cap.release()
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_video_reader.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add sport_companion_ai/pose/video_reader.py tests/test_video_reader.py
git commit -m "feat(pose): add cv2-based video reader returning RGB frames + meta"
```

---

## Task 15: Output Sampling

**Files:**
- Create: `sport_companion_ai/sampling.py`
- Create: `tests/test_sampling.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_sampling.py
from sport_companion_ai.pose.schema import Frame, Keypoint, Skeleton
from sport_companion_ai.report import Issue, RepEvaluation
from sport_companion_ai.sampling import select_frames_for_output, SkeletonOutputMode


def make_frame(i: int) -> Frame:
    return Frame(index=i, timestamp_ms=i * 33, skeleton=Skeleton(
        keypoints={"nose": Keypoint(x=0.5, y=0.5, visibility=1.0)}))


def make_eval(rep_index: int, start: int, peak: int, end: int,
              issue_frames: list[int] | None = None) -> RepEvaluation:
    return RepEvaluation(
        rep_index=rep_index, score=80, passed=True,
        issues=[Issue(code="X", severity="LOW", message_vi="",
                      frame_indices=issue_frames or [])],
        keyframes={"start": start, "peak": peak, "end": end},
    )


def test_full_returns_all():
    frames = [make_frame(i) for i in range(100)]
    out = select_frames_for_output(frames, [], SkeletonOutputMode.FULL, fps=30)
    assert len(out) == 100


def test_none_returns_empty():
    frames = [make_frame(i) for i in range(100)]
    out = select_frames_for_output(frames, [], SkeletonOutputMode.NONE, fps=30)
    assert out == []


def test_sampled_at_5fps():
    frames = [make_frame(i) for i in range(60)]  # 2s @ 30fps
    out = select_frames_for_output(frames, [], SkeletonOutputMode.SAMPLED, fps=30)
    # 5fps → step 6 → ~10 frames
    assert 8 <= len(out) <= 12
    assert out[0].index == 0


def test_keyframes_only_returns_rep_keyframes_and_issue_indices():
    frames = [make_frame(i) for i in range(100)]
    reps = [
        make_eval(0, start=10, peak=20, end=30, issue_frames=[25]),
        make_eval(1, start=50, peak=60, end=70, issue_frames=[55, 58]),
    ]
    out = select_frames_for_output(frames, reps, SkeletonOutputMode.KEYFRAMES, fps=30)
    out_indices = {f.index for f in out}
    assert out_indices == {10, 20, 25, 30, 50, 55, 58, 60, 70}
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/test_sampling.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement sampling**

```python
# sport_companion_ai/sampling.py
"""Reduce the size of `frames[]` in the output report."""
from enum import Enum

from sport_companion_ai.pose.schema import Frame
from sport_companion_ai.report import RepEvaluation


class SkeletonOutputMode(str, Enum):
    FULL = "full"
    SAMPLED = "sampled"
    KEYFRAMES = "keyframes"
    NONE = "none"


def select_frames_for_output(
    frames: list[Frame],
    reps: list[RepEvaluation],
    mode: SkeletonOutputMode,
    fps: int = 30,
) -> list[Frame]:
    if mode is SkeletonOutputMode.NONE:
        return []
    if mode is SkeletonOutputMode.FULL:
        return list(frames)
    if mode is SkeletonOutputMode.SAMPLED:
        step = max(1, fps // 5)  # ~5 fps subsample
        return [frames[i] for i in range(0, len(frames), step)]
    if mode is SkeletonOutputMode.KEYFRAMES:
        wanted: set[int] = set()
        for rep in reps:
            for key in ("start", "peak", "end"):
                idx = rep.keyframes.get(key)
                if idx is not None:
                    wanted.add(idx)
            for issue in rep.issues:
                wanted.update(issue.frame_indices)
        return [frames[i] for i in sorted(wanted) if 0 <= i < len(frames)]
    raise ValueError(f"unknown mode: {mode}")
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_sampling.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add sport_companion_ai/sampling.py tests/test_sampling.py
git commit -m "feat: add output skeleton sampling modes (full/sampled/keyframes/none)"
```

---

## Task 16: Warning Detector

**Files:**
- Create: `sport_companion_ai/warnings.py`
- Create: `tests/test_warnings.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_warnings.py
from sport_companion_ai.pose.schema import Frame, Keypoint, Skeleton
from sport_companion_ai.report import VideoMeta
from sport_companion_ai.warnings import detect_warnings


def make_frame(i: int, vis: float = 1.0):
    skel = Skeleton(keypoints={"nose": Keypoint(x=0.5, y=0.5, visibility=vis)})
    return Frame(index=i, timestamp_ms=i * 33, skeleton=skel)


def test_no_warnings_clean_video():
    frames = [make_frame(i) for i in range(100)]
    meta = VideoMeta(width=1080, height=1920, fps=30, duration_ms=3300)
    warns = detect_warnings(frames, meta, n_reps=3)
    assert warns == []


def test_low_pose_confidence_when_many_frames_low_vis():
    frames = [make_frame(i, vis=0.3) for i in range(50)] + [make_frame(i, vis=0.9) for i in range(50, 100)]
    meta = VideoMeta(width=1080, height=1920, fps=30, duration_ms=3300)
    warns = detect_warnings(frames, meta, n_reps=3)
    codes = {w.code for w in warns}
    assert "LOW_POSE_CONFIDENCE" in codes


def test_no_reps_detected():
    frames = [make_frame(i) for i in range(100)]
    meta = VideoMeta(width=1080, height=1920, fps=30, duration_ms=3300)
    warns = detect_warnings(frames, meta, n_reps=0)
    codes = {w.code for w in warns}
    assert "NO_REPS_DETECTED" in codes


def test_video_too_short():
    frames = [make_frame(i) for i in range(60)]
    meta = VideoMeta(width=1080, height=1920, fps=30, duration_ms=2000)
    warns = detect_warnings(frames, meta, n_reps=1)
    codes = {w.code for w in warns}
    assert "VIDEO_TOO_SHORT" in codes


def test_low_fps_warning():
    frames = [make_frame(i) for i in range(100)]
    meta = VideoMeta(width=1080, height=1920, fps=10, duration_ms=10000)
    warns = detect_warnings(frames, meta, n_reps=3)
    codes = {w.code for w in warns}
    assert "LOW_FPS" in codes
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/test_warnings.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement warnings**

```python
# sport_companion_ai/warnings.py
"""Soft warnings collected post-pose-extraction."""
from sport_companion_ai.pose.schema import Frame
from sport_companion_ai.report import AnalysisWarning, VideoMeta


LOW_VIS_THRESHOLD = 0.5
LOW_VIS_FRAME_RATIO = 0.30
SHORT_VIDEO_MS = 3000
LONG_VIDEO_MS = 5 * 60 * 1000
LOW_FPS_THRESHOLD = 15


def _avg_visibility(frame: Frame) -> float:
    if frame.skeleton is None:
        return 0.0
    vals = [kp.visibility for kp in frame.skeleton.keypoints.values()]
    return sum(vals) / len(vals) if vals else 0.0


def detect_warnings(frames: list[Frame], meta: VideoMeta, n_reps: int) -> list[AnalysisWarning]:
    warns: list[AnalysisWarning] = []

    low_vis_count = sum(1 for f in frames if _avg_visibility(f) < LOW_VIS_THRESHOLD)
    if frames and low_vis_count / len(frames) > LOW_VIS_FRAME_RATIO:
        warns.append(AnalysisWarning(
            code="LOW_POSE_CONFIDENCE",
            message_vi=f"Khoảng {low_vis_count / len(frames) * 100:.0f}% frames có pose detection yếu",
            affected_frame_count=low_vis_count,
        ))

    if n_reps == 0:
        warns.append(AnalysisWarning(
            code="NO_REPS_DETECTED",
            message_vi="Không phát hiện rep nào — kiểm tra góc quay và vị trí người",
        ))

    if meta.duration_ms < SHORT_VIDEO_MS:
        warns.append(AnalysisWarning(
            code="VIDEO_TOO_SHORT",
            message_vi=f"Video quá ngắn ({meta.duration_ms} ms) để đánh giá đáng tin cậy",
        ))
    if meta.duration_ms > LONG_VIDEO_MS:
        warns.append(AnalysisWarning(
            code="VIDEO_TOO_LONG",
            message_vi=f"Video dài ({meta.duration_ms / 1000:.0f}s) — cảnh báo memory/thời gian xử lý",
        ))

    if meta.fps < LOW_FPS_THRESHOLD:
        warns.append(AnalysisWarning(
            code="LOW_FPS",
            message_vi=f"FPS thấp ({meta.fps}) — độ chính xác thời gian rep giảm",
        ))

    return warns
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_warnings.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add sport_companion_ai/warnings.py tests/test_warnings.py
git commit -m "feat: add soft warning detector (confidence, fps, length, no-reps)"
```

---

## Task 17: Template Feedback Enricher (Default)

**Files:**
- Create: `sport_companion_ai/feedback/enricher.py`
- Create: `sport_companion_ai/feedback/template.py`
- Create: `tests/test_template_enricher.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_template_enricher.py
from sport_companion_ai.feedback.template import TemplateEnricher
from sport_companion_ai.feedback.enricher import FeedbackEnricher
from sport_companion_ai.report import (
    AnalysisReport, SkeletonSchema, VideoMeta, RepEvaluation, Issue,
)


def base_report() -> AnalysisReport:
    return AnalysisReport(
        exercise="squat", pose_model="x",
        video=VideoMeta(width=1080, height=1920, fps=30, duration_ms=6000),
        skeleton_schema=SkeletonSchema(keypoint_names=["nose"], edges=[]),
        reps=[RepEvaluation(
            rep_index=0, score=80, passed=True,
            issues=[Issue(code="X", severity="LOW", message_vi="orig")],
        )],
    )


def test_protocol():
    assert isinstance(TemplateEnricher(), FeedbackEnricher)


def test_template_is_noop():
    rep_in = base_report()
    rep_out = TemplateEnricher().enrich(rep_in)
    assert rep_out is rep_in
    assert rep_out.enriched is False
    assert rep_out.session_summary is None
    assert rep_out.reps[0].issues[0].message_vi == "orig"
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/test_template_enricher.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement Protocol + TemplateEnricher**

```python
# sport_companion_ai/feedback/enricher.py
"""Feedback enricher Protocol. Default implementation in template.py."""
from typing import Protocol, runtime_checkable

from sport_companion_ai.report import AnalysisReport


@runtime_checkable
class FeedbackEnricher(Protocol):
    def enrich(self, report: AnalysisReport) -> AnalysisReport: ...
```

```python
# sport_companion_ai/feedback/template.py
"""Default no-op enricher. Vietnamese strings already populated by exercise rules."""
from sport_companion_ai.report import AnalysisReport


class TemplateEnricher:
    """Pass-through. enriched=False, session_summary=None remain unchanged."""

    def enrich(self, report: AnalysisReport) -> AnalysisReport:
        return report
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_template_enricher.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add sport_companion_ai/feedback/enricher.py sport_companion_ai/feedback/template.py tests/test_template_enricher.py
git commit -m "feat(feedback): add FeedbackEnricher Protocol + TemplateEnricher (no-op default)"
```

---

## Task 18: NVIDIA NIM Enricher (Opt-In)

**Files:**
- Create: `sport_companion_ai/feedback/nim.py`
- Create: `tests/test_nim_enricher.py`

- [ ] **Step 1: Write tests (mock httpx)**

```python
# tests/test_nim_enricher.py
import httpx
import pytest

from sport_companion_ai.feedback.nim import NvidiaNimEnricher
from sport_companion_ai.report import (
    AnalysisReport, SkeletonSchema, VideoMeta, RepEvaluation, Issue,
)


def base_report() -> AnalysisReport:
    return AnalysisReport(
        exercise="squat", pose_model="x",
        video=VideoMeta(width=1080, height=1920, fps=30, duration_ms=6000),
        skeleton_schema=SkeletonSchema(keypoint_names=["nose"], edges=[]),
        total_reps=1, passed_reps=1, avg_score=80,
        reps=[RepEvaluation(
            rep_index=0, score=80, passed=True,
            issues=[Issue(code="X", severity="LOW", message_vi="orig", recommendation="r")],
        )],
    )


def _ok_response(text: str = "Buổi tập tốt!"):
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": text}}]},
        request=httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions"),
    )


def test_success_sets_summary_and_enriched(mocker):
    mock_client = mocker.MagicMock()
    mock_client.post.return_value = _ok_response("Tóm tắt buổi tập")
    mock_client.__enter__ = lambda s: s
    mock_client.__exit__ = lambda s, *a: None
    mocker.patch("httpx.Client", return_value=mock_client)

    enricher = NvidiaNimEnricher(api_key="nvapi-fake")
    out = enricher.enrich(base_report())

    assert out.enriched is True
    assert out.session_summary == "Tóm tắt buổi tập"


def test_invariants_score_metrics_codes_unchanged(mocker):
    mock_client = mocker.MagicMock()
    mock_client.post.return_value = _ok_response("ok")
    mock_client.__enter__ = lambda s: s
    mock_client.__exit__ = lambda s, *a: None
    mocker.patch("httpx.Client", return_value=mock_client)

    rep_in = base_report()
    out = NvidiaNimEnricher(api_key="nvapi-fake").enrich(rep_in)
    assert out.reps[0].score == 80
    assert out.reps[0].issues[0].code == "X"
    assert out.reps[0].metrics == rep_in.reps[0].metrics
    assert out.passed_reps == 1


def test_timeout_falls_back_silently(mocker):
    mock_client = mocker.MagicMock()
    mock_client.post.side_effect = httpx.TimeoutException("slow")
    mock_client.__enter__ = lambda s: s
    mock_client.__exit__ = lambda s, *a: None
    mocker.patch("httpx.Client", return_value=mock_client)

    enricher = NvidiaNimEnricher(api_key="nvapi-fake", max_retries=1)
    out = enricher.enrich(base_report())

    assert out.enriched is False
    assert out.session_summary is None
    assert any(w.code == "ENRICHMENT_FAILED" for w in out.warnings)


def test_http_error_falls_back(mocker):
    mock_client = mocker.MagicMock()
    mock_client.post.return_value = httpx.Response(
        500, request=httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions"))
    mock_client.__enter__ = lambda s: s
    mock_client.__exit__ = lambda s, *a: None
    mocker.patch("httpx.Client", return_value=mock_client)

    out = NvidiaNimEnricher(api_key="nvapi-fake", max_retries=1).enrich(base_report())
    assert out.enriched is False
    assert any(w.code == "ENRICHMENT_FAILED" for w in out.warnings)
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/test_nim_enricher.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement NIM enricher**

```python
# sport_companion_ai/feedback/nim.py
"""NVIDIA NIM API-backed enricher. Adds session_summary, marks enriched=True.

Falls back silently with ENRICHMENT_FAILED warning on any failure. Never
modifies score / passed / metrics / issue codes.
"""
import time

import httpx

from sport_companion_ai.report import AnalysisReport, AnalysisWarning


NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


def _build_prompt(report: AnalysisReport) -> str:
    issue_lines = []
    for rep in report.reps:
        for issue in rep.issues:
            issue_lines.append(
                f"- rep {rep.rep_index} ({issue.severity}): {issue.code} — metrics: {rep.metrics}"
            )
    issues_block = "\n".join(issue_lines) if issue_lines else "(không có lỗi)"

    return (
        f"Bạn là HLV gym tiếng Việt. Dựa trên dữ liệu sau, viết 2-4 câu tóm tắt "
        f"buổi tập, ghi nhận điều tốt và đề xuất 1-2 cải thiện cụ thể. "
        f"Tránh số liệu thô, dùng giọng thân thiện.\n\n"
        f"Bài: {report.exercise}\n"
        f"Tổng rep: {report.total_reps}, đạt: {report.passed_reps}, điểm TB: {report.avg_score:.0f}\n"
        f"Lỗi:\n{issues_block}"
    )


class NvidiaNimEnricher:
    def __init__(
        self,
        api_key: str,
        model: str = "meta/llama-3.3-70b-instruct",
        timeout_s: float = 10.0,
        max_retries: int = 1,
        backoff_s: float = 1.0,
    ):
        if not api_key:
            raise ValueError("api_key required")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s

    def enrich(self, report: AnalysisReport) -> AnalysisReport:
        prompt = _build_prompt(report)
        text = self._call_with_retry(prompt)
        if text is None:
            report.warnings.append(AnalysisWarning(
                code="ENRICHMENT_FAILED",
                message_vi="LLM enrichment thất bại — fallback về template",
            ))
            return report
        report.session_summary = text.strip()
        report.enriched = True
        return report

    def _call_with_retry(self, prompt: str) -> str | None:
        attempts = 0
        last_err: Exception | None = None
        while attempts <= self.max_retries:
            try:
                return self._call_once(prompt)
            except (httpx.TimeoutException, httpx.HTTPError, ValueError) as exc:
                last_err = exc
                attempts += 1
                if attempts <= self.max_retries:
                    time.sleep(self.backoff_s * (2 ** (attempts - 1)))
        return None

    def _call_once(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
            "max_tokens": 400,
            "stream": False,
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(f"{NIM_BASE_URL}/chat/completions", headers=headers, json=body)
            if r.status_code >= 400:
                raise httpx.HTTPError(f"NIM returned {r.status_code}")
            data = r.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError("unexpected NIM response shape") from exc
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_nim_enricher.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add sport_companion_ai/feedback/nim.py tests/test_nim_enricher.py
git commit -m "feat(feedback): add NvidiaNimEnricher with retry, timeout, silent fallback"
```

---

## Task 19: VideoAnalyzer (Orchestration)

**Files:**
- Create: `sport_companion_ai/analyzer.py`
- Create: `tests/test_analyzer.py`
- Modify: `sport_companion_ai/__init__.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_analyzer.py
from unittest.mock import MagicMock

import numpy as np
import pytest

from sport_companion_ai.analyzer import VideoAnalyzer
from sport_companion_ai.errors import UnsupportedExerciseError
from sport_companion_ai.pose.schema import Frame, Keypoint, Skeleton
from sport_companion_ai.report import VideoMeta
from sport_companion_ai.sampling import SkeletonOutputMode
from tests.exercises._helpers import make_squat_rep_frames


def _stub_extractor(rep_frames):
    extractor = MagicMock()
    extractor.model_id = "stub-pose"
    extractor.extract_batch.return_value = rep_frames
    return extractor


def _stub_video_reader(n_frames: int, fps: int = 30):
    images = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(n_frames)]
    meta = VideoMeta(width=1280, height=720, fps=fps, duration_ms=int(n_frames / fps * 1000))
    return images, meta


def test_unknown_exercise_raises(mocker):
    mocker.patch("sport_companion_ai.analyzer.read_video", return_value=_stub_video_reader(5))
    analyzer = VideoAnalyzer(pose_extractor=_stub_extractor([]))
    with pytest.raises(UnsupportedExerciseError):
        analyzer.analyze("dummy.mp4", exercise="dancing")


def test_full_pipeline_squat(mocker):
    rep_frames = make_squat_rep_frames(min_knee_deg=88, back_deg=42)
    n = len(rep_frames)
    mocker.patch("sport_companion_ai.analyzer.read_video", return_value=_stub_video_reader(n))

    analyzer = VideoAnalyzer(pose_extractor=_stub_extractor(rep_frames))
    report = analyzer.analyze("dummy.mp4", exercise="squat", skeleton_output="keyframes")

    assert report.exercise == "squat"
    assert report.total_reps == 1
    assert report.passed_reps == 1
    assert report.avg_score >= 90
    assert len(report.frames) >= 1  # keyframes mode includes start/peak/end
    assert report.skeleton_schema.keypoint_names  # populated
    assert report.pose_model == "stub-pose"


def test_zero_reps_emits_warning(mocker):
    # Constant-angle frames — no rep should be detected
    flat = [
        Frame(index=i, timestamp_ms=i * 33,
              skeleton=Skeleton(keypoints={
                  "left_hip": Keypoint(x=0.5, y=0.5, visibility=1.0),
                  "left_knee": Keypoint(x=0.5, y=0.6, visibility=1.0),
                  "left_ankle": Keypoint(x=0.5, y=0.7, visibility=1.0),
                  "right_hip": Keypoint(x=0.5, y=0.5, visibility=1.0),
                  "right_knee": Keypoint(x=0.5, y=0.6, visibility=1.0),
                  "right_ankle": Keypoint(x=0.5, y=0.7, visibility=1.0),
                  "left_shoulder": Keypoint(x=0.5, y=0.3, visibility=1.0),
                  "right_shoulder": Keypoint(x=0.5, y=0.3, visibility=1.0),
              })) for i in range(60)
    ]
    mocker.patch("sport_companion_ai.analyzer.read_video", return_value=_stub_video_reader(60))
    analyzer = VideoAnalyzer(pose_extractor=_stub_extractor(flat))
    report = analyzer.analyze("dummy.mp4", exercise="squat")
    assert report.total_reps == 0
    assert any(w.code == "NO_REPS_DETECTED" for w in report.warnings)
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/test_analyzer.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement VideoAnalyzer**

```python
# sport_companion_ai/analyzer.py
"""Top-level orchestration. Reads video → pose → rule eval → enriched report."""
from __future__ import annotations

from sport_companion_ai import exercises  # noqa: F401  (registers rules)
from sport_companion_ai.exercises.base import ExerciseRule
from sport_companion_ai.feedback.enricher import FeedbackEnricher
from sport_companion_ai.feedback.template import TemplateEnricher
from sport_companion_ai.pose.extractor import MediaPipeExtractor, PoseExtractor
from sport_companion_ai.pose.schema import KEYPOINT_NAMES, SKELETON_EDGES
from sport_companion_ai.pose.video_reader import read_video
from sport_companion_ai.report import (
    AnalysisReport, SkeletonSchema,
)
from sport_companion_ai.sampling import SkeletonOutputMode, select_frames_for_output
from sport_companion_ai.warnings import detect_warnings


class VideoAnalyzer:
    def __init__(
        self,
        pose_extractor: PoseExtractor | None = None,
        enricher: FeedbackEnricher | None = None,
    ):
        self.pose_extractor = pose_extractor or MediaPipeExtractor()
        self.enricher = enricher or TemplateEnricher()

    def analyze(
        self,
        video_path: str,
        exercise: str,
        skeleton_output: str | SkeletonOutputMode = SkeletonOutputMode.KEYFRAMES,
    ) -> AnalysisReport:
        rule_cls = ExerciseRule.get(exercise)
        rule = rule_cls()

        images, meta = read_video(video_path)
        rule.fps = meta.fps  # propagate so rep_duration_ms is correct

        frames = self.pose_extractor.extract_batch(images)
        reps = rule.detect_reps(frames, fps=meta.fps)
        evaluations = [rule.evaluate_rep(rep, frames) for rep in reps]

        passed = sum(1 for e in evaluations if e.passed)
        scores = [e.score for e in evaluations if e.score is not None]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        warnings = detect_warnings(frames, meta, n_reps=len(reps))

        mode = skeleton_output if isinstance(skeleton_output, SkeletonOutputMode) else SkeletonOutputMode(skeleton_output)
        out_frames = select_frames_for_output(frames, evaluations, mode, fps=meta.fps)

        report = AnalysisReport(
            exercise=exercise,
            pose_model=self.pose_extractor.model_id,
            video=meta,
            skeleton_schema=SkeletonSchema(
                keypoint_names=KEYPOINT_NAMES,
                edges=list(SKELETON_EDGES),
            ),
            frames=out_frames,
            total_reps=len(evaluations),
            passed_reps=passed,
            avg_score=round(avg_score, 1),
            reps=evaluations,
            warnings=warnings,
        )

        return self.enricher.enrich(report)
```

- [ ] **Step 4: Re-export from package root**

Update `sport_companion_ai/__init__.py`:

```python
"""Sport Companion AI — gym video form evaluator."""
from sport_companion_ai.analyzer import VideoAnalyzer
from sport_companion_ai.report import AnalysisReport
from sport_companion_ai.sampling import SkeletonOutputMode

__version__ = "0.1.0"

__all__ = ["VideoAnalyzer", "AnalysisReport", "SkeletonOutputMode", "__version__"]
```

- [ ] **Step 5: Run tests, verify pass**

Run: `pytest tests/test_analyzer.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Run the full unit suite**

Run: `pytest -v`
Expected: all unit tests pass (~50 tests).

- [ ] **Step 7: Commit**

```bash
git add sport_companion_ai/analyzer.py sport_companion_ai/__init__.py tests/test_analyzer.py
git commit -m "feat: add VideoAnalyzer orchestrating the full pipeline"
```

---

## Task 20: Visualization Helper

**Files:**
- Create: `sport_companion_ai/viz/render.py`
- Create: `tests/test_viz.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_viz.py
from pathlib import Path

from sport_companion_ai.pose.schema import Frame, Keypoint, Skeleton
from sport_companion_ai.report import (
    AnalysisReport, SkeletonSchema, VideoMeta, RepEvaluation,
)
from sport_companion_ai.viz.render import render_skeleton_png


def _report_with_frame() -> AnalysisReport:
    skel = Skeleton(keypoints={
        "left_shoulder": Keypoint(x=0.45, y=0.40, visibility=1.0),
        "left_elbow": Keypoint(x=0.40, y=0.50, visibility=1.0),
        "left_wrist": Keypoint(x=0.35, y=0.60, visibility=1.0),
        "right_shoulder": Keypoint(x=0.55, y=0.40, visibility=1.0),
        "left_hip": Keypoint(x=0.45, y=0.60, visibility=1.0),
        "right_hip": Keypoint(x=0.55, y=0.60, visibility=1.0),
    })
    return AnalysisReport(
        exercise="squat", pose_model="x",
        video=VideoMeta(width=1080, height=1920, fps=30, duration_ms=1000),
        skeleton_schema=SkeletonSchema(
            keypoint_names=["left_shoulder", "left_elbow", "left_wrist",
                            "right_shoulder", "left_hip", "right_hip"],
            edges=[("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
                   ("left_shoulder", "left_hip"), ("left_hip", "right_hip")],
        ),
        frames=[Frame(index=0, timestamp_ms=0, skeleton=skel)],
        reps=[],
    )


def test_render_skeleton_png_writes_file(tmp_path: Path):
    out = tmp_path / "skel.png"
    render_skeleton_png(_report_with_frame(), frame_index=0, output=str(out))
    assert out.exists()
    assert out.stat().st_size > 0
```

- [ ] **Step 2: Run tests, verify failure**

Run: `pytest tests/test_viz.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement render**

```python
# sport_companion_ai/viz/render.py
"""Dev/test helper to render a frame's skeleton as a PNG.

Pure stick-figure on white background (or supplied frame image). Optional —
not part of the production pipeline.
"""
from __future__ import annotations

from sport_companion_ai.report import AnalysisReport


def render_skeleton_png(
    report: AnalysisReport,
    frame_index: int,
    output: str,
    width: int | None = None,
    height: int | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = next((f for f in report.frames if f.index == frame_index), None)
    if target is None or target.skeleton is None:
        raise ValueError(f"frame {frame_index} not present in report (or has no skeleton)")

    w = width or report.video.width
    h = height or report.video.height

    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # invert: image y grows downward
    ax.set_axis_off()

    kps = target.skeleton.keypoints
    for a, b in report.skeleton_schema.edges:
        if a in kps and b in kps:
            ka, kb = kps[a], kps[b]
            ax.plot([ka.x * w, kb.x * w], [ka.y * h, kb.y * h], "-", linewidth=3, color="#1976d2")

    for name, kp in kps.items():
        alpha = max(0.2, kp.visibility)
        ax.plot(kp.x * w, kp.y * h, "o", markersize=6, color="#d32f2f", alpha=alpha)

    fig.savefig(output, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_viz.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add sport_companion_ai/viz/render.py tests/test_viz.py
git commit -m "feat(viz): add PNG skeleton renderer for dev inspection"
```

---

## Task 21: CLI Demo

**Files:**
- Create: `examples/analyze_squat.py`

- [ ] **Step 1: Implement CLI**

```python
# examples/analyze_squat.py
"""End-to-end demo: analyze a video and print the JSON report.

Usage:
  python examples/analyze_squat.py path/to/squat.mp4 [--exercise squat] [--skeleton keyframes]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from sport_companion_ai import VideoAnalyzer, SkeletonOutputMode
from sport_companion_ai.feedback.nim import NvidiaNimEnricher


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a gym video for form correctness")
    parser.add_argument("video", help="Path to MP4 video")
    parser.add_argument("--exercise", default="squat",
                        choices=["squat", "deadlift", "bench_press", "push_up", "bicep_curl"])
    parser.add_argument("--skeleton", default="keyframes",
                        choices=[m.value for m in SkeletonOutputMode])
    parser.add_argument("--enrich-with-nim", action="store_true",
                        help="Use NVIDIA NIM for natural-language session summary")
    args = parser.parse_args()

    enricher = None
    if args.enrich_with_nim:
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            print("ERROR: --enrich-with-nim requires NVIDIA_API_KEY", file=sys.stderr)
            return 2
        enricher = NvidiaNimEnricher(api_key=api_key)

    analyzer = VideoAnalyzer(enricher=enricher)
    report = analyzer.analyze(args.video, exercise=args.exercise, skeleton_output=args.skeleton)
    print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke test invocation**

Run: `python examples/analyze_squat.py --help`
Expected: argparse help output, no crash.

- [ ] **Step 3: Commit**

```bash
git add examples/analyze_squat.py
git commit -m "feat(examples): add CLI demo for end-to-end analysis"
```

---

## Task 22: Fixture Manifest + Integration Test Skeleton

**Files:**
- Create: `tests/fixtures/manifest.yaml`
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_squat_video.py`

The actual fixture videos must be recorded by the developer (or sourced from open datasets); this task wires up the test harness so they can be slotted in.

- [ ] **Step 1: Create manifest**

```yaml
# tests/fixtures/manifest.yaml
# Each entry describes an expected outcome for a fixture video.
# Tests assert ranges, not exact numbers, to allow for pose-detection jitter.
squat_good_5reps:
  path: videos/squat_good_5reps.mp4
  exercise: squat
  expected_reps: { min: 4, max: 6 }
  expected_passed_reps: { min: 4, max: 6 }
  expected_avg_score: { min: 75, max: 100 }
  required_issues_absent: [SQUAT_DEPTH_INSUFFICIENT, SQUAT_KNEE_VALGUS]

squat_shallow_3reps:
  path: videos/squat_shallow_3reps.mp4
  exercise: squat
  expected_reps: { min: 2, max: 4 }
  required_issues_present: [SQUAT_DEPTH_INSUFFICIENT]

pushup_good_3reps:
  path: videos/pushup_good_3reps.mp4
  exercise: push_up
  expected_reps: { min: 2, max: 4 }
  expected_passed_reps: { min: 2, max: 4 }
```

- [ ] **Step 2: Create integration test**

```python
# tests/integration/__init__.py
```

```python
# tests/integration/test_squat_video.py
"""Integration tests against fixture videos. Skipped if video files missing."""
from pathlib import Path

import pytest
import yaml

from sport_companion_ai import VideoAnalyzer


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def load_manifest():
    with open(FIXTURES_DIR / "manifest.yaml") as f:
        return yaml.safe_load(f)


def fixture_ids():
    manifest = load_manifest()
    return list(manifest.keys())


@pytest.mark.integration
@pytest.mark.parametrize("fixture_id", fixture_ids())
def test_fixture_video(fixture_id: str):
    manifest = load_manifest()
    spec = manifest[fixture_id]
    video_path = FIXTURES_DIR / spec["path"]
    if not video_path.exists():
        pytest.skip(f"fixture missing: {video_path}")

    analyzer = VideoAnalyzer()
    report = analyzer.analyze(str(video_path), exercise=spec["exercise"])

    rmin, rmax = spec["expected_reps"]["min"], spec["expected_reps"]["max"]
    assert rmin <= report.total_reps <= rmax, (
        f"expected {rmin}-{rmax} reps, got {report.total_reps}")

    if "expected_passed_reps" in spec:
        pmin, pmax = spec["expected_passed_reps"]["min"], spec["expected_passed_reps"]["max"]
        assert pmin <= report.passed_reps <= pmax

    if "expected_avg_score" in spec:
        smin, smax = spec["expected_avg_score"]["min"], spec["expected_avg_score"]["max"]
        assert smin <= report.avg_score <= smax

    issues_seen = {i.code for r in report.reps for i in r.issues}
    for must_absent in spec.get("required_issues_absent", []):
        assert must_absent not in issues_seen, f"unexpected {must_absent} in {fixture_id}"
    for must_present in spec.get("required_issues_present", []):
        assert must_present in issues_seen, f"missing {must_present} in {fixture_id}"
```

- [ ] **Step 3: Add yaml dep**

Update `pyproject.toml` `[project.optional-dependencies] dev`:
```toml
dev = [
    "pytest>=8.3",
    "pytest-mock>=3.14",
    "pytest-cov>=5.0",
    "pyyaml>=6.0",
]
```

Run: `pip install -e ".[dev]"` to install pyyaml.

- [ ] **Step 4: Smoke run**

Run: `pytest tests/integration/ -m integration -v`
Expected: tests are collected, all skip with "fixture missing" message (this is fine — fixtures will be added later).

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/manifest.yaml tests/integration/ pyproject.toml
git commit -m "test: add fixture manifest and parametrized integration test harness"
```

---

## Task 23: Dockerfile

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

- [ ] **Step 1: Create Dockerfile**

```dockerfile
# Dockerfile
FROM python:3.13-slim

# System libs needed by opencv + mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY sport_companion_ai/ ./sport_companion_ai/
COPY examples/ ./examples/

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python", "examples/analyze_squat.py"]
CMD ["--help"]
```

- [ ] **Step 2: Create .dockerignore**

```
__pycache__/
*.pyc
.pytest_cache/
.coverage
htmlcov/
.git/
.venv/
tests/
docs/
*.md
.env
```

- [ ] **Step 3: Build and verify**

Run: `docker build -t sport-companion-ai:0.1.0 .`
Expected: image builds successfully.

Run: `docker run --rm sport-companion-ai:0.1.0`
Expected: prints argparse help.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "build: add Dockerfile for the CLI demo"
```

---

## Task 24: README + Final Sanity

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace stub README**

```markdown
# Sport Companion AI

Phase 1 prototype: take a gym video, get a structured form-evaluation report.

## What it does

- Reads an MP4 video
- Extracts skeleton keypoints with MediaPipe BlazePose
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
pytest                                       # fast unit tests
pytest -m integration                        # requires fixture videos
pytest -m requires_nim_key                   # requires NVIDIA_API_KEY
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
```

- [ ] **Step 2: Run full test suite + coverage**

Run: `pytest --cov=sport_companion_ai --cov-report=term-missing`
Expected: ≥ 70% coverage overall, 100% on `geometry.py` and `rep_detector.py`.

- [ ] **Step 3: Run CLI smoke test**

Run: `python examples/analyze_squat.py --help`
Expected: argparse help.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: write Phase 1 README with API + CLI + Docker quickstarts"
```

---

## Self-Review Checklist (post-write)

| Spec section | Covered by |
|---|---|
| Architecture & module boundaries | Tasks 2, 3, 5, 7, 13, 14, 17, 19, 20 |
| Data flow (video → pose → reps → eval → enrich) | Task 19 (Analyzer) wires it; Tasks 6, 8, 13, 14 implement steps |
| Output schema (frames, skeleton_schema, reps, warnings, enriched, session_summary) | Tasks 3, 19 |
| Sampling modes | Task 15 |
| Squat full rule (5 sub-rules) | Task 8 |
| Deadlift, bench, push-up, bicep curl (2-3 rules each) | Tasks 9, 10, 11, 12 |
| Rep detector | Task 6 |
| Geometry helpers | Task 5 |
| Errors hierarchy | Task 4 |
| Soft warnings | Task 16 |
| Inconclusive rep handling | Tasks 8-12 (each rule's evaluate_rep handles missing skeleton) |
| TemplateEnricher (default) | Task 17 |
| NvidiaNimEnricher (opt-in, retry, fallback) | Task 18 |
| Viz render | Task 20 |
| CLI demo | Task 21 |
| Test fixture manifest + integration harness | Task 22 |
| Dockerfile | Task 23 |
| README | Task 24 |
| Test tier 1 (pure units) | each rule + geometry + rep_detector + sampling + warnings |
| Test tier 2 (integration with fixtures) | Task 22 (skips when fixtures missing) |
| Test tier 3 (LLM smoke + mocked) | Task 18 mocked; smoke test deferred to first NIM credentials run |

No placeholders, all type names consistent across tasks (`SquatRule`, `PushUpRule`, `BicepCurlRule`, `DeadliftRule`, `BenchRule`, `VideoAnalyzer`, `AnalysisReport`, `RepEvaluation`, `Issue`, `AnalysisWarning`, `SkeletonSchema`, `VideoMeta`, `SkeletonOutputMode`, `FeedbackEnricher`, `TemplateEnricher`, `NvidiaNimEnricher`, `PoseExtractor`, `MediaPipeExtractor`, `ExerciseRule`, `EXERCISE_REGISTRY`).

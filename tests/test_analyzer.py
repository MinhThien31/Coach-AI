from unittest.mock import MagicMock

import numpy as np
import pytest

from sport_companion_ai.analyzer import VideoAnalyzer
from sport_companion_ai.errors import UnsupportedExerciseError
from sport_companion_ai.pose.schema import Frame, Keypoint, Skeleton
from sport_companion_ai.report import VideoMeta
from tests.exercises._helpers import make_squat_rep_frames
from tests.exercises.test_new_rules import make_badminton_frames, make_plank_frames, make_press_frames


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
    assert report.skeleton_schema.keypoint_names
    assert report.pose_model == "stub-pose"


def test_zero_reps_emits_warning(mocker):
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


def test_full_pipeline_overhead_press(mocker):
    rep_frames = make_press_frames()
    n = len(rep_frames)
    mocker.patch("sport_companion_ai.analyzer.read_video", return_value=_stub_video_reader(n))

    analyzer = VideoAnalyzer(pose_extractor=_stub_extractor(rep_frames))
    report = analyzer.analyze("dummy.mp4", exercise="overhead_press")

    assert report.exercise == "overhead_press"
    assert report.total_reps == 1
    assert report.passed_reps == 1
    assert report.avg_score >= 90


def test_full_pipeline_plank_hold(mocker):
    rep_frames = make_plank_frames()
    n = len(rep_frames)
    mocker.patch("sport_companion_ai.analyzer.read_video", return_value=_stub_video_reader(n))

    analyzer = VideoAnalyzer(pose_extractor=_stub_extractor(rep_frames))
    report = analyzer.analyze("dummy.mp4", exercise="plank")

    assert report.exercise == "plank"
    assert report.total_reps == 1
    assert report.passed_reps == 1
    assert report.reps[0].metrics["hold_duration_ms"] >= 10000


def test_full_pipeline_badminton(mocker):
    rep_frames = make_badminton_frames()
    n = len(rep_frames)
    mocker.patch("sport_companion_ai.analyzer.read_video", return_value=_stub_video_reader(n))

    analyzer = VideoAnalyzer(pose_extractor=_stub_extractor(rep_frames))
    report = analyzer.analyze("dummy.mp4", exercise="badminton")

    assert report.exercise == "badminton"
    assert report.total_reps == 1
    assert report.passed_reps == 1
    assert "max_racket_shoulder_angle" in report.reps[0].metrics

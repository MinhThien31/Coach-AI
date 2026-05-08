from unittest.mock import MagicMock

import numpy as np

from sport_companion_ai.pose.extractor import MediaPipeExtractor, PoseExtractor


def test_protocol_satisfied():
    assert isinstance(MediaPipeExtractor(), PoseExtractor)


def _setup_landmarker_mock(mocker, has_pose: bool = True):
    """Patch PoseLandmarker.create_from_options + _ensure_model so no real model file is needed."""
    fake_lm = MagicMock(x=0.5, y=0.5, z=0.0, visibility=0.9)
    pose_landmarks = [[fake_lm] * 33] if has_pose else []
    fake_result = MagicMock(pose_landmarks=pose_landmarks)

    landmarker = MagicMock()
    landmarker.detect_for_video.return_value = fake_result
    landmarker.__enter__ = lambda self: self
    landmarker.__exit__ = lambda self, *a: None

    mocker.patch(
        "mediapipe.tasks.python.vision.PoseLandmarker.create_from_options",
        return_value=landmarker,
    )
    mocker.patch(
        "sport_companion_ai.pose.extractor._ensure_model",
        return_value="/tmp/fake_model.task",
    )
    return landmarker


def test_extract_batch_returns_one_frame_per_input(mocker):
    _setup_landmarker_mock(mocker, has_pose=True)

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
    _setup_landmarker_mock(mocker, has_pose=False)
    images = [np.zeros((720, 1280, 3), dtype=np.uint8)]
    frames = MediaPipeExtractor().extract_batch(images)
    assert frames[0].skeleton is None

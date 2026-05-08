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

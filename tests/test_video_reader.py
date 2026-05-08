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

"""Read a video file into a list of RGB numpy frames + metadata."""
from typing import Tuple

import cv2
import numpy as np

from sport_companion_ai.errors import VideoReadError
from sport_companion_ai.report import VideoMeta

# Re-export VideoMeta so callers can import it from here directly.
__all__ = ["read_video", "VideoMeta"]


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
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not frames:
            raise VideoReadError(f"decoded 0 frames from {path}")

        duration_ms = int(len(frames) / fps * 1000)
        meta = VideoMeta(width=width, height=height, fps=fps, duration_ms=duration_ms)
        return frames, meta
    finally:
        cap.release()

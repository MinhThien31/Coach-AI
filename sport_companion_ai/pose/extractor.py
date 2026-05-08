"""Pose extraction backends. MediaPipe is the Phase 1 default.

On first use, ``MediaPipeExtractor.extract_batch`` will download the
BlazePose Full model (~17 MB) from Google's CDN and cache it at
``~/.cache/sport_companion_ai/pose_landmarker_full.task``.  Subsequent calls
use the cached file and incur no network I/O.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from sport_companion_ai.errors import PoseExtractionError
from sport_companion_ai.pose.schema import Frame, Keypoint, Skeleton, KEYPOINT_NAMES


_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
_DEFAULT_CACHE = Path.home() / ".cache" / "sport_companion_ai"
_MODEL_FILENAME = "pose_landmarker_full.task"


def _ensure_model(cache_dir: Path = _DEFAULT_CACHE) -> Path:
    """Return the cached model path, downloading it if necessary."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / _MODEL_FILENAME
    if not target.exists():
        urllib.request.urlretrieve(_MODEL_URL, str(target))
    return target


@runtime_checkable
class PoseExtractor(Protocol):
    """Anything that turns video frames into typed Frames."""

    model_id: str

    def extract_batch(self, images: list[np.ndarray]) -> list[Frame]: ...


class MediaPipeExtractor:
    """BlazePose Full via MediaPipe Tasks API (PoseLandmarker).

    Landmark ordering from PoseLandmarker matches the original BlazePose 33-
    point topology, so ``KEYPOINT_NAMES`` indices remain correct.

    Parameters
    ----------
    fps:
        Frame-rate of the input video; used to compute ``timestamp_ms``.
    model_complexity:
        Retained for API compatibility (ignored by the Tasks API which always
        uses the full model specified by ``model_path``).
    min_detection_confidence:
        Minimum confidence threshold forwarded to all three Tasks API
        confidence knobs (detection, presence, tracking).
    model_path:
        Override the cached model path.  Useful for testing or offline use.
    """

    model_id: str = "mediapipe-blazepose-full"

    def __init__(
        self,
        fps: int = 30,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        model_path: str | None = None,
    ) -> None:
        self.fps = fps
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.model_path = model_path

    def _resolve_model_path(self) -> str:
        if self.model_path:
            return self.model_path
        return str(_ensure_model())

    def extract_batch(self, images: list[np.ndarray]) -> list[Frame]:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision as mp_vision
            from mediapipe.tasks.python import BaseOptions
        except ImportError as exc:
            raise PoseExtractionError("mediapipe Tasks API not available") from exc

        options = mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._resolve_model_path()),
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_detection_confidence,
        )

        with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
            frames: list[Frame] = []
            for i, img in enumerate(images):
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
                ts_ms = int(i / self.fps * 1000)
                result = landmarker.detect_for_video(mp_image, ts_ms)

                if not result.pose_landmarks:
                    skel = None
                else:
                    landmarks = result.pose_landmarks[0]  # first detected pose
                    kp_dict: dict[str, Keypoint] = {}
                    for idx, lm in enumerate(landmarks):
                        if idx >= len(KEYPOINT_NAMES):
                            break
                        name = KEYPOINT_NAMES[idx]
                        kp_dict[name] = Keypoint(
                            x=float(min(max(lm.x, 0.0), 1.0)),
                            y=float(min(max(lm.y, 0.0), 1.0)),
                            z=float(getattr(lm, "z", 0.0)),
                            visibility=float(
                                min(max(getattr(lm, "visibility", 0.0), 0.0), 1.0)
                            ),
                        )
                    skel = Skeleton(keypoints=kp_dict)

                frames.append(Frame(index=i, timestamp_ms=ts_ms, skeleton=skel))
            return frames

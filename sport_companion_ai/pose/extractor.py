"""Pose extraction backends. MediaPipe is the Phase 1 default."""
import sys
import types
from typing import Protocol, runtime_checkable

import numpy as np

from sport_companion_ai.errors import PoseExtractionError
from sport_companion_ai.pose.schema import Frame, Keypoint, Skeleton, KEYPOINT_NAMES


def _ensure_mediapipe_solutions() -> bool:
    """Import mediapipe and ensure ``mediapipe.solutions.pose`` is accessible.

    MediaPipe ≥ 0.10 removed the legacy ``solutions`` subpackage from the
    top-level namespace but still ships it as a sub-module.  We create a thin
    namespace shim so the conventional ``mp.solutions.pose.Pose`` path (and
    unittest.mock patch targets) work regardless of which MediaPipe version is
    installed.

    Returns True if mediapipe is available, False otherwise.
    """
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        return False

    # If ``solutions`` already exists as an attribute we have nothing to do.
    if hasattr(mediapipe, "solutions"):
        return True

    # Build a lightweight namespace shim so that:
    #   mediapipe.solutions.pose.Pose
    # resolves as an attribute chain AND as an importable module path
    # (the latter is required for unittest.mock.patch to resolve the target).
    solutions_mod = types.ModuleType("mediapipe.solutions")
    pose_mod = types.ModuleType("mediapipe.solutions.pose")
    # Provide a sentinel ``Pose`` class so unittest.mock.patch can locate and
    # replace the attribute (patch requires the attribute to already exist).
    pose_mod.Pose = None  # type: ignore[attr-defined]

    mediapipe.solutions = solutions_mod
    solutions_mod.pose = pose_mod
    sys.modules.setdefault("mediapipe.solutions", solutions_mod)
    sys.modules.setdefault("mediapipe.solutions.pose", pose_mod)

    return True


_MP_AVAILABLE: bool = _ensure_mediapipe_solutions()

# Keep a stable reference to the shim/real module so that mocker.patch on
# "mediapipe.solutions.pose.Pose" replaces the same object we call at runtime.
if _MP_AVAILABLE:
    import mediapipe as _mp  # noqa: E402 – after shim is in place
    _mp_pose_mod = _mp.solutions.pose
else:
    _mp_pose_mod = None  # type: ignore[assignment]


@runtime_checkable
class PoseExtractor(Protocol):
    """Anything that turns video frames into typed Frames."""
    model_id: str

    def extract_batch(self, images: list[np.ndarray]) -> list[Frame]: ...


class MediaPipeExtractor:
    """BlazePose Full via mediapipe.solutions.pose.

    Returns 33 keypoints in normalized [0,1] coordinates. Maps MediaPipe's
    integer indices onto our ``KEYPOINT_NAMES`` (same order as MediaPipe's).
    """

    model_id: str = "mediapipe-blazepose-full"

    def __init__(self, fps: int = 30, model_complexity: int = 1,
                 min_detection_confidence: float = 0.5):
        self.fps = fps
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence

    def extract_batch(self, images: list[np.ndarray]) -> list[Frame]:
        if not _MP_AVAILABLE:
            raise PoseExtractionError("mediapipe not installed")

        # Access through the module reference so mocker.patch can intercept.
        import mediapipe as mp
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

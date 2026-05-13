"""Reduce the size of `frames[]` in the output report."""
from enum import Enum

from sport_companion_ai.pose.schema import Frame
from sport_companion_ai.report import RepEvaluation


class SkeletonOutputMode(str, Enum):
    FULL = "full"
    SAMPLED = "sampled"
    KEYFRAMES = "keyframes"
    NONE = "none"


def _sample_frames(frames: list[Frame], fps: int) -> list[Frame]:
    step = max(1, fps // 5)
    sampled = [frames[i] for i in range(0, len(frames), step)]
    if any(frame.skeleton is not None for frame in sampled):
        return sampled
    skeleton_frames = [frame for frame in frames if frame.skeleton is not None]
    return skeleton_frames or sampled


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
        return _sample_frames(frames, fps)
    if mode is SkeletonOutputMode.KEYFRAMES:
        wanted: set[int] = set()
        for rep in reps:
            for key in ("start", "peak", "end"):
                idx = rep.keyframes.get(key)
                if idx is not None:
                    wanted.add(idx)
            for issue in rep.issues:
                wanted.update(issue.frame_indices)
        if not wanted:
            return _sample_frames(frames, fps)
        out = [frames[i] for i in sorted(wanted) if 0 <= i < len(frames)]
        if any(frame.skeleton is not None for frame in out):
            return out
        return _sample_frames(frames, fps)
    raise ValueError(f"unknown mode: {mode}")

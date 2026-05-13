"""Shared helpers for exercise rule implementations."""
from __future__ import annotations

import math
from collections.abc import Callable

from sport_companion_ai.pose.schema import Frame, Skeleton
from sport_companion_ai.report import Rep, RepEvaluation


def frames_with_keypoints(
    frames: list[Frame],
    rep: Rep,
    required: tuple[str, ...],
) -> list[Frame]:
    out: list[Frame] = []
    for frame in frames[rep.start_idx:rep.end_idx + 1]:
        if frame.skeleton is None:
            continue
        if all(name in frame.skeleton.keypoints for name in required):
            out.append(frame)
    return out


def inconclusive(rep: Rep, reason: str = "MISSING_KEYPOINTS") -> RepEvaluation:
    return RepEvaluation(
        rep_index=rep.rep_index,
        score=None,
        passed=None,
        inconclusive=True,
        inconclusive_reason=reason,
        keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
    )


def safe_series(
    frames: list[Frame],
    measure: Callable[[Skeleton], float],
) -> list[float]:
    out: list[float] = []
    for frame in frames:
        if frame.skeleton is None:
            out.append(float("nan"))
            continue
        try:
            out.append(measure(frame.skeleton))
        except KeyError:
            out.append(float("nan"))
    return out


def finite(values: list[float]) -> list[float]:
    return [value for value in values if not math.isnan(value)]


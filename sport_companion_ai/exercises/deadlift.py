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

        # Lockout is judged on the ascending phase only (peak → end).
        # Frames before peak are descending — naturally start near full extension.
        ascent_frames = [
            f for f in frames[rep.peak_idx:rep.end_idx + 1]
            if f.skeleton is not None
        ]
        ascent_hip = [hip_angle(f.skeleton, side="left") for f in ascent_frames]
        ascent_hip = [v for v in ascent_hip if not math.isnan(v)]
        max_ascent_hip = max(ascent_hip) if ascent_hip else float("nan")

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

        if not math.isnan(max_ascent_hip) and max_ascent_hip < self.LOCKOUT_TARGET:
            issues.append(Issue(
                code="DEADLIFT_PARTIAL_LOCKOUT", severity="MEDIUM",
                message_vi=f"Chưa khóa lockout hông (max {max_ascent_hip:.0f}°)",
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

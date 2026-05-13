"""Overhead press form evaluator."""
import math

from sport_companion_ai.exercises._common import (
    finite,
    frames_with_keypoints,
    inconclusive,
    safe_series,
)
from sport_companion_ai.exercises.base import ExerciseRule, register_rule
from sport_companion_ai.geometry import back_angle, elbow_angle
from sport_companion_ai.pose.schema import Frame
from sport_companion_ai.report import Issue, Rep, RepEvaluation


REQUIRED = (
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_hip", "right_hip",
)


@register_rule
class OverheadPressRule(ExerciseRule):
    name = "overhead_press"
    display_name_vi = "Overhead press"
    category = "upper_body_push"
    equipment = ["barbell", "dumbbell"]
    primary_joints = ["shoulder", "elbow", "spine"]
    issue_codes = ["OHP_PARTIAL_LOCKOUT", "OHP_BACK_LEAN", "OHP_ASYMMETRY"]
    primary_angle = "elbow_lockout"
    rep_threshold_low = 30.0
    rep_threshold_high = 90.0

    LOCKOUT_TARGET = 165.0
    BACK_LEAN_MAX = 18.0
    ASYMMETRY_THRESHOLD = 18.0

    def _primary_angle_series(self, frames: list[Frame]) -> list[float]:
        return safe_series(frames, lambda skel: 180.0 - elbow_angle(skel, side="left"))

    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation:
        rep_frames = frames_with_keypoints(frames, rep, REQUIRED)
        if not rep_frames:
            return inconclusive(rep)

        left = finite([elbow_angle(f.skeleton, side="left") for f in rep_frames])
        right = finite([elbow_angle(f.skeleton, side="right") for f in rep_frames])
        if not left:
            return inconclusive(rep)

        max_left = max(left)
        max_right = max(right) if right else max_left
        max_lockout = max(max_left, max_right)
        asymmetry = abs(max_left - max_right)
        peak_frame = frames[rep.peak_idx]
        lean = back_angle(peak_frame.skeleton) if peak_frame.skeleton else float("nan")

        issues: list[Issue] = []
        score = 100

        if max_lockout < self.LOCKOUT_TARGET:
            issues.append(Issue(
                code="OHP_PARTIAL_LOCKOUT",
                severity="MEDIUM",
                message_vi=f"Chua khoa het khuyu tay tren dau ({max_lockout:.0f} deg)",
                frame_indices=[rep.peak_idx],
                recommendation="Day ta len den khi khuyu gan thang, giu xuong suon chat.",
            ))
            score -= 20

        if not math.isnan(lean) and lean > self.BACK_LEAN_MAX:
            issues.append(Issue(
                code="OHP_BACK_LEAN",
                severity="MEDIUM",
                message_vi=f"Nga nguoi ra sau qua nhieu ({lean:.0f} deg)",
                frame_indices=[rep.peak_idx],
                recommendation="Siet mong va bung, dua dau qua cua so khi ta qua tran.",
            ))
            score -= 15

        if asymmetry > self.ASYMMETRY_THRESHOLD:
            issues.append(Issue(
                code="OHP_ASYMMETRY",
                severity="LOW",
                message_vi=f"Hai tay lockout lech nhau {asymmetry:.0f} deg",
                frame_indices=[rep.peak_idx],
                recommendation="Kiem tra grip va day hai ben cung toc do.",
            ))
            score -= 5

        score = max(0, score)
        return RepEvaluation(
            rep_index=rep.rep_index,
            score=score,
            passed=score >= 70 and not any(i.severity == "HIGH" for i in issues),
            issues=issues,
            metrics={
                "max_elbow_angle": round(max_lockout, 1),
                "back_lean_at_lockout": round(lean, 1) if not math.isnan(lean) else None,
                "left_right_asymmetry": round(asymmetry, 1),
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )


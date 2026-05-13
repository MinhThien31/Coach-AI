"""Lunge form evaluator."""
import math

from sport_companion_ai.exercises._common import (
    finite,
    frames_with_keypoints,
    inconclusive,
    safe_series,
)
from sport_companion_ai.exercises.base import ExerciseRule, register_rule
from sport_companion_ai.geometry import back_angle, knee_angle
from sport_companion_ai.pose.schema import Frame, Skeleton
from sport_companion_ai.report import Issue, Rep, RepEvaluation


REQUIRED = (
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "left_ankle",
)


def _front_knee_valgus(skel: Skeleton) -> float:
    hip = skel.keypoints["left_hip"]
    knee = skel.keypoints["left_knee"]
    ankle = skel.keypoints["left_ankle"]
    hip_width = abs(skel.keypoints["right_hip"].x - hip.x) or 1.0
    expected_x = (hip.x + ankle.x) / 2
    return abs(knee.x - expected_x) / hip_width


@register_rule
class LungeRule(ExerciseRule):
    name = "lunge"
    display_name_vi = "Lunge"
    category = "lower_body"
    equipment = ["bodyweight", "dumbbell", "barbell"]
    primary_joints = ["hip", "knee", "ankle"]
    issue_codes = ["LUNGE_DEPTH_INSUFFICIENT", "LUNGE_KNEE_VALGUS", "LUNGE_TORSO_LEAN"]
    primary_angle = "front_knee"
    rep_threshold_low = 105.0
    rep_threshold_high = 155.0

    DEPTH_TARGET = 105.0
    VALGUS_THRESHOLD = 0.18
    TORSO_LEAN_MAX = 55.0

    def _primary_angle_series(self, frames: list[Frame]) -> list[float]:
        return safe_series(frames, lambda skel: knee_angle(skel, side="left"))

    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation:
        rep_frames = frames_with_keypoints(frames, rep, REQUIRED)
        if not rep_frames:
            return inconclusive(rep)

        knees = finite([knee_angle(f.skeleton, side="left") for f in rep_frames])
        if not knees:
            return inconclusive(rep)

        min_knee = min(knees)
        peak_frame = frames[rep.peak_idx]
        peak_skel = peak_frame.skeleton
        valgus = _front_knee_valgus(peak_skel) if peak_skel else 0.0
        lean = back_angle(peak_skel) if peak_skel else float("nan")

        issues: list[Issue] = []
        score = 100

        if min_knee > self.DEPTH_TARGET:
            issues.append(Issue(
                code="LUNGE_DEPTH_INSUFFICIENT",
                severity="MEDIUM",
                message_vi=f"Buoc lunge chua du sau (goi {min_knee:.0f} deg)",
                frame_indices=[rep.peak_idx],
                recommendation="Ha nguoi den khi goi truoc gan 90 deg va goi sau gan san.",
            ))
            score -= 20

        if valgus > self.VALGUS_THRESHOLD:
            issues.append(Issue(
                code="LUNGE_KNEE_VALGUS",
                severity="HIGH",
                message_vi=f"Goi truoc lech vao trong (ratio {valgus:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Day goi theo huong mui chan va giu ban chan bam san.",
            ))
            score -= 25

        if not math.isnan(lean) and lean > self.TORSO_LEAN_MAX:
            issues.append(Issue(
                code="LUNGE_TORSO_LEAN",
                severity="MEDIUM",
                message_vi=f"Than nghieng qua nhieu ({lean:.0f} deg)",
                frame_indices=[rep.peak_idx],
                recommendation="Giu nguc mo va than nguoi cao khi ha xuong.",
            ))
            score -= 15

        score = max(0, score)
        return RepEvaluation(
            rep_index=rep.rep_index,
            score=score,
            passed=score >= 70 and not any(i.severity == "HIGH" for i in issues),
            issues=issues,
            metrics={
                "min_front_knee_angle": round(min_knee, 1),
                "front_knee_valgus_ratio": round(valgus, 3),
                "torso_lean_at_bottom": round(lean, 1) if not math.isnan(lean) else None,
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )


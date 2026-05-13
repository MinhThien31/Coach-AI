"""Lateral raise form evaluator."""

from sport_companion_ai.exercises._common import (
    finite,
    frames_with_keypoints,
    inconclusive,
    safe_series,
)
from sport_companion_ai.exercises.base import ExerciseRule, register_rule
from sport_companion_ai.geometry import shoulder_angle
from sport_companion_ai.pose.schema import Frame
from sport_companion_ai.report import Issue, Rep, RepEvaluation


REQUIRED = (
    "left_hip", "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_shoulder", "right_elbow", "right_wrist",
    "left_ear", "right_ear",
)


def _shrug_offset(skel) -> float:
    ear_y = (skel.keypoints["left_ear"].y + skel.keypoints["right_ear"].y) / 2
    shoulder_y = (skel.keypoints["left_shoulder"].y + skel.keypoints["right_shoulder"].y) / 2
    neck_gap = shoulder_y - ear_y
    return max(0.0, 0.16 - neck_gap)


@register_rule
class LateralRaiseRule(ExerciseRule):
    name = "lateral_raise"
    display_name_vi = "Lateral raise"
    category = "upper_body_accessory"
    equipment = ["dumbbell", "cable"]
    primary_joints = ["shoulder", "elbow"]
    issue_codes = [
        "LATERAL_RAISE_PARTIAL_ROM",
        "LATERAL_RAISE_SHRUG",
        "LATERAL_RAISE_TOO_HIGH",
    ]
    primary_angle = "shoulder_abduction"
    rep_threshold_low = 40.0
    rep_threshold_high = 100.0

    ROM_TARGET = 75.0
    TOO_HIGH_THRESHOLD = 115.0
    SHRUG_THRESHOLD = 0.06

    def _primary_angle_series(self, frames: list[Frame]) -> list[float]:
        return safe_series(frames, lambda skel: 120.0 - shoulder_angle(skel, side="left"))

    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation:
        rep_frames = frames_with_keypoints(frames, rep, REQUIRED)
        if not rep_frames:
            return inconclusive(rep)

        left = finite([shoulder_angle(f.skeleton, side="left") for f in rep_frames])
        right = finite([shoulder_angle(f.skeleton, side="right") for f in rep_frames])
        if not left:
            return inconclusive(rep)

        max_left = max(left)
        max_right = max(right) if right else max_left
        max_raise = max(max_left, max_right)
        max_shrug = max(_shrug_offset(f.skeleton) for f in rep_frames)

        issues: list[Issue] = []
        score = 100

        if max_raise < self.ROM_TARGET:
            issues.append(Issue(
                code="LATERAL_RAISE_PARTIAL_ROM",
                severity="MEDIUM",
                message_vi=f"Tay nang chua du cao ({max_raise:.0f} deg)",
                frame_indices=[rep.peak_idx],
                recommendation="Nang den gan ngang vai voi khuyu hoi mem.",
            ))
            score -= 15

        if max_shrug > self.SHRUG_THRESHOLD:
            issues.append(Issue(
                code="LATERAL_RAISE_SHRUG",
                severity="MEDIUM",
                message_vi=f"Co dau hieu nhun vai (offset {max_shrug:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Ha vai xuong, nghi den viec dua tay ra ngang thay vi keo bang co thang.",
            ))
            score -= 15

        if max_raise > self.TOO_HIGH_THRESHOLD:
            issues.append(Issue(
                code="LATERAL_RAISE_TOO_HIGH",
                severity="LOW",
                message_vi=f"Nang tay qua cao ({max_raise:.0f} deg)",
                frame_indices=[rep.peak_idx],
                recommendation="Dung o ngang vai hoac cao hon mot chut, khong can vuot dau.",
            ))
            score -= 5

        score = max(0, score)
        return RepEvaluation(
            rep_index=rep.rep_index,
            score=score,
            passed=score >= 70 and not any(i.severity == "HIGH" for i in issues),
            issues=issues,
            metrics={
                "max_shoulder_angle": round(max_raise, 1),
                "max_shrug_offset": round(max_shrug, 3),
                "left_right_asymmetry": round(abs(max_left - max_right), 1),
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )

"""Pull-up form evaluator."""
import math

from sport_companion_ai.exercises._common import (
    finite,
    frames_with_keypoints,
    inconclusive,
    safe_series,
)
from sport_companion_ai.exercises.base import ExerciseRule, register_rule
from sport_companion_ai.geometry import elbow_angle
from sport_companion_ai.pose.schema import Frame
from sport_companion_ai.report import Issue, Rep, RepEvaluation


REQUIRED = (
    "nose",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_hip", "right_hip",
)


def _body_swing(skel) -> float:
    shoulder_x = (skel.keypoints["left_shoulder"].x + skel.keypoints["right_shoulder"].x) / 2
    hip_x = (skel.keypoints["left_hip"].x + skel.keypoints["right_hip"].x) / 2
    return abs(hip_x - shoulder_x)


@register_rule
class PullUpRule(ExerciseRule):
    name = "pull_up"
    display_name_vi = "Pull-up"
    category = "upper_body_pull"
    equipment = ["pull_up_bar"]
    primary_joints = ["shoulder", "elbow", "core"]
    issue_codes = ["PULLUP_PARTIAL_ROM", "PULLUP_INCOMPLETE_TOP", "PULLUP_BODY_SWING"]
    primary_angle = "elbow"
    rep_threshold_low = 85.0
    rep_threshold_high = 155.0

    DEAD_HANG_TARGET = 155.0
    TOP_TARGET = 85.0
    BODY_SWING_THRESHOLD = 0.12

    def _primary_angle_series(self, frames: list[Frame]) -> list[float]:
        return safe_series(frames, lambda skel: elbow_angle(skel, side="left"))

    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation:
        rep_frames = frames_with_keypoints(frames, rep, REQUIRED)
        if not rep_frames:
            return inconclusive(rep)

        elbows = finite([elbow_angle(f.skeleton, side="left") for f in rep_frames])
        if not elbows:
            return inconclusive(rep)

        min_elbow = min(elbows)
        max_elbow = max(elbows)
        peak_frame = frames[rep.peak_idx]
        peak_skel = peak_frame.skeleton
        nose_y = peak_skel.keypoints["nose"].y if peak_skel else 1.0
        wrist_y = (
            (peak_skel.keypoints["left_wrist"].y + peak_skel.keypoints["right_wrist"].y) / 2
            if peak_skel else 0.0
        )
        max_swing = max(_body_swing(f.skeleton) for f in rep_frames)

        issues: list[Issue] = []
        score = 100

        if max_elbow < self.DEAD_HANG_TARGET:
            issues.append(Issue(
                code="PULLUP_PARTIAL_ROM",
                severity="MEDIUM",
                message_vi=f"Chua ve dead hang du ({max_elbow:.0f} deg)",
                frame_indices=[rep.start_idx, rep.end_idx],
                recommendation="Ha nguoi co kiem soat den gan thang tay truoc rep tiep theo.",
            ))
            score -= 15

        if min_elbow > self.TOP_TARGET or nose_y > wrist_y + 0.05:
            issues.append(Issue(
                code="PULLUP_INCOMPLETE_TOP",
                severity="MEDIUM",
                message_vi=f"Chua keo len du cao (khuyu nho nhat {min_elbow:.0f} deg)",
                frame_indices=[rep.peak_idx],
                recommendation="Keo nguc len gan xa, cam vuot qua tay nam neu co the.",
            ))
            score -= 20

        if max_swing > self.BODY_SWING_THRESHOLD:
            issues.append(Issue(
                code="PULLUP_BODY_SWING",
                severity="LOW",
                message_vi=f"Than nguoi vung qua nhieu (offset {max_swing:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Siet core va tam dung nhip truoc khi bat dau rep tiep theo.",
            ))
            score -= 5

        score = max(0, score)
        return RepEvaluation(
            rep_index=rep.rep_index,
            score=score,
            passed=score >= 70 and not any(i.severity == "HIGH" for i in issues),
            issues=issues,
            metrics={
                "min_elbow_angle": round(min_elbow, 1),
                "max_elbow_angle": round(max_elbow, 1),
                "body_swing_offset": round(max_swing, 3),
                "top_clearance": round(wrist_y - nose_y, 3),
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )


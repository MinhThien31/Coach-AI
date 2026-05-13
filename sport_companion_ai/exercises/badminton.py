"""Badminton stroke form evaluator.

Prototype scope: one-person side/front-ish camera analysis for repeated
overhead strokes such as clear/drop/smash. This keeps the same deterministic
rule engine contract as gym movements while giving the FE a sport-style entry.
"""
import math

from sport_companion_ai.exercises._common import (
    finite,
    frames_with_keypoints,
    inconclusive,
    safe_series,
)
from sport_companion_ai.exercises.base import ExerciseRule, register_rule
from sport_companion_ai.geometry import back_angle, elbow_angle, knee_angle, shoulder_angle
from sport_companion_ai.pose.schema import Frame, Skeleton
from sport_companion_ai.report import Issue, Rep, RepEvaluation


REQUIRED = (
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "left_hip", "right_hip",
    "left_knee", "left_ankle",
    "right_knee", "right_ankle",
)


def _front_knee_valgus(skel: Skeleton) -> float:
    """Approximate lead-leg knee collapse for a badminton lunge."""
    hip = skel.keypoints["left_hip"]
    knee = skel.keypoints["left_knee"]
    ankle = skel.keypoints["left_ankle"]
    hip_width = abs(skel.keypoints["right_hip"].x - hip.x) or 1.0
    expected_x = (hip.x + ankle.x) / 2
    return abs(knee.x - expected_x) / hip_width


def _contact_height_margin(skel: Skeleton) -> float:
    """Positive means racket-side wrist is above shoulder level."""
    shoulder_y = (
        skel.keypoints["left_shoulder"].y + skel.keypoints["right_shoulder"].y
    ) / 2
    wrist_y = skel.keypoints["right_wrist"].y
    return shoulder_y - wrist_y


@register_rule
class BadmintonRule(ExerciseRule):
    name = "badminton"
    display_name_vi = "Cầu lông"
    category = "racket_sport"
    equipment = ["racket", "shuttlecock"]
    primary_joints = ["shoulder", "elbow", "wrist", "hip", "knee"]
    issue_codes = [
        "BADMINTON_CONTACT_TOO_LOW",
        "BADMINTON_ELBOW_COLLAPSE",
        "BADMINTON_LUNGE_KNEE_VALGUS",
        "BADMINTON_TORSO_OVERLEAN",
    ]
    primary_angle = "racket_shoulder"
    rep_threshold_low = 35.0
    rep_threshold_high = 95.0

    CONTACT_HEIGHT_MIN = 0.08
    SHOULDER_CONTACT_MIN = 95.0
    ELBOW_EXTENSION_MIN = 145.0
    LUNGE_VALGUS_MAX = 0.18
    TORSO_LEAN_MAX = 65.0

    def _primary_angle_series(self, frames: list[Frame]) -> list[float]:
        return safe_series(frames, lambda skel: 130.0 - shoulder_angle(skel, side="right"))

    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation:
        rep_frames = frames_with_keypoints(frames, rep, REQUIRED)
        if not rep_frames:
            return inconclusive(rep)

        shoulder_series = finite([shoulder_angle(f.skeleton, side="right") for f in rep_frames])
        elbow_series = finite([elbow_angle(f.skeleton, side="right") for f in rep_frames])
        knee_series = finite([knee_angle(f.skeleton, side="left") for f in rep_frames])
        if not shoulder_series or not elbow_series:
            return inconclusive(rep)

        max_shoulder = max(shoulder_series)
        max_elbow = max(elbow_series)
        min_front_knee = min(knee_series) if knee_series else float("nan")

        contact_frame = frames[rep.peak_idx]
        contact_skel = contact_frame.skeleton
        if contact_skel is None:
            return inconclusive(rep)

        contact_margin = _contact_height_margin(contact_skel)
        valgus = _front_knee_valgus(contact_skel)
        torso_lean = back_angle(contact_skel)

        issues: list[Issue] = []
        score = 100

        if contact_margin < self.CONTACT_HEIGHT_MIN or max_shoulder < self.SHOULDER_CONTACT_MIN:
            issues.append(Issue(
                code="BADMINTON_CONTACT_TOO_LOW",
                severity="MEDIUM",
                message_vi=(
                    f"Diem cham cau hoi thap (wrist cao hon vai {contact_margin:.2f}, "
                    f"vai mo {max_shoulder:.0f} deg)"
                ),
                frame_indices=[rep.peak_idx],
                recommendation="Don cau som hon, dua tay danh len cao truoc dau de co goc dap/clear tot hon.",
            ))
            score -= 20

        if max_elbow < self.ELBOW_EXTENSION_MIN:
            issues.append(Issue(
                code="BADMINTON_ELBOW_COLLAPSE",
                severity="MEDIUM",
                message_vi=f"Khuyu tay chua duoi du o diem danh ({max_elbow:.0f} deg)",
                frame_indices=[rep.peak_idx],
                recommendation="Mo khuyu va vut co tay sau khi tiep xuc, tranh co tay danh mot minh.",
            ))
            score -= 15

        if valgus > self.LUNGE_VALGUS_MAX:
            issues.append(Issue(
                code="BADMINTON_LUNGE_KNEE_VALGUS",
                severity="HIGH",
                message_vi=f"Goi truoc lech vao trong khi buoc lunge (ratio {valgus:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Dat goi theo huong mui chan khi do nguoi bat cau, giu ban chan truoc bam san.",
            ))
            score -= 25

        if not math.isnan(torso_lean) and torso_lean > self.TORSO_LEAN_MAX:
            issues.append(Issue(
                code="BADMINTON_TORSO_OVERLEAN",
                severity="LOW",
                message_vi=f"Than nguoi do qua nhieu khi danh ({torso_lean:.0f} deg)",
                frame_indices=[rep.peak_idx],
                recommendation="Giu trong tam thap nhung nguc mo, phuc hoi ve giua san ngay sau cu danh.",
            ))
            score -= 5

        score = max(0, score)
        return RepEvaluation(
            rep_index=rep.rep_index,
            score=score,
            passed=score >= 70 and not any(i.severity == "HIGH" for i in issues),
            issues=issues,
            metrics={
                "max_racket_shoulder_angle": round(max_shoulder, 1),
                "max_racket_elbow_angle": round(max_elbow, 1),
                "contact_height_margin": round(contact_margin, 3),
                "front_knee_angle_min": round(min_front_knee, 1) if not math.isnan(min_front_knee) else None,
                "front_knee_valgus_ratio": round(valgus, 3),
                "torso_lean_at_contact": round(torso_lean, 1) if not math.isnan(torso_lean) else None,
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )


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

FOOTWORK_REQUIRED = (
    "left_shoulder", "right_shoulder",
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


def _stance_width_ratio(skel: Skeleton) -> float:
    shoulder_width = abs(skel.keypoints["right_shoulder"].x - skel.keypoints["left_shoulder"].x)
    if shoulder_width == 0:
        return 0.0
    ankle_width = abs(skel.keypoints["right_ankle"].x - skel.keypoints["left_ankle"].x)
    return ankle_width / shoulder_width


def _average_knee_angle(skel: Skeleton) -> float:
    return (knee_angle(skel, side="left") + knee_angle(skel, side="right")) / 2


class BaseBadmintonStrokeRule(ExerciseRule):
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


@register_rule
class BadmintonClearRule(BaseBadmintonStrokeRule):
    name = "badminton_clear"
    display_name_vi = "Cầu lông - Phông cầu"


@register_rule
class BadmintonSmashRule(BaseBadmintonStrokeRule):
    name = "badminton_smash"
    display_name_vi = "Cầu lông - Đập cầu"
    CONTACT_HEIGHT_MIN = 0.10
    SHOULDER_CONTACT_MIN = 105.0
    ELBOW_EXTENSION_MIN = 145.0
    TORSO_LEAN_MAX = 70.0


@register_rule
class BadmintonDropShotRule(BaseBadmintonStrokeRule):
    name = "badminton_drop_shot"
    display_name_vi = "Cầu lông - Bỏ nhỏ"
    CONTACT_HEIGHT_MIN = 0.06
    SHOULDER_CONTACT_MIN = 90.0
    ELBOW_EXTENSION_MIN = 135.0


@register_rule
class BadmintonDriveRule(BaseBadmintonStrokeRule):
    name = "badminton_drive"
    display_name_vi = "Cầu lông - Tạt cầu"
    rep_threshold_low = 60.0
    CONTACT_HEIGHT_MIN = -0.03
    SHOULDER_CONTACT_MIN = 55.0
    ELBOW_EXTENSION_MIN = 125.0
    TORSO_LEAN_MAX = 55.0


@register_rule
class BadmintonLungeRule(BaseBadmintonStrokeRule):
    name = "badminton_lunge"
    display_name_vi = "Cầu lông - Bước lunge đỡ cầu"
    CONTACT_HEIGHT_MIN = -0.02
    SHOULDER_CONTACT_MIN = 60.0
    ELBOW_EXTENSION_MIN = 125.0
    LUNGE_VALGUS_MAX = 0.16


@register_rule
class BadmintonServeRule(BaseBadmintonStrokeRule):
    name = "badminton_serve"
    display_name_vi = "Cầu lông - Giao cầu"
    rep_threshold_low = 45.0
    rep_threshold_high = 85.0
    CONTACT_HEIGHT_MIN = -0.10
    SHOULDER_CONTACT_MIN = 65.0
    ELBOW_EXTENSION_MIN = 120.0
    TORSO_LEAN_MAX = 50.0


@register_rule
class BadmintonLowServeRule(BaseBadmintonStrokeRule):
    name = "badminton_low_serve"
    display_name_vi = "Cầu lông - Giao cầu thấp ngắn"
    rep_threshold_low = 55.0
    rep_threshold_high = 88.0
    CONTACT_HEIGHT_MIN = -0.12
    SHOULDER_CONTACT_MIN = 52.0
    ELBOW_EXTENSION_MIN = 112.0
    TORSO_LEAN_MAX = 45.0


@register_rule
class BadmintonHighServeRule(BaseBadmintonStrokeRule):
    name = "badminton_high_serve"
    display_name_vi = "Cầu lông - Giao cầu cao sâu"
    rep_threshold_low = 42.0
    rep_threshold_high = 90.0
    CONTACT_HEIGHT_MIN = -0.04
    SHOULDER_CONTACT_MIN = 78.0
    ELBOW_EXTENSION_MIN = 125.0
    TORSO_LEAN_MAX = 55.0


@register_rule
class BadmintonBackhandClearRule(BaseBadmintonStrokeRule):
    name = "badminton_backhand_clear"
    display_name_vi = "Cầu lông - Phông trái tay"
    rep_threshold_low = 40.0
    CONTACT_HEIGHT_MIN = 0.03
    SHOULDER_CONTACT_MIN = 82.0
    ELBOW_EXTENSION_MIN = 130.0
    TORSO_LEAN_MAX = 60.0


@register_rule
class BadmintonNetShotRule(BaseBadmintonStrokeRule):
    name = "badminton_net_shot"
    display_name_vi = "Cầu lông - Đánh lưới"
    rep_threshold_low = 55.0
    rep_threshold_high = 88.0
    CONTACT_HEIGHT_MIN = -0.04
    SHOULDER_CONTACT_MIN = 58.0
    ELBOW_EXTENSION_MIN = 118.0
    TORSO_LEAN_MAX = 58.0


@register_rule
class BadmintonDefensiveBlockRule(BaseBadmintonStrokeRule):
    name = "badminton_defensive_block"
    display_name_vi = "Cầu lông - Chặn cầu"
    rep_threshold_low = 50.0
    rep_threshold_high = 88.0
    CONTACT_HEIGHT_MIN = -0.06
    SHOULDER_CONTACT_MIN = 62.0
    ELBOW_EXTENSION_MIN = 115.0
    TORSO_LEAN_MAX = 55.0


@register_rule
class BadmintonPushShotRule(BaseBadmintonStrokeRule):
    name = "badminton_push_shot"
    display_name_vi = "Cầu lông - Đẩy cầu / Tạt cầu"
    rep_threshold_low = 55.0
    rep_threshold_high = 88.0
    CONTACT_HEIGHT_MIN = -0.04
    SHOULDER_CONTACT_MIN = 58.0
    ELBOW_EXTENSION_MIN = 120.0
    TORSO_LEAN_MAX = 55.0


@register_rule
class BadmintonLiftShotRule(BaseBadmintonStrokeRule):
    name = "badminton_lift_shot"
    display_name_vi = "Cầu lông - Búng cầu / Lob lưới"
    rep_threshold_low = 45.0
    CONTACT_HEIGHT_MIN = 0.02
    SHOULDER_CONTACT_MIN = 82.0
    ELBOW_EXTENSION_MIN = 128.0
    TORSO_LEAN_MAX = 62.0


@register_rule
class BadmintonNetKillRule(BaseBadmintonStrokeRule):
    name = "badminton_net_kill"
    display_name_vi = "Cầu lông - Bỏ nhỏ / Chụp lưới"
    rep_threshold_low = 55.0
    rep_threshold_high = 88.0
    CONTACT_HEIGHT_MIN = -0.03
    SHOULDER_CONTACT_MIN = 60.0
    ELBOW_EXTENSION_MIN = 118.0
    TORSO_LEAN_MAX = 55.0


@register_rule
class BadmintonJuggleRule(BaseBadmintonStrokeRule):
    name = "badminton_juggle"
    display_name_vi = "Cầu lông - Tâng cầu"
    rep_threshold_low = 58.0
    rep_threshold_high = 88.0
    CONTACT_HEIGHT_MIN = -0.08
    SHOULDER_CONTACT_MIN = 52.0
    ELBOW_EXTENSION_MIN = 110.0
    TORSO_LEAN_MAX = 50.0


@register_rule
class BadmintonJumpSmashRule(BaseBadmintonStrokeRule):
    name = "badminton_jump_smash"
    display_name_vi = "Cầu lông - Bước nhảy đánh cầu"
    CONTACT_HEIGHT_MIN = 0.12
    SHOULDER_CONTACT_MIN = 108.0
    ELBOW_EXTENSION_MIN = 145.0
    TORSO_LEAN_MAX = 75.0


@register_rule
class BadmintonMultiShuttleRule(BaseBadmintonStrokeRule):
    name = "badminton_multi_shuttle"
    display_name_vi = "Cầu lông - Tập đa cầu"
    rep_threshold_low = 50.0
    rep_threshold_high = 90.0
    CONTACT_HEIGHT_MIN = -0.02
    SHOULDER_CONTACT_MIN = 65.0
    ELBOW_EXTENSION_MIN = 122.0
    TORSO_LEAN_MAX = 65.0


@register_rule
class BadmintonWallRallyRule(BaseBadmintonStrokeRule):
    name = "badminton_wall_rally"
    display_name_vi = "Cầu lông - Đánh cầu vào tường"
    rep_threshold_low = 55.0
    rep_threshold_high = 88.0
    CONTACT_HEIGHT_MIN = -0.06
    SHOULDER_CONTACT_MIN = 55.0
    ELBOW_EXTENSION_MIN = 115.0
    TORSO_LEAN_MAX = 50.0


@register_rule
class BadmintonHeavyRacketRule(BaseBadmintonStrokeRule):
    name = "badminton_heavy_racket"
    display_name_vi = "Cầu lông - Tập với vợt nặng"
    rep_threshold_low = 48.0
    rep_threshold_high = 90.0
    CONTACT_HEIGHT_MIN = -0.02
    SHOULDER_CONTACT_MIN = 65.0
    ELBOW_EXTENSION_MIN = 120.0
    TORSO_LEAN_MAX = 55.0


@register_rule
class BadmintonSplitStepRule(ExerciseRule):
    name = "badminton_split_step"
    display_name_vi = "Cầu lông - Split-step"
    category = "racket_sport"
    equipment = ["racket", "shuttlecock"]
    primary_joints = ["hip", "knee", "ankle"]
    issue_codes = [
        "BADMINTON_SPLIT_STEP_NO_KNEE_BEND",
        "BADMINTON_SPLIT_STEP_STANCE_NARROW",
        "BADMINTON_TORSO_OVERLEAN",
    ]
    primary_angle = "average_knee_angle"
    rep_threshold_low = 145.0
    rep_threshold_high = 160.0

    KNEE_BEND_MIN = 30.0
    STANCE_WIDTH_MIN = 1.15
    TORSO_LEAN_MAX = 45.0

    def _primary_angle_series(self, frames: list[Frame]) -> list[float]:
        return safe_series(frames, _average_knee_angle)

    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation:
        rep_frames = frames_with_keypoints(frames, rep, FOOTWORK_REQUIRED)
        if not rep_frames:
            return inconclusive(rep)

        knee_flexion = finite([180.0 - _average_knee_angle(f.skeleton) for f in rep_frames])
        stance_widths = finite([_stance_width_ratio(f.skeleton) for f in rep_frames])
        torso_angles = finite([back_angle(f.skeleton) for f in rep_frames])
        if not knee_flexion or not stance_widths:
            return inconclusive(rep)

        max_knee_flexion = max(knee_flexion)
        peak_frame = frames[rep.peak_idx]
        peak_skel = peak_frame.skeleton
        if peak_skel is None:
            return inconclusive(rep)

        stance_width = _stance_width_ratio(peak_skel)
        torso_lean = max(torso_angles) if torso_angles else float("nan")

        issues: list[Issue] = []
        score = 100

        if max_knee_flexion < self.KNEE_BEND_MIN:
            issues.append(Issue(
                code="BADMINTON_SPLIT_STEP_NO_KNEE_BEND",
                severity="MEDIUM",
                message_vi=f"Split-step chưa hạ trọng tâm đủ (gối gập {max_knee_flexion:.0f} deg)",
                frame_indices=[rep.peak_idx],
                recommendation="Hạ nhẹ gối khi tiếp đất để sẵn sàng bật về hướng cầu.",
            ))
            score -= 20

        if stance_width < self.STANCE_WIDTH_MIN:
            issues.append(Issue(
                code="BADMINTON_SPLIT_STEP_STANCE_NARROW",
                severity="MEDIUM",
                message_vi=f"Khoảng cách hai chân còn hẹp (ratio {stance_width:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Tiếp đất rộng hơn vai một chút để đổi hướng nhanh và cân bằng hơn.",
            ))
            score -= 15

        if not math.isnan(torso_lean) and torso_lean > self.TORSO_LEAN_MAX:
            issues.append(Issue(
                code="BADMINTON_TORSO_OVERLEAN",
                severity="LOW",
                message_vi=f"Thân người đổ quá nhiều khi split-step ({torso_lean:.0f} deg)",
                frame_indices=[rep.peak_idx],
                recommendation="Giữ ngực mở và trọng tâm ở giữa hai chân khi tiếp đất.",
            ))
            score -= 5

        score = max(0, score)
        return RepEvaluation(
            rep_index=rep.rep_index,
            score=score,
            passed=score >= 70 and not any(i.severity == "HIGH" for i in issues),
            issues=issues,
            metrics={
                "max_knee_flexion": round(max_knee_flexion, 1),
                "stance_width_ratio": round(stance_width, 3),
                "torso_lean_max": round(torso_lean, 1) if not math.isnan(torso_lean) else None,
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )


class BaseBadmintonFootworkDrillRule(BadmintonSplitStepRule):
    equipment = ["bodyweight"]


@register_rule
class BadmintonFrontCornersFootworkRule(BaseBadmintonFootworkDrillRule):
    name = "badminton_front_corners_footwork"
    display_name_vi = "Cầu lông - Di chuyển 2 góc lưới"


@register_rule
class BadmintonRearCornersFootworkRule(BaseBadmintonFootworkDrillRule):
    name = "badminton_rear_corners_footwork"
    display_name_vi = "Cầu lông - Di chuyển 2 góc cuối sân"


@register_rule
class BadmintonMidCornersFootworkRule(BaseBadmintonFootworkDrillRule):
    name = "badminton_mid_corners_footwork"
    display_name_vi = "Cầu lông - Di chuyển 2 góc giữa sân"


@register_rule
class BadmintonForwardBackwardFootworkRule(BaseBadmintonFootworkDrillRule):
    name = "badminton_forward_backward_footwork"
    display_name_vi = "Cầu lông - Di chuyển tiến lùi thẳng đứng"


@register_rule
class BadmintonMultiPointFootworkRule(BaseBadmintonFootworkDrillRule):
    name = "badminton_multi_point_footwork"
    display_name_vi = "Cầu lông - Di chuyển đa điểm"


@register_rule
class BadmintonJumpRopeRule(BaseBadmintonFootworkDrillRule):
    name = "badminton_jump_rope"
    display_name_vi = "Cầu lông - Nhảy dây"
    equipment = ["jump_rope"]
    KNEE_BEND_MIN = 18.0
    STANCE_WIDTH_MIN = 0.85
    TORSO_LEAN_MAX = 30.0


@register_rule
class BadmintonIntervalRunRule(BaseBadmintonFootworkDrillRule):
    name = "badminton_interval_run"
    display_name_vi = "Cầu lông - Chạy biến tốc"
    KNEE_BEND_MIN = 24.0
    STANCE_WIDTH_MIN = 0.90
    TORSO_LEAN_MAX = 42.0

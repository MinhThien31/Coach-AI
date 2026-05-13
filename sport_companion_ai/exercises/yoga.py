"""Yoga pose hold evaluators.

The prototype treats yoga poses as static holds over the full clip. Each rule
uses lightweight geometry from MediaPipe skeletons and returns FE-ready metrics.
"""
from __future__ import annotations

import math

from sport_companion_ai.exercises._common import frames_with_keypoints, inconclusive
from sport_companion_ai.exercises.base import ExerciseRule, register_rule
from sport_companion_ai.geometry import back_angle, elbow_angle, knee_angle, knee_valgus_ratio
from sport_companion_ai.pose.schema import Frame, Skeleton
from sport_companion_ai.report import Issue, Rep, RepEvaluation


FULL_BODY_REQUIRED = (
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)

LOWER_BODY_REQUIRED = (
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)


def _avg_y(skel: Skeleton, left: str, right: str) -> float:
    return (skel.keypoints[left].y + skel.keypoints[right].y) / 2


def _avg_x(skel: Skeleton, left: str, right: str) -> float:
    return (skel.keypoints[left].x + skel.keypoints[right].x) / 2


def _hold_ms(rep: Rep, fps: int) -> int:
    return int((rep.end_idx - rep.start_idx) / fps * 1000)


def _avg_knee_angle(skel: Skeleton) -> float:
    return (knee_angle(skel, side="left") + knee_angle(skel, side="right")) / 2


def _arm_height_offset(skel: Skeleton) -> float:
    shoulder_y = _avg_y(skel, "left_shoulder", "right_shoulder")
    left_offset = abs(skel.keypoints["left_wrist"].y - shoulder_y)
    right_offset = abs(skel.keypoints["right_wrist"].y - shoulder_y)
    return (left_offset + right_offset) / 2


def _downward_dog_hip_lift(skel: Skeleton) -> float:
    shoulder_y = _avg_y(skel, "left_shoulder", "right_shoulder")
    hip_y = _avg_y(skel, "left_hip", "right_hip")
    ankle_y = _avg_y(skel, "left_ankle", "right_ankle")
    expected_line_y = (shoulder_y + ankle_y) / 2
    return expected_line_y - hip_y


def _tree_foot_height_margin(skel: Skeleton) -> float:
    support_knee_y = skel.keypoints["left_knee"].y
    lifted_ankle_y = skel.keypoints["right_ankle"].y
    return support_knee_y - lifted_ankle_y


def _tree_knee_open_ratio(skel: Skeleton) -> float:
    hip_width = abs(skel.keypoints["right_hip"].x - skel.keypoints["left_hip"].x) or 1.0
    return abs(skel.keypoints["right_knee"].x - skel.keypoints["left_knee"].x) / hip_width


def _cobra_chest_lift(skel: Skeleton) -> float:
    hip_y = _avg_y(skel, "left_hip", "right_hip")
    shoulder_y = _avg_y(skel, "left_shoulder", "right_shoulder")
    return hip_y - shoulder_y


class BaseYogaHoldRule(ExerciseRule):
    category = "yoga"
    equipment = ["bodyweight", "yoga_mat"]
    movement_type = "hold"
    primary_joints = ["shoulder", "hip", "knee", "ankle", "core"]
    primary_angle = "pose_hold"
    rep_threshold_low = 0.0
    rep_threshold_high = 0.0
    MIN_HOLD_MS = 8000

    def _primary_angle_series(self, frames: list[Frame]) -> list[float]:
        return []

    def detect_reps(self, frames: list[Frame], fps: int = 30) -> list[Rep]:
        if not frames:
            return []
        return [Rep(rep_index=0, start_idx=0, peak_idx=len(frames) // 2, end_idx=len(frames) - 1)]

    def _short_hold_issue(self, rep: Rep, hold_ms: int) -> Issue | None:
        if hold_ms >= self.MIN_HOLD_MS:
            return None
        return Issue(
            code="YOGA_SHORT_HOLD",
            severity="LOW",
            message_vi=f"Thời gian giữ tư thế còn ngắn ({hold_ms} ms)",
            frame_indices=[rep.start_idx, rep.end_idx],
            recommendation="Giữ tư thế tối thiểu 8 giây với hơi thở ổn định trước khi đổi bên.",
        )


@register_rule
class YogaWarriorIIRule(BaseYogaHoldRule):
    name = "yoga_warrior_ii"
    display_name_vi = "Yoga - Chiến binh II"
    issue_codes = [
        "YOGA_WARRIOR_FRONT_KNEE_SHALLOW",
        "YOGA_WARRIOR_ARM_ALIGNMENT",
        "YOGA_TORSO_OVERLEAN",
        "YOGA_SHORT_HOLD",
    ]

    FRONT_KNEE_MAX = 120.0
    ARM_OFFSET_MAX = 0.07
    TORSO_LEAN_MAX = 25.0

    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation:
        rep_frames = frames_with_keypoints(frames, rep, FULL_BODY_REQUIRED)
        if not rep_frames:
            return inconclusive(rep)

        front_knee = min(knee_angle(f.skeleton, side="left") for f in rep_frames)
        arm_offset = max(_arm_height_offset(f.skeleton) for f in rep_frames)
        torso_lean = max(back_angle(f.skeleton) for f in rep_frames)
        hold_ms = _hold_ms(rep, self.fps)

        issues: list[Issue] = []
        score = 100
        if front_knee > self.FRONT_KNEE_MAX:
            issues.append(Issue(
                code="YOGA_WARRIOR_FRONT_KNEE_SHALLOW",
                severity="MEDIUM",
                message_vi=f"Gối trước chưa khuỵu đủ trong Warrior II ({front_knee:.0f} deg)",
                frame_indices=[rep.peak_idx],
                recommendation="Khuỵu gối trước gần 90 độ và giữ gối cùng hướng với mũi chân.",
            ))
            score -= 20
        if arm_offset > self.ARM_OFFSET_MAX:
            issues.append(Issue(
                code="YOGA_WARRIOR_ARM_ALIGNMENT",
                severity="LOW",
                message_vi=f"Hai tay chưa ngang vai (offset {arm_offset:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Duỗi hai tay ngang vai, kéo dài từ đầu ngón tay trái sang phải.",
            ))
            score -= 10
        if torso_lean > self.TORSO_LEAN_MAX:
            issues.append(Issue(
                code="YOGA_TORSO_OVERLEAN",
                severity="LOW",
                message_vi=f"Thân người nghiêng quá nhiều ({torso_lean:.0f} deg)",
                frame_indices=[rep.peak_idx],
                recommendation="Xếp vai trên hông và giữ thân người thẳng ở giữa hai chân.",
            ))
            score -= 10
        short_hold = self._short_hold_issue(rep, hold_ms)
        if short_hold:
            issues.append(short_hold)
            score -= 5

        return RepEvaluation(
            rep_index=rep.rep_index,
            score=max(0, score),
            passed=score >= 70,
            issues=issues,
            metrics={
                "front_knee_angle_min": round(front_knee, 1),
                "arm_height_offset_max": round(arm_offset, 3),
                "torso_lean_max": round(torso_lean, 1),
                "hold_duration_ms": float(hold_ms),
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )


@register_rule
class YogaTreePoseRule(BaseYogaHoldRule):
    name = "yoga_tree_pose"
    display_name_vi = "Yoga - Tư thế cây"
    issue_codes = [
        "YOGA_TREE_FOOT_TOO_LOW",
        "YOGA_TREE_HIP_NOT_OPEN",
        "YOGA_TORSO_OVERLEAN",
        "YOGA_SHORT_HOLD",
    ]

    FOOT_HEIGHT_MIN = -0.02
    KNEE_OPEN_MIN = 0.85
    TORSO_LEAN_MAX = 14.0

    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation:
        rep_frames = frames_with_keypoints(frames, rep, LOWER_BODY_REQUIRED)
        if not rep_frames:
            return inconclusive(rep)

        foot_margin = min(_tree_foot_height_margin(f.skeleton) for f in rep_frames)
        knee_open = min(_tree_knee_open_ratio(f.skeleton) for f in rep_frames)
        torso_lean = max(back_angle(f.skeleton) for f in rep_frames)
        hold_ms = _hold_ms(rep, self.fps)

        issues: list[Issue] = []
        score = 100
        if foot_margin < self.FOOT_HEIGHT_MIN:
            issues.append(Issue(
                code="YOGA_TREE_FOOT_TOO_LOW",
                severity="MEDIUM",
                message_vi=f"Bàn chân đặt còn thấp so với gối trụ (margin {foot_margin:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Đặt bàn chân lên bắp chân hoặc đùi trong, tránh tì trực tiếp vào khớp gối.",
            ))
            score -= 20
        if knee_open < self.KNEE_OPEN_MIN:
            issues.append(Issue(
                code="YOGA_TREE_HIP_NOT_OPEN",
                severity="LOW",
                message_vi=f"Hông bên chân nâng chưa mở đủ (ratio {knee_open:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Xoay gối nâng sang bên trong biên độ thoải mái và giữ hông cân.",
            ))
            score -= 10
        if torso_lean > self.TORSO_LEAN_MAX:
            issues.append(Issue(
                code="YOGA_TORSO_OVERLEAN",
                severity="MEDIUM",
                message_vi=f"Thân người lệch khỏi trục đứng ({torso_lean:.0f} deg)",
                frame_indices=[rep.peak_idx],
                recommendation="Ấn chân trụ xuống sàn, kéo đỉnh đầu lên và nhìn một điểm cố định.",
            ))
            score -= 20
        short_hold = self._short_hold_issue(rep, hold_ms)
        if short_hold:
            issues.append(short_hold)
            score -= 5

        return RepEvaluation(
            rep_index=rep.rep_index,
            score=max(0, score),
            passed=score >= 70,
            issues=issues,
            metrics={
                "foot_height_margin_min": round(foot_margin, 3),
                "knee_open_ratio_min": round(knee_open, 3),
                "torso_lean_max": round(torso_lean, 1),
                "hold_duration_ms": float(hold_ms),
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )


@register_rule
class YogaDownwardDogRule(BaseYogaHoldRule):
    name = "yoga_downward_dog"
    display_name_vi = "Yoga - Chó úp mặt"
    issue_codes = [
        "YOGA_DOWNDOG_HIPS_LOW",
        "YOGA_DOWNDOG_KNEES_BENT",
        "YOGA_SHORT_HOLD",
    ]

    HIP_LIFT_MIN = 0.06
    KNEE_ANGLE_MIN = 155.0

    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation:
        rep_frames = frames_with_keypoints(frames, rep, LOWER_BODY_REQUIRED)
        if not rep_frames:
            return inconclusive(rep)

        hip_lift = min(_downward_dog_hip_lift(f.skeleton) for f in rep_frames)
        knee_avg = min(_avg_knee_angle(f.skeleton) for f in rep_frames)
        hold_ms = _hold_ms(rep, self.fps)

        issues: list[Issue] = []
        score = 100
        if hip_lift < self.HIP_LIFT_MIN:
            issues.append(Issue(
                code="YOGA_DOWNDOG_HIPS_LOW",
                severity="MEDIUM",
                message_vi=f"Hông chưa đẩy đủ cao trong chó úp mặt (lift {hip_lift:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Đẩy hông lên và ra sau, kéo dài lưng trước khi cố duỗi gối hoàn toàn.",
            ))
            score -= 20
        if knee_avg < self.KNEE_ANGLE_MIN:
            issues.append(Issue(
                code="YOGA_DOWNDOG_KNEES_BENT",
                severity="LOW",
                message_vi=f"Gối còn gập nhiều ({knee_avg:.0f} deg)",
                frame_indices=[rep.peak_idx],
                recommendation="Duỗi gối dần trong biên độ thoải mái, ưu tiên lưng dài và vai ổn định.",
            ))
            score -= 10
        short_hold = self._short_hold_issue(rep, hold_ms)
        if short_hold:
            issues.append(short_hold)
            score -= 5

        return RepEvaluation(
            rep_index=rep.rep_index,
            score=max(0, score),
            passed=score >= 70,
            issues=issues,
            metrics={
                "hip_lift_min": round(hip_lift, 3),
                "average_knee_angle_min": round(knee_avg, 1),
                "hold_duration_ms": float(hold_ms),
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )


@register_rule
class YogaChairPoseRule(BaseYogaHoldRule):
    name = "yoga_chair_pose"
    display_name_vi = "Yoga - Tư thế ghế"
    issue_codes = [
        "YOGA_CHAIR_DEPTH_INSUFFICIENT",
        "YOGA_CHAIR_KNEE_VALGUS",
        "YOGA_TORSO_OVERLEAN",
        "YOGA_SHORT_HOLD",
    ]

    KNEE_DEPTH_MAX = 145.0
    VALGUS_MAX = 0.16
    TORSO_LEAN_MAX = 52.0

    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation:
        rep_frames = frames_with_keypoints(frames, rep, LOWER_BODY_REQUIRED)
        if not rep_frames:
            return inconclusive(rep)

        knee_avg = min(_avg_knee_angle(f.skeleton) for f in rep_frames)
        valgus = max(knee_valgus_ratio(f.skeleton) for f in rep_frames)
        torso_lean = max(back_angle(f.skeleton) for f in rep_frames)
        hold_ms = _hold_ms(rep, self.fps)

        issues: list[Issue] = []
        score = 100
        if knee_avg > self.KNEE_DEPTH_MAX:
            issues.append(Issue(
                code="YOGA_CHAIR_DEPTH_INSUFFICIENT",
                severity="MEDIUM",
                message_vi=f"Tư thế ghế còn quá cao (gối {knee_avg:.0f} deg)",
                frame_indices=[rep.peak_idx],
                recommendation="Hạ hông ra sau như ngồi ghế, giữ trọng lượng đều trên hai bàn chân.",
            ))
            score -= 20
        if valgus > self.VALGUS_MAX:
            issues.append(Issue(
                code="YOGA_CHAIR_KNEE_VALGUS",
                severity="HIGH",
                message_vi=f"Gối có xu hướng đổ vào trong (ratio {valgus:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Ấn gối theo hướng ngón chân thứ hai và giữ bàn chân bám sàn.",
            ))
            score -= 25
        if torso_lean > self.TORSO_LEAN_MAX:
            issues.append(Issue(
                code="YOGA_TORSO_OVERLEAN",
                severity="LOW",
                message_vi=f"Thân người đổ quá nhiều ({torso_lean:.0f} deg)",
                frame_indices=[rep.peak_idx],
                recommendation="Kéo xương sườn về trung tâm, nâng ngực vừa đủ và giữ cổ dài.",
            ))
            score -= 10
        short_hold = self._short_hold_issue(rep, hold_ms)
        if short_hold:
            issues.append(short_hold)
            score -= 5

        return RepEvaluation(
            rep_index=rep.rep_index,
            score=max(0, score),
            passed=score >= 70 and not any(i.severity == "HIGH" for i in issues),
            issues=issues,
            metrics={
                "average_knee_angle_min": round(knee_avg, 1),
                "knee_valgus_ratio_max": round(valgus, 3),
                "torso_lean_max": round(torso_lean, 1),
                "hold_duration_ms": float(hold_ms),
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )


@register_rule
class YogaCobraPoseRule(BaseYogaHoldRule):
    name = "yoga_cobra_pose"
    display_name_vi = "Yoga - Rắn hổ mang"
    issue_codes = [
        "YOGA_COBRA_CHEST_LOW",
        "YOGA_COBRA_ELBOW_COLLAPSE",
        "YOGA_COBRA_HIP_LIFT",
        "YOGA_SHORT_HOLD",
    ]

    CHEST_LIFT_MIN = 0.12
    ELBOW_EXTENSION_MIN = 115.0
    HIP_LIFT_MAX = 0.06

    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation:
        rep_frames = frames_with_keypoints(frames, rep, FULL_BODY_REQUIRED)
        if not rep_frames:
            return inconclusive(rep)

        chest_lift = min(_cobra_chest_lift(f.skeleton) for f in rep_frames)
        elbow_min = min(
            min(elbow_angle(f.skeleton, side="left"), elbow_angle(f.skeleton, side="right"))
            for f in rep_frames
        )
        hip_y = min(_avg_y(f.skeleton, "left_hip", "right_hip") for f in rep_frames)
        knee_y = min(_avg_y(f.skeleton, "left_knee", "right_knee") for f in rep_frames)
        hip_lift = knee_y - hip_y
        hold_ms = _hold_ms(rep, self.fps)

        issues: list[Issue] = []
        score = 100
        if chest_lift < self.CHEST_LIFT_MIN:
            issues.append(Issue(
                code="YOGA_COBRA_CHEST_LOW",
                severity="MEDIUM",
                message_vi=f"Ngực nâng chưa đủ trong cobra (lift {chest_lift:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Ấn nhẹ tay xuống sàn, kéo ngực về phía trước và giữ vai rời tai.",
            ))
            score -= 20
        if elbow_min < self.ELBOW_EXTENSION_MIN:
            issues.append(Issue(
                code="YOGA_COBRA_ELBOW_COLLAPSE",
                severity="LOW",
                message_vi=f"Khuỷu tay gập quá nhiều ({elbow_min:.0f} deg)",
                frame_indices=[rep.peak_idx],
                recommendation="Giữ khuỷu tay ôm thân và duỗi vừa đủ, không nhún vai.",
            ))
            score -= 10
        if hip_lift > self.HIP_LIFT_MAX:
            issues.append(Issue(
                code="YOGA_COBRA_HIP_LIFT",
                severity="MEDIUM",
                message_vi=f"Hông nhấc khỏi sàn quá nhiều (lift {hip_lift:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Giữ xương chậu nặng xuống thảm và kéo dài thắt lưng.",
            ))
            score -= 20
        short_hold = self._short_hold_issue(rep, hold_ms)
        if short_hold:
            issues.append(short_hold)
            score -= 5

        return RepEvaluation(
            rep_index=rep.rep_index,
            score=max(0, score),
            passed=score >= 70,
            issues=issues,
            metrics={
                "chest_lift_min": round(chest_lift, 3),
                "elbow_angle_min": round(elbow_min, 1),
                "hip_lift": round(hip_lift, 3),
                "hold_duration_ms": float(hold_ms),
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )


@register_rule
class YogaTrianglePoseRule(BaseYogaHoldRule):
    name = "yoga_triangle_pose"
    display_name_vi = "Yoga - Tư thế tam giác"
    issue_codes = [
        "YOGA_TRIANGLE_TORSO_TOO_UPRIGHT",
        "YOGA_TRIANGLE_FRONT_KNEE_BENT",
        "YOGA_TRIANGLE_ARM_STACK",
        "YOGA_SHORT_HOLD",
    ]

    TORSO_LEAN_MIN = 45.0
    FRONT_KNEE_MIN = 155.0
    WRIST_STACK_MAX = 0.12

    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation:
        rep_frames = frames_with_keypoints(frames, rep, FULL_BODY_REQUIRED)
        if not rep_frames:
            return inconclusive(rep)

        torso_lean = min(back_angle(f.skeleton) for f in rep_frames)
        front_knee = min(knee_angle(f.skeleton, side="left") for f in rep_frames)
        wrist_stack = max(
            abs(f.skeleton.keypoints["left_wrist"].x - f.skeleton.keypoints["right_wrist"].x)
            for f in rep_frames
        )
        hold_ms = _hold_ms(rep, self.fps)

        issues: list[Issue] = []
        score = 100
        if torso_lean < self.TORSO_LEAN_MIN:
            issues.append(Issue(
                code="YOGA_TRIANGLE_TORSO_TOO_UPRIGHT",
                severity="MEDIUM",
                message_vi=f"Thân người chưa nghiêng đủ trong tam giác ({torso_lean:.0f} deg)",
                frame_indices=[rep.peak_idx],
                recommendation="Gập từ khớp hông sang bên, kéo dài hai sườn trước khi đặt tay xuống.",
            ))
            score -= 20
        if front_knee < self.FRONT_KNEE_MIN:
            issues.append(Issue(
                code="YOGA_TRIANGLE_FRONT_KNEE_BENT",
                severity="LOW",
                message_vi=f"Chân trước còn gập nhiều ({front_knee:.0f} deg)",
                frame_indices=[rep.peak_idx],
                recommendation="Duỗi chân trước trong biên độ thoải mái và giữ gối không khóa cứng.",
            ))
            score -= 10
        if wrist_stack > self.WRIST_STACK_MAX:
            issues.append(Issue(
                code="YOGA_TRIANGLE_ARM_STACK",
                severity="LOW",
                message_vi=f"Hai tay chưa xếp trên một trục (delta {wrist_stack:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Xếp cổ tay trên và dưới gần cùng một đường thẳng, mở ngực sang bên.",
            ))
            score -= 10
        short_hold = self._short_hold_issue(rep, hold_ms)
        if short_hold:
            issues.append(short_hold)
            score -= 5

        return RepEvaluation(
            rep_index=rep.rep_index,
            score=max(0, score),
            passed=score >= 70,
            issues=issues,
            metrics={
                "torso_lean_min": round(torso_lean, 1),
                "front_knee_angle_min": round(front_knee, 1),
                "wrist_stack_delta_max": round(wrist_stack, 3),
                "hold_duration_ms": float(hold_ms),
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )


@register_rule
class YogaBridgePoseRule(BaseYogaHoldRule):
    name = "yoga_bridge_pose"
    display_name_vi = "Yoga - Tư thế cây cầu"
    issue_codes = [
        "YOGA_BRIDGE_HIPS_LOW",
        "YOGA_BRIDGE_KNEES_SPLAY",
        "YOGA_SHORT_HOLD",
    ]

    HIP_LIFT_MIN = 0.08
    KNEE_WIDTH_RATIO_MAX = 1.6

    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation:
        rep_frames = frames_with_keypoints(frames, rep, LOWER_BODY_REQUIRED)
        if not rep_frames:
            return inconclusive(rep)

        hip_lift = min(
            _avg_y(f.skeleton, "left_knee", "right_knee") - _avg_y(f.skeleton, "left_hip", "right_hip")
            for f in rep_frames
        )
        knee_width_ratio = max(
            abs(f.skeleton.keypoints["right_knee"].x - f.skeleton.keypoints["left_knee"].x)
            / (abs(f.skeleton.keypoints["right_hip"].x - f.skeleton.keypoints["left_hip"].x) or 1.0)
            for f in rep_frames
        )
        hold_ms = _hold_ms(rep, self.fps)

        issues: list[Issue] = []
        score = 100
        if hip_lift < self.HIP_LIFT_MIN:
            issues.append(Issue(
                code="YOGA_BRIDGE_HIPS_LOW",
                severity="MEDIUM",
                message_vi=f"Hông nâng chưa đủ cao trong cây cầu (lift {hip_lift:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Ấn gót xuống thảm, siết mông nhẹ và nâng hông lên cùng nhịp thở.",
            ))
            score -= 20
        if knee_width_ratio > self.KNEE_WIDTH_RATIO_MAX:
            issues.append(Issue(
                code="YOGA_BRIDGE_KNEES_SPLAY",
                severity="LOW",
                message_vi=f"Hai gối mở quá rộng (ratio {knee_width_ratio:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Giữ hai gối song song theo hướng bàn chân, tránh để gối đổ ra ngoài.",
            ))
            score -= 10
        short_hold = self._short_hold_issue(rep, hold_ms)
        if short_hold:
            issues.append(short_hold)
            score -= 5

        return RepEvaluation(
            rep_index=rep.rep_index,
            score=max(0, score),
            passed=score >= 70,
            issues=issues,
            metrics={
                "hip_lift_min": round(hip_lift, 3),
                "knee_width_ratio_max": round(knee_width_ratio, 3),
                "hold_duration_ms": float(hold_ms),
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )


@register_rule
class YogaChildPoseRule(BaseYogaHoldRule):
    name = "yoga_child_pose"
    display_name_vi = "Yoga - Tư thế em bé"
    issue_codes = [
        "YOGA_CHILD_CHEST_TOO_HIGH",
        "YOGA_CHILD_HIPS_NOT_BACK",
        "YOGA_SHORT_HOLD",
    ]

    CHEST_FOLD_MIN = 0.06
    HIP_TO_ANKLE_MAX = 0.24

    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation:
        rep_frames = frames_with_keypoints(frames, rep, LOWER_BODY_REQUIRED)
        if not rep_frames:
            return inconclusive(rep)

        chest_fold = min(
            _avg_y(f.skeleton, "left_shoulder", "right_shoulder")
            - _avg_y(f.skeleton, "left_hip", "right_hip")
            for f in rep_frames
        )
        hip_to_ankle = max(
            _avg_y(f.skeleton, "left_ankle", "right_ankle")
            - _avg_y(f.skeleton, "left_hip", "right_hip")
            for f in rep_frames
        )
        hold_ms = _hold_ms(rep, self.fps)

        issues: list[Issue] = []
        score = 100
        if chest_fold < self.CHEST_FOLD_MIN:
            issues.append(Issue(
                code="YOGA_CHILD_CHEST_TOO_HIGH",
                severity="LOW",
                message_vi=f"Ngực còn nâng cao, chưa thả xuống thảm (fold {chest_fold:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Thả ngực về giữa hai đùi, để vai mềm và thở chậm.",
            ))
            score -= 10
        if hip_to_ankle > self.HIP_TO_ANKLE_MAX:
            issues.append(Issue(
                code="YOGA_CHILD_HIPS_NOT_BACK",
                severity="LOW",
                message_vi=f"Hông chưa lùi đủ về gót chân (delta {hip_to_ankle:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Đưa hông lùi về gót chân trong biên độ thoải mái, có thể kê gối nếu cần.",
            ))
            score -= 10
        short_hold = self._short_hold_issue(rep, hold_ms)
        if short_hold:
            issues.append(short_hold)
            score -= 5

        return RepEvaluation(
            rep_index=rep.rep_index,
            score=max(0, score),
            passed=score >= 70,
            issues=issues,
            metrics={
                "chest_fold_min": round(chest_fold, 3),
                "hip_to_ankle_delta_max": round(hip_to_ankle, 3),
                "hold_duration_ms": float(hold_ms),
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )


@register_rule
class YogaBoatPoseRule(BaseYogaHoldRule):
    name = "yoga_boat_pose"
    display_name_vi = "Yoga - Tư thế con thuyền"
    issue_codes = [
        "YOGA_BOAT_FEET_LOW",
        "YOGA_BOAT_TORSO_TOO_UPRIGHT",
        "YOGA_BOAT_KNEES_TOO_BENT",
        "YOGA_SHORT_HOLD",
    ]

    FEET_LIFT_MIN = 0.08
    TORSO_LEAN_MIN = 28.0
    KNEE_ANGLE_MIN = 150.0

    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation:
        rep_frames = frames_with_keypoints(frames, rep, LOWER_BODY_REQUIRED)
        if not rep_frames:
            return inconclusive(rep)

        feet_lift = min(
            _avg_y(f.skeleton, "left_knee", "right_knee")
            - _avg_y(f.skeleton, "left_ankle", "right_ankle")
            for f in rep_frames
        )
        torso_lean = min(back_angle(f.skeleton) for f in rep_frames)
        knee_avg = min(_avg_knee_angle(f.skeleton) for f in rep_frames)
        hold_ms = _hold_ms(rep, self.fps)

        issues: list[Issue] = []
        score = 100
        if feet_lift < self.FEET_LIFT_MIN:
            issues.append(Issue(
                code="YOGA_BOAT_FEET_LOW",
                severity="MEDIUM",
                message_vi=f"Bàn chân nâng chưa đủ cao trong con thuyền (lift {feet_lift:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Nâng cẳng chân lên, giữ ngực mở và trọng tâm sau xương ngồi.",
            ))
            score -= 20
        if torso_lean < self.TORSO_LEAN_MIN:
            issues.append(Issue(
                code="YOGA_BOAT_TORSO_TOO_UPRIGHT",
                severity="LOW",
                message_vi=f"Thân người còn quá thẳng đứng ({torso_lean:.0f} deg)",
                frame_indices=[rep.peak_idx],
                recommendation="Ngả thân người nhẹ ra sau nhưng vẫn giữ lưng dài và ngực mở.",
            ))
            score -= 10
        if knee_avg < self.KNEE_ANGLE_MIN:
            issues.append(Issue(
                code="YOGA_BOAT_KNEES_TOO_BENT",
                severity="LOW",
                message_vi=f"Gối gập quá nhiều ({knee_avg:.0f} deg)",
                frame_indices=[rep.peak_idx],
                recommendation="Duỗi cẳng chân dần theo sức core, hoặc giữ gối gập vừa phải nhưng bàn chân cao.",
            ))
            score -= 10
        short_hold = self._short_hold_issue(rep, hold_ms)
        if short_hold:
            issues.append(short_hold)
            score -= 5

        return RepEvaluation(
            rep_index=rep.rep_index,
            score=max(0, score),
            passed=score >= 70,
            issues=issues,
            metrics={
                "feet_lift_min": round(feet_lift, 3),
                "torso_lean_min": round(torso_lean, 1),
                "average_knee_angle_min": round(knee_avg, 1),
                "hold_duration_ms": float(hold_ms),
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )


@register_rule
class YogaLowLungeRule(BaseYogaHoldRule):
    name = "yoga_low_lunge"
    display_name_vi = "Yoga - Low lunge"
    issue_codes = [
        "YOGA_LUNGE_FRONT_KNEE_SHALLOW",
        "YOGA_LUNGE_KNEE_VALGUS",
        "YOGA_TORSO_OVERLEAN",
        "YOGA_SHORT_HOLD",
    ]

    FRONT_KNEE_MAX = 125.0
    VALGUS_MAX = 0.18
    TORSO_LEAN_MAX = 40.0

    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation:
        rep_frames = frames_with_keypoints(frames, rep, LOWER_BODY_REQUIRED)
        if not rep_frames:
            return inconclusive(rep)

        front_knee = min(knee_angle(f.skeleton, side="left") for f in rep_frames)
        valgus = max(knee_valgus_ratio(f.skeleton) for f in rep_frames)
        torso_lean = max(back_angle(f.skeleton) for f in rep_frames)
        hold_ms = _hold_ms(rep, self.fps)

        issues: list[Issue] = []
        score = 100
        if front_knee > self.FRONT_KNEE_MAX:
            issues.append(Issue(
                code="YOGA_LUNGE_FRONT_KNEE_SHALLOW",
                severity="MEDIUM",
                message_vi=f"Gối trước chưa khuỵu đủ trong low lunge ({front_knee:.0f} deg)",
                frame_indices=[rep.peak_idx],
                recommendation="Đưa gối trước gần 90 độ và giữ hông hạ đều về phía thảm.",
            ))
            score -= 20
        if valgus > self.VALGUS_MAX:
            issues.append(Issue(
                code="YOGA_LUNGE_KNEE_VALGUS",
                severity="HIGH",
                message_vi=f"Gối trước lệch vào trong (ratio {valgus:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Canh gối theo hướng mũi chân trước và trải đều trọng lượng qua bàn chân.",
            ))
            score -= 25
        if torso_lean > self.TORSO_LEAN_MAX:
            issues.append(Issue(
                code="YOGA_TORSO_OVERLEAN",
                severity="LOW",
                message_vi=f"Thân người đổ quá nhiều ({torso_lean:.0f} deg)",
                frame_indices=[rep.peak_idx],
                recommendation="Kéo đỉnh đầu lên, thả hông xuống và giữ ngực mở.",
            ))
            score -= 10
        short_hold = self._short_hold_issue(rep, hold_ms)
        if short_hold:
            issues.append(short_hold)
            score -= 5

        return RepEvaluation(
            rep_index=rep.rep_index,
            score=max(0, score),
            passed=score >= 70 and not any(i.severity == "HIGH" for i in issues),
            issues=issues,
            metrics={
                "front_knee_angle_min": round(front_knee, 1),
                "knee_valgus_ratio_max": round(valgus, 3),
                "torso_lean_max": round(torso_lean, 1),
                "hold_duration_ms": float(hold_ms),
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )

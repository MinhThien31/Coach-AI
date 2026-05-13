import math

import pytest

from sport_companion_ai.exercises.base import ExerciseRule
from sport_companion_ai.exercises.badminton import (
    BadmintonBackhandClearRule,
    BadmintonClearRule,
    BadmintonDefensiveBlockRule,
    BadmintonDriveRule,
    BadmintonDropShotRule,
    BadmintonForwardBackwardFootworkRule,
    BadmintonFrontCornersFootworkRule,
    BadmintonHeavyRacketRule,
    BadmintonHighServeRule,
    BadmintonIntervalRunRule,
    BadmintonJuggleRule,
    BadmintonJumpRopeRule,
    BadmintonJumpSmashRule,
    BadmintonLiftShotRule,
    BadmintonLowServeRule,
    BadmintonLungeRule,
    BadmintonMidCornersFootworkRule,
    BadmintonMultiPointFootworkRule,
    BadmintonMultiShuttleRule,
    BadmintonNetKillRule,
    BadmintonNetShotRule,
    BadmintonPushShotRule,
    BadmintonRearCornersFootworkRule,
    BadmintonServeRule,
    BadmintonSmashRule,
    BadmintonSplitStepRule,
    BadmintonWallRallyRule,
)
from sport_companion_ai.exercises.lateral_raise import LateralRaiseRule
from sport_companion_ai.exercises.lunge import LungeRule
from sport_companion_ai.exercises.overhead_press import OverheadPressRule
from sport_companion_ai.exercises.plank import PlankRule
from sport_companion_ai.exercises.pullup import PullUpRule
from sport_companion_ai.exercises.yoga import (
    YogaBoatPoseRule,
    YogaBridgePoseRule,
    YogaChairPoseRule,
    YogaChildPoseRule,
    YogaCobraPoseRule,
    YogaDownwardDogRule,
    YogaLowLungeRule,
    YogaTreePoseRule,
    YogaTrianglePoseRule,
    YogaWarriorIIRule,
)
from sport_companion_ai.pose.schema import Frame, Skeleton
from sport_companion_ai.report import Rep
from tests.exercises._helpers import kp, squat_skeleton


def _elbow_chain(shoulder, angle_deg: float, side: str) -> dict:
    sign = -1 if side == "left" else 1
    elbow_len = 0.12
    forearm_len = 0.12
    elbow = (shoulder[0] + sign * 0.02, shoulder[1] + elbow_len)
    theta = math.radians(angle_deg)
    wrist = (
        elbow[0] + sign * forearm_len * math.sin(theta),
        elbow[1] - forearm_len * math.cos(theta),
    )
    return {
        f"{side}_shoulder": kp(*shoulder),
        f"{side}_elbow": kp(*elbow),
        f"{side}_wrist": kp(*wrist),
    }


def _press_skeleton(left_elbow: float, right_elbow: float | None = None,
                    lean_deg: float = 6.0) -> Skeleton:
    right_elbow = left_elbow if right_elbow is None else right_elbow
    torso_len = 0.25
    lean = math.radians(lean_deg)
    hip_mid = (0.5, 0.72)
    shoulder_mid = (
        hip_mid[0] + torso_len * math.sin(lean),
        hip_mid[1] - torso_len * math.cos(lean),
    )
    points = {
        "left_hip": kp(0.44, 0.72),
        "right_hip": kp(0.56, 0.72),
    }
    points.update(_elbow_chain((shoulder_mid[0] - 0.06, shoulder_mid[1]), left_elbow, "left"))
    points.update(_elbow_chain((shoulder_mid[0] + 0.06, shoulder_mid[1]), right_elbow, "right"))
    return Skeleton(keypoints=points)


def make_press_frames(lockout_angle: float = 170.0, lean_deg: float = 6.0,
                      right_lockout: float | None = None) -> list[Frame]:
    frames = []
    n = 60
    for i in range(n):
        progress = abs(2 * (i / n) - 1)
        left = lockout_angle - (lockout_angle - 80) * progress
        right_peak = lockout_angle if right_lockout is None else right_lockout
        right = right_peak - (right_peak - 80) * progress
        frames.append(Frame(
            index=i,
            timestamp_ms=i * 33,
            skeleton=_press_skeleton(left, right_elbow=right, lean_deg=lean_deg),
        ))
    return frames


def _pullup_skeleton(elbow_deg: float, top_clearance: float = 0.1,
                     swing: float = 0.0) -> Skeleton:
    left = _elbow_chain((0.44, 0.40), elbow_deg, "left")
    right = _elbow_chain((0.56, 0.40), elbow_deg, "right")
    wrist_y = (left["left_wrist"].y + right["right_wrist"].y) / 2
    return Skeleton(keypoints={
        **left,
        **right,
        "nose": kp(0.5, wrist_y - top_clearance),
        "left_hip": kp(0.44 + swing, 0.66),
        "right_hip": kp(0.56 + swing, 0.66),
    })


def make_pullup_frames(top_elbow: float = 65.0, dead_hang: float = 170.0,
                       top_clearance: float = 0.1, swing: float = 0.0) -> list[Frame]:
    frames = []
    n = 60
    for i in range(n):
        progress = abs(2 * (i / n) - 1)
        elbow = top_elbow + (dead_hang - top_elbow) * progress
        clearance = top_clearance * (1 - progress)
        frames.append(Frame(
            index=i,
            timestamp_ms=i * 33,
            skeleton=_pullup_skeleton(elbow, top_clearance=clearance, swing=swing * (1 - progress)),
        ))
    return frames


def make_lunge_frames(min_knee: float = 90.0, knee_offset: float = 0.0,
                      lean_deg: float = 35.0) -> list[Frame]:
    frames = []
    n = 60
    for i in range(n):
        progress = abs(2 * (i / n) - 1)
        knee = 170 - (170 - min_knee) * (1 - progress)
        offset = knee_offset * (1 - progress)
        frames.append(Frame(
            index=i,
            timestamp_ms=i * 33,
            skeleton=squat_skeleton(knee, back_deg=lean_deg, knee_offset=offset),
        ))
    return frames


def _plank_skeleton(offset: float = 0.0) -> Skeleton:
    return Skeleton(keypoints={
        "left_shoulder": kp(0.35, 0.35),
        "right_shoulder": kp(0.45, 0.35),
        "left_hip": kp(0.45, 0.55 + offset),
        "right_hip": kp(0.55, 0.55 + offset),
        "left_ankle": kp(0.65, 0.75),
        "right_ankle": kp(0.75, 0.75),
    })


def make_plank_frames(offset: float = 0.0, n: int = 360) -> list[Frame]:
    return [
        Frame(index=i, timestamp_ms=i * 33, skeleton=_plank_skeleton(offset))
        for i in range(n)
    ]


def _raise_skeleton(shoulder_deg: float, shrug: bool = False) -> Skeleton:
    def arm(side: str, shoulder):
        sign = -1 if side == "left" else 1
        upper = 0.16
        lower = 0.10
        theta = math.radians(shoulder_deg)
        elbow = (
            shoulder[0] + sign * upper * math.sin(theta),
            shoulder[1] + upper * math.cos(theta),
        )
        wrist = (elbow[0] + sign * lower * math.sin(theta), elbow[1] + lower * math.cos(theta))
        return {
            f"{side}_shoulder": kp(*shoulder),
            f"{side}_elbow": kp(*elbow),
            f"{side}_wrist": kp(*wrist),
        }

    shoulder_y = 0.40 if shrug else 0.48
    points = {
        "left_hip": kp(0.43, 0.75),
        "right_hip": kp(0.57, 0.75),
        "left_ear": kp(0.46, 0.32),
        "right_ear": kp(0.54, 0.32),
    }
    points.update(arm("left", (0.43, shoulder_y)))
    points.update(arm("right", (0.57, shoulder_y)))
    return Skeleton(keypoints=points)


def make_raise_frames(max_shoulder: float = 90.0, shrug: bool = False) -> list[Frame]:
    frames = []
    n = 60
    for i in range(n):
        progress = abs(2 * (i / n) - 1)
        shoulder = max_shoulder * (1 - progress)
        frames.append(Frame(
            index=i,
            timestamp_ms=i * 33,
            skeleton=_raise_skeleton(shoulder, shrug=shrug and progress < 0.25),
        ))
    return frames


def _badminton_skeleton(
    shoulder_deg: float = 110.0,
    elbow_deg: float = 165.0,
    contact_high: float = 0.14,
    knee_offset: float = 0.0,
    lean_deg: float = 35.0,
) -> Skeleton:
    torso_len = 0.25
    lean = math.radians(lean_deg)
    hip_mid = (0.5, 0.70)
    shoulder_mid = (
        hip_mid[0] + torso_len * math.sin(lean),
        hip_mid[1] - torso_len * math.cos(lean),
    )

    right_shoulder = (shoulder_mid[0] + 0.06, shoulder_mid[1])
    left_shoulder = (shoulder_mid[0] - 0.06, shoulder_mid[1])
    upper = 0.16
    forearm = 0.12
    theta = math.radians(shoulder_deg)
    right_elbow = (
        right_shoulder[0] + upper * math.sin(theta),
        right_shoulder[1] + upper * math.cos(theta),
    )
    phi = math.radians(elbow_deg)
    right_wrist = (
        right_elbow[0] + forearm * math.sin(phi),
        shoulder_mid[1] - contact_high,
    )

    left_hip = (0.42, 0.70)
    right_hip = (0.58, 0.70)
    left_knee = (0.36 + knee_offset, 0.78)
    left_ankle = (0.30, 0.90)

    return Skeleton(keypoints={
        "left_shoulder": kp(*left_shoulder),
        "right_shoulder": kp(*right_shoulder),
        "right_elbow": kp(*right_elbow),
        "right_wrist": kp(*right_wrist),
        "left_elbow": kp(left_shoulder[0] - 0.02, left_shoulder[1] + 0.12),
        "left_wrist": kp(left_shoulder[0] - 0.04, left_shoulder[1] + 0.22),
        "left_hip": kp(*left_hip),
        "right_hip": kp(*right_hip),
        "left_knee": kp(*left_knee),
        "left_ankle": kp(*left_ankle),
        "right_knee": kp(0.62, 0.82),
        "right_ankle": kp(0.67, 0.92),
    })


def make_badminton_frames(
    max_shoulder: float = 110.0,
    max_elbow: float = 145.0,
    contact_high: float = 0.14,
    knee_offset: float = 0.0,
    lean_deg: float = 35.0,
) -> list[Frame]:
    frames = []
    n = 60
    for i in range(n):
        progress = abs(2 * (i / n) - 1)
        shoulder = max_shoulder - (max_shoulder - 35) * progress
        elbow = max_elbow - (max_elbow - 90) * progress
        high = contact_high * (1 - progress)
        frames.append(Frame(
            index=i,
            timestamp_ms=i * 33,
            skeleton=_badminton_skeleton(
                shoulder_deg=shoulder,
                elbow_deg=elbow,
                contact_high=high,
                knee_offset=knee_offset * (1 - progress),
                lean_deg=lean_deg,
            ),
        ))
    return frames


def _badminton_split_step_skeleton(
    knee_deg: float = 135.0,
    stance_ratio: float = 1.5,
    lean_deg: float = 18.0,
) -> Skeleton:
    skel = squat_skeleton(knee_deg, back_deg=lean_deg)
    left_shoulder = skel.keypoints["left_shoulder"]
    right_shoulder = skel.keypoints["right_shoulder"]
    left_ankle = skel.keypoints["left_ankle"]
    right_ankle = skel.keypoints["right_ankle"]
    ankle_width = abs(right_ankle.x - left_ankle.x)
    shoulder_width = ankle_width / stance_ratio
    shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
    shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
    points = dict(skel.keypoints)
    points["left_shoulder"] = kp(shoulder_mid_x - shoulder_width / 2, shoulder_mid_y)
    points["right_shoulder"] = kp(shoulder_mid_x + shoulder_width / 2, shoulder_mid_y)
    return Skeleton(keypoints=points)


def make_badminton_split_step_frames(
    max_knee_flexion: float = 45.0,
    stance_ratio: float = 1.5,
    lean_deg: float = 18.0,
) -> list[Frame]:
    frames = []
    n = 60
    for i in range(n):
        progress = abs(2 * (i / n) - 1)
        knee = 170 - max_knee_flexion * (1 - progress)
        stance = 1.0 + (stance_ratio - 1.0) * (1 - progress)
        frames.append(Frame(
            index=i,
            timestamp_ms=i * 33,
            skeleton=_badminton_split_step_skeleton(
                knee_deg=knee,
                stance_ratio=stance,
                lean_deg=lean_deg,
            ),
        ))
    return frames


def _with_warrior_arms(skel: Skeleton, arm_offset: float = 0.0) -> Skeleton:
    points = dict(skel.keypoints)
    left_shoulder = points["left_shoulder"]
    right_shoulder = points["right_shoulder"]
    points.update({
        "left_elbow": kp(left_shoulder.x - 0.12, left_shoulder.y + arm_offset),
        "left_wrist": kp(left_shoulder.x - 0.24, left_shoulder.y + arm_offset),
        "right_elbow": kp(right_shoulder.x + 0.12, right_shoulder.y + arm_offset),
        "right_wrist": kp(right_shoulder.x + 0.24, right_shoulder.y + arm_offset),
    })
    return Skeleton(keypoints=points)


def make_yoga_warrior_frames(
    knee_deg: float = 95.0,
    arm_offset: float = 0.0,
    lean_deg: float = 8.0,
    n: int = 300,
) -> list[Frame]:
    skel = _with_warrior_arms(squat_skeleton(knee_deg, back_deg=lean_deg), arm_offset=arm_offset)
    return [Frame(index=i, timestamp_ms=i * 33, skeleton=skel) for i in range(n)]


def _tree_skeleton(
    foot_margin: float = 0.08,
    knee_open_ratio: float = 1.1,
    lean_deg: float = 5.0,
) -> Skeleton:
    hip_width = 0.12
    hip_mid = (0.5, 0.62)
    torso_len = 0.26
    lean = math.radians(lean_deg)
    shoulder_mid = (
        hip_mid[0] + torso_len * math.sin(lean),
        hip_mid[1] - torso_len * math.cos(lean),
    )
    left_knee = (0.47, 0.76)
    right_knee = (left_knee[0] + hip_width * knee_open_ratio, 0.68)
    right_ankle = (0.49, left_knee[1] - foot_margin)
    return Skeleton(keypoints={
        "left_shoulder": kp(shoulder_mid[0] - 0.07, shoulder_mid[1]),
        "right_shoulder": kp(shoulder_mid[0] + 0.07, shoulder_mid[1]),
        "left_hip": kp(hip_mid[0] - hip_width / 2, hip_mid[1]),
        "right_hip": kp(hip_mid[0] + hip_width / 2, hip_mid[1]),
        "left_knee": kp(*left_knee),
        "right_knee": kp(*right_knee),
        "left_ankle": kp(0.47, 0.94),
        "right_ankle": kp(*right_ankle),
    })


def make_yoga_tree_frames(
    foot_margin: float = 0.08,
    knee_open_ratio: float = 1.1,
    lean_deg: float = 5.0,
    n: int = 300,
) -> list[Frame]:
    skel = _tree_skeleton(
        foot_margin=foot_margin,
        knee_open_ratio=knee_open_ratio,
        lean_deg=lean_deg,
    )
    return [Frame(index=i, timestamp_ms=i * 33, skeleton=skel) for i in range(n)]


def _downward_dog_skeleton(hip_y: float = 0.34, knee_y: float = 0.64) -> Skeleton:
    return Skeleton(keypoints={
        "left_shoulder": kp(0.32, 0.64),
        "right_shoulder": kp(0.42, 0.64),
        "left_hip": kp(0.48, hip_y),
        "right_hip": kp(0.58, hip_y),
        "left_knee": kp(0.64, knee_y),
        "right_knee": kp(0.74, knee_y),
        "left_ankle": kp(0.78, 0.88),
        "right_ankle": kp(0.88, 0.88),
    })


def make_yoga_downward_dog_frames(
    hip_y: float = 0.34,
    knee_y: float = 0.64,
    n: int = 300,
) -> list[Frame]:
    skel = _downward_dog_skeleton(hip_y=hip_y, knee_y=knee_y)
    return [Frame(index=i, timestamp_ms=i * 33, skeleton=skel) for i in range(n)]


def make_yoga_chair_frames(
    knee_deg: float = 120.0,
    knee_offset: float = 0.0,
    lean_deg: float = 35.0,
    n: int = 300,
) -> list[Frame]:
    skel = squat_skeleton(knee_deg, back_deg=lean_deg, knee_offset=knee_offset)
    return [Frame(index=i, timestamp_ms=i * 33, skeleton=skel) for i in range(n)]


def _cobra_skeleton(
    chest_lift: float = 0.17,
    elbow_low: bool = False,
    hip_lift: float = 0.03,
) -> Skeleton:
    hip_y = 0.74 - hip_lift
    shoulder_y = hip_y - chest_lift
    left_shoulder = (0.42, shoulder_y)
    right_shoulder = (0.58, shoulder_y)
    if elbow_low:
        left_elbow, right_elbow = (0.40, shoulder_y + 0.08), (0.60, shoulder_y + 0.08)
        left_wrist, right_wrist = (0.44, shoulder_y + 0.09), (0.56, shoulder_y + 0.09)
    else:
        left_elbow, right_elbow = (0.39, shoulder_y + 0.10), (0.61, shoulder_y + 0.10)
        left_wrist, right_wrist = (0.37, shoulder_y + 0.22), (0.63, shoulder_y + 0.22)
    return Skeleton(keypoints={
        "left_shoulder": kp(*left_shoulder),
        "right_shoulder": kp(*right_shoulder),
        "left_elbow": kp(*left_elbow),
        "right_elbow": kp(*right_elbow),
        "left_wrist": kp(*left_wrist),
        "right_wrist": kp(*right_wrist),
        "left_hip": kp(0.44, hip_y),
        "right_hip": kp(0.56, hip_y),
        "left_knee": kp(0.44, 0.77),
        "right_knee": kp(0.56, 0.77),
        "left_ankle": kp(0.42, 0.91),
        "right_ankle": kp(0.58, 0.91),
    })


def make_yoga_cobra_frames(
    chest_lift: float = 0.17,
    elbow_low: bool = False,
    hip_lift: float = 0.03,
    n: int = 300,
) -> list[Frame]:
    skel = _cobra_skeleton(chest_lift=chest_lift, elbow_low=elbow_low, hip_lift=hip_lift)
    return [Frame(index=i, timestamp_ms=i * 33, skeleton=skel) for i in range(n)]


def _triangle_skeleton(
    torso_lean: float = 62.0,
    knee_deg: float = 170.0,
    wrist_stack_delta: float = 0.04,
) -> Skeleton:
    skel = squat_skeleton(knee_deg, back_deg=torso_lean)
    points = dict(skel.keypoints)
    left_shoulder = points["left_shoulder"]
    right_shoulder = points["right_shoulder"]
    wrist_x = (left_shoulder.x + right_shoulder.x) / 2
    points.update({
        "left_elbow": kp(wrist_x, left_shoulder.y + 0.12),
        "left_wrist": kp(wrist_x, left_shoulder.y + 0.24),
        "right_elbow": kp(wrist_x + wrist_stack_delta / 2, right_shoulder.y - 0.12),
        "right_wrist": kp(wrist_x + wrist_stack_delta, right_shoulder.y - 0.24),
    })
    return Skeleton(keypoints=points)


def make_yoga_triangle_frames(
    torso_lean: float = 62.0,
    knee_deg: float = 170.0,
    wrist_stack_delta: float = 0.04,
    n: int = 300,
) -> list[Frame]:
    skel = _triangle_skeleton(
        torso_lean=torso_lean,
        knee_deg=knee_deg,
        wrist_stack_delta=wrist_stack_delta,
    )
    return [Frame(index=i, timestamp_ms=i * 33, skeleton=skel) for i in range(n)]


def _bridge_skeleton(
    hip_y: float = 0.62,
    knee_width: float = 0.18,
) -> Skeleton:
    hip_width = 0.16
    return Skeleton(keypoints={
        "left_shoulder": kp(0.42, 0.76),
        "right_shoulder": kp(0.58, 0.76),
        "left_hip": kp(0.5 - hip_width / 2, hip_y),
        "right_hip": kp(0.5 + hip_width / 2, hip_y),
        "left_knee": kp(0.5 - knee_width / 2, 0.75),
        "right_knee": kp(0.5 + knee_width / 2, 0.75),
        "left_ankle": kp(0.40, 0.90),
        "right_ankle": kp(0.60, 0.90),
    })


def make_yoga_bridge_frames(
    hip_y: float = 0.62,
    knee_width: float = 0.18,
    n: int = 300,
) -> list[Frame]:
    skel = _bridge_skeleton(hip_y=hip_y, knee_width=knee_width)
    return [Frame(index=i, timestamp_ms=i * 33, skeleton=skel) for i in range(n)]


def _child_pose_skeleton(
    shoulder_y: float = 0.72,
    hip_y: float = 0.60,
    ankle_y: float = 0.78,
) -> Skeleton:
    return Skeleton(keypoints={
        "left_shoulder": kp(0.42, shoulder_y),
        "right_shoulder": kp(0.58, shoulder_y),
        "left_hip": kp(0.44, hip_y),
        "right_hip": kp(0.56, hip_y),
        "left_knee": kp(0.38, 0.70),
        "right_knee": kp(0.62, 0.70),
        "left_ankle": kp(0.38, ankle_y),
        "right_ankle": kp(0.62, ankle_y),
    })


def make_yoga_child_pose_frames(
    shoulder_y: float = 0.72,
    hip_y: float = 0.60,
    ankle_y: float = 0.78,
    n: int = 300,
) -> list[Frame]:
    skel = _child_pose_skeleton(shoulder_y=shoulder_y, hip_y=hip_y, ankle_y=ankle_y)
    return [Frame(index=i, timestamp_ms=i * 33, skeleton=skel) for i in range(n)]


def _boat_skeleton(
    torso_lean: float = 38.0,
    ankle_y: float = 0.54,
) -> Skeleton:
    hip_mid = (0.5, 0.74)
    torso_len = 0.24
    lean = math.radians(torso_lean)
    shoulder_mid = (
        hip_mid[0] + torso_len * math.sin(lean),
        hip_mid[1] - torso_len * math.cos(lean),
    )
    return Skeleton(keypoints={
        "left_shoulder": kp(shoulder_mid[0] - 0.07, shoulder_mid[1]),
        "right_shoulder": kp(shoulder_mid[0] + 0.07, shoulder_mid[1]),
        "left_hip": kp(0.44, hip_mid[1]),
        "right_hip": kp(0.56, hip_mid[1]),
        "left_knee": kp(0.40, 0.67),
        "right_knee": kp(0.60, 0.67),
        "left_ankle": kp(0.36, ankle_y),
        "right_ankle": kp(0.64, ankle_y),
    })


def make_yoga_boat_frames(
    torso_lean: float = 38.0,
    ankle_y: float = 0.54,
    n: int = 300,
) -> list[Frame]:
    skel = _boat_skeleton(torso_lean=torso_lean, ankle_y=ankle_y)
    return [Frame(index=i, timestamp_ms=i * 33, skeleton=skel) for i in range(n)]


def make_yoga_low_lunge_frames(
    knee_deg: float = 95.0,
    knee_offset: float = 0.0,
    lean_deg: float = 20.0,
    n: int = 300,
) -> list[Frame]:
    skel = squat_skeleton(knee_deg, back_deg=lean_deg, knee_offset=knee_offset)
    return [Frame(index=i, timestamp_ms=i * 33, skeleton=skel) for i in range(n)]


def _rep_for(frames: list[Frame]) -> Rep:
    return Rep(rep_index=0, start_idx=0, peak_idx=len(frames) // 2, end_idx=len(frames) - 1)


@pytest.mark.parametrize(("name", "rule"), [
    ("badminton_backhand_clear", BadmintonBackhandClearRule),
    ("badminton_clear", BadmintonClearRule),
    ("badminton_defensive_block", BadmintonDefensiveBlockRule),
    ("badminton_forward_backward_footwork", BadmintonForwardBackwardFootworkRule),
    ("badminton_front_corners_footwork", BadmintonFrontCornersFootworkRule),
    ("badminton_heavy_racket", BadmintonHeavyRacketRule),
    ("badminton_high_serve", BadmintonHighServeRule),
    ("badminton_interval_run", BadmintonIntervalRunRule),
    ("badminton_juggle", BadmintonJuggleRule),
    ("badminton_jump_rope", BadmintonJumpRopeRule),
    ("badminton_jump_smash", BadmintonJumpSmashRule),
    ("badminton_lift_shot", BadmintonLiftShotRule),
    ("badminton_low_serve", BadmintonLowServeRule),
    ("badminton_smash", BadmintonSmashRule),
    ("badminton_drop_shot", BadmintonDropShotRule),
    ("badminton_drive", BadmintonDriveRule),
    ("badminton_lunge", BadmintonLungeRule),
    ("badminton_mid_corners_footwork", BadmintonMidCornersFootworkRule),
    ("badminton_multi_point_footwork", BadmintonMultiPointFootworkRule),
    ("badminton_multi_shuttle", BadmintonMultiShuttleRule),
    ("badminton_net_kill", BadmintonNetKillRule),
    ("badminton_net_shot", BadmintonNetShotRule),
    ("badminton_push_shot", BadmintonPushShotRule),
    ("badminton_rear_corners_footwork", BadmintonRearCornersFootworkRule),
    ("badminton_serve", BadmintonServeRule),
    ("badminton_split_step", BadmintonSplitStepRule),
    ("badminton_wall_rally", BadmintonWallRallyRule),
    ("overhead_press", OverheadPressRule),
    ("pull_up", PullUpRule),
    ("lunge", LungeRule),
    ("plank", PlankRule),
    ("lateral_raise", LateralRaiseRule),
    ("yoga_boat_pose", YogaBoatPoseRule),
    ("yoga_bridge_pose", YogaBridgePoseRule),
    ("yoga_chair_pose", YogaChairPoseRule),
    ("yoga_child_pose", YogaChildPoseRule),
    ("yoga_cobra_pose", YogaCobraPoseRule),
    ("yoga_downward_dog", YogaDownwardDogRule),
    ("yoga_low_lunge", YogaLowLungeRule),
    ("yoga_tree_pose", YogaTreePoseRule),
    ("yoga_triangle_pose", YogaTrianglePoseRule),
    ("yoga_warrior_ii", YogaWarriorIIRule),
])
def test_new_rules_registered(name, rule):
    assert ExerciseRule.get(name) is rule


def test_overhead_press_clean_and_issues():
    rule = OverheadPressRule()
    clean = make_press_frames()
    rep = rule.detect_reps(clean)[0]
    assert rule.evaluate_rep(rep, clean).passed is True

    cases = [
        (make_press_frames(lockout_angle=145), "OHP_PARTIAL_LOCKOUT"),
        (make_press_frames(lean_deg=25), "OHP_BACK_LEAN"),
        (make_press_frames(lockout_angle=170, right_lockout=130), "OHP_ASYMMETRY"),
    ]
    for frames, code in cases:
        eval_ = rule.evaluate_rep(_rep_for(frames), frames)
        assert code in {issue.code for issue in eval_.issues}


def test_pullup_clean_and_issues():
    rule = PullUpRule()
    clean = make_pullup_frames()
    rep = rule.detect_reps(clean)[0]
    assert rule.evaluate_rep(rep, clean).passed is True

    cases = [
        (make_pullup_frames(dead_hang=145), "PULLUP_PARTIAL_ROM"),
        (make_pullup_frames(top_elbow=105, top_clearance=-0.02), "PULLUP_INCOMPLETE_TOP"),
        (make_pullup_frames(swing=0.16), "PULLUP_BODY_SWING"),
    ]
    for frames, code in cases:
        eval_ = rule.evaluate_rep(_rep_for(frames), frames)
        assert code in {issue.code for issue in eval_.issues}


def test_lunge_clean_and_issues():
    rule = LungeRule()
    clean = make_lunge_frames()
    rep = rule.detect_reps(clean)[0]
    assert rule.evaluate_rep(rep, clean).passed is True

    cases = [
        (make_lunge_frames(min_knee=125), "LUNGE_DEPTH_INSUFFICIENT"),
        (make_lunge_frames(knee_offset=0.10), "LUNGE_KNEE_VALGUS"),
        (make_lunge_frames(lean_deg=65), "LUNGE_TORSO_LEAN"),
    ]
    for frames, code in cases:
        eval_ = rule.evaluate_rep(_rep_for(frames), frames)
        assert code in {issue.code for issue in eval_.issues}


def test_plank_clean_and_issues():
    rule = PlankRule()
    clean = make_plank_frames()
    rep = rule.detect_reps(clean)[0]
    assert rule.evaluate_rep(rep, clean).passed is True

    cases = [
        (make_plank_frames(offset=0.06), "PLANK_HIP_SAG"),
        (make_plank_frames(offset=-0.06), "PLANK_HIP_PIKE"),
        (make_plank_frames(n=120), "PLANK_SHORT_HOLD"),
    ]
    for frames, code in cases:
        eval_ = rule.evaluate_rep(_rep_for(frames), frames)
        assert code in {issue.code for issue in eval_.issues}


def test_lateral_raise_clean_and_issues():
    rule = LateralRaiseRule()
    clean = make_raise_frames()
    rep = rule.detect_reps(clean)[0]
    assert rule.evaluate_rep(rep, clean).passed is True

    cases = [
        (make_raise_frames(max_shoulder=55), "LATERAL_RAISE_PARTIAL_ROM"),
        (make_raise_frames(shrug=True), "LATERAL_RAISE_SHRUG"),
        (make_raise_frames(max_shoulder=125), "LATERAL_RAISE_TOO_HIGH"),
    ]
    for frames, code in cases:
        eval_ = rule.evaluate_rep(_rep_for(frames), frames)
        assert code in {issue.code for issue in eval_.issues}


@pytest.mark.parametrize(("rule", "frames"), [
    (BadmintonBackhandClearRule(), make_badminton_frames(max_shoulder=90, max_elbow=140, contact_high=0.08)),
    (BadmintonClearRule(), make_badminton_frames()),
    (BadmintonDefensiveBlockRule(), make_badminton_frames(max_shoulder=72, max_elbow=130, contact_high=0.02)),
    (BadmintonHeavyRacketRule(), make_badminton_frames(max_shoulder=75, max_elbow=130, contact_high=0.02)),
    (BadmintonHighServeRule(), make_badminton_frames(max_shoulder=88, max_elbow=132, contact_high=0.04)),
    (BadmintonJuggleRule(), make_badminton_frames(max_shoulder=65, max_elbow=122, contact_high=0.0)),
    (BadmintonJumpSmashRule(), make_badminton_frames(max_shoulder=118, max_elbow=150, contact_high=0.18)),
    (BadmintonLiftShotRule(), make_badminton_frames(max_shoulder=92, max_elbow=135, contact_high=0.08)),
    (BadmintonLowServeRule(), make_badminton_frames(max_shoulder=62, max_elbow=122, contact_high=0.0)),
    (BadmintonSmashRule(), make_badminton_frames(max_shoulder=115, contact_high=0.16)),
    (BadmintonDropShotRule(), make_badminton_frames(max_shoulder=100, max_elbow=145, contact_high=0.10)),
    (BadmintonDriveRule(), make_badminton_frames(max_shoulder=80, max_elbow=140, contact_high=0.02)),
    (BadmintonLungeRule(), make_badminton_frames(max_shoulder=75, max_elbow=140, contact_high=0.02)),
    (BadmintonMultiShuttleRule(), make_badminton_frames(max_shoulder=78, max_elbow=132, contact_high=0.04)),
    (BadmintonNetKillRule(), make_badminton_frames(max_shoulder=72, max_elbow=130, contact_high=0.02)),
    (BadmintonNetShotRule(), make_badminton_frames(max_shoulder=72, max_elbow=130, contact_high=0.02)),
    (BadmintonPushShotRule(), make_badminton_frames(max_shoulder=72, max_elbow=132, contact_high=0.02)),
    (BadmintonServeRule(), make_badminton_frames(max_shoulder=75, max_elbow=130, contact_high=0.0)),
    (BadmintonWallRallyRule(), make_badminton_frames(max_shoulder=68, max_elbow=125, contact_high=0.0)),
])
def test_badminton_category_exercises_clean(rule, frames):
    rep = rule.detect_reps(frames)[0]
    assert rule.evaluate_rep(rep, frames).passed is True


def test_badminton_split_step_clean_and_issues():
    rule = BadmintonSplitStepRule()
    clean = make_badminton_split_step_frames()
    rep = rule.detect_reps(clean)[0]
    assert rule.evaluate_rep(rep, clean).passed is True

    cases = [
        (
            make_badminton_split_step_frames(max_knee_flexion=15),
            "BADMINTON_SPLIT_STEP_NO_KNEE_BEND",
        ),
        (
            make_badminton_split_step_frames(stance_ratio=1.05),
            "BADMINTON_SPLIT_STEP_STANCE_NARROW",
        ),
        (
            make_badminton_split_step_frames(lean_deg=55),
            "BADMINTON_TORSO_OVERLEAN",
        ),
    ]
    for frames, code in cases:
        eval_ = rule.evaluate_rep(_rep_for(frames), frames)
        assert code in {issue.code for issue in eval_.issues}


@pytest.mark.parametrize("rule", [
    BadmintonForwardBackwardFootworkRule(),
    BadmintonFrontCornersFootworkRule(),
    BadmintonIntervalRunRule(),
    BadmintonJumpRopeRule(),
    BadmintonMidCornersFootworkRule(),
    BadmintonMultiPointFootworkRule(),
    BadmintonRearCornersFootworkRule(),
])
def test_badminton_footwork_drills_clean(rule):
    frames = make_badminton_split_step_frames()
    rep = rule.detect_reps(frames)[0]
    assert rule.evaluate_rep(rep, frames).passed is True


def test_badminton_smash_issues():
    rule = BadmintonSmashRule()
    clean = make_badminton_frames()
    rep = rule.detect_reps(clean)[0]
    assert rule.evaluate_rep(rep, clean).passed is True

    cases = [
        (make_badminton_frames(max_shoulder=80, contact_high=0.02), "BADMINTON_CONTACT_TOO_LOW"),
        (make_badminton_frames(max_elbow=205), "BADMINTON_ELBOW_COLLAPSE"),
        (make_badminton_frames(knee_offset=0.08), "BADMINTON_LUNGE_KNEE_VALGUS"),
        (make_badminton_frames(lean_deg=75), "BADMINTON_TORSO_OVERLEAN"),
    ]
    for frames, code in cases:
        eval_ = rule.evaluate_rep(_rep_for(frames), frames)
        assert code in {issue.code for issue in eval_.issues}


@pytest.mark.parametrize(("rule", "frames"), [
    (YogaWarriorIIRule(), make_yoga_warrior_frames()),
    (YogaTrianglePoseRule(), make_yoga_triangle_frames()),
    (YogaTreePoseRule(), make_yoga_tree_frames()),
    (YogaDownwardDogRule(), make_yoga_downward_dog_frames()),
    (YogaChairPoseRule(), make_yoga_chair_frames()),
    (YogaCobraPoseRule(), make_yoga_cobra_frames()),
    (YogaBridgePoseRule(), make_yoga_bridge_frames()),
    (YogaChildPoseRule(), make_yoga_child_pose_frames()),
    (YogaBoatPoseRule(), make_yoga_boat_frames()),
    (YogaLowLungeRule(), make_yoga_low_lunge_frames()),
])
def test_yoga_poses_clean(rule, frames):
    rep = rule.detect_reps(frames)[0]
    assert rule.evaluate_rep(rep, frames).passed is True


@pytest.mark.parametrize(("rule", "frames", "code"), [
    (YogaWarriorIIRule(), make_yoga_warrior_frames(knee_deg=145), "YOGA_WARRIOR_FRONT_KNEE_SHALLOW"),
    (YogaWarriorIIRule(), make_yoga_warrior_frames(arm_offset=0.12), "YOGA_WARRIOR_ARM_ALIGNMENT"),
    (YogaTrianglePoseRule(), make_yoga_triangle_frames(torso_lean=25), "YOGA_TRIANGLE_TORSO_TOO_UPRIGHT"),
    (YogaTrianglePoseRule(), make_yoga_triangle_frames(knee_deg=130), "YOGA_TRIANGLE_FRONT_KNEE_BENT"),
    (YogaTrianglePoseRule(), make_yoga_triangle_frames(wrist_stack_delta=0.22), "YOGA_TRIANGLE_ARM_STACK"),
    (YogaTreePoseRule(), make_yoga_tree_frames(foot_margin=-0.06), "YOGA_TREE_FOOT_TOO_LOW"),
    (YogaTreePoseRule(), make_yoga_tree_frames(knee_open_ratio=0.5), "YOGA_TREE_HIP_NOT_OPEN"),
    (YogaDownwardDogRule(), make_yoga_downward_dog_frames(hip_y=0.74), "YOGA_DOWNDOG_HIPS_LOW"),
    (YogaDownwardDogRule(), make_yoga_downward_dog_frames(knee_y=0.77), "YOGA_DOWNDOG_KNEES_BENT"),
    (YogaChairPoseRule(), make_yoga_chair_frames(knee_deg=160), "YOGA_CHAIR_DEPTH_INSUFFICIENT"),
    (YogaChairPoseRule(), make_yoga_chair_frames(knee_offset=0.10), "YOGA_CHAIR_KNEE_VALGUS"),
    (YogaCobraPoseRule(), make_yoga_cobra_frames(chest_lift=0.05), "YOGA_COBRA_CHEST_LOW"),
    (YogaCobraPoseRule(), make_yoga_cobra_frames(elbow_low=True), "YOGA_COBRA_ELBOW_COLLAPSE"),
    (YogaCobraPoseRule(), make_yoga_cobra_frames(hip_lift=0.12), "YOGA_COBRA_HIP_LIFT"),
    (YogaBridgePoseRule(), make_yoga_bridge_frames(hip_y=0.71), "YOGA_BRIDGE_HIPS_LOW"),
    (YogaBridgePoseRule(), make_yoga_bridge_frames(knee_width=0.34), "YOGA_BRIDGE_KNEES_SPLAY"),
    (YogaChildPoseRule(), make_yoga_child_pose_frames(shoulder_y=0.62), "YOGA_CHILD_CHEST_TOO_HIGH"),
    (YogaChildPoseRule(), make_yoga_child_pose_frames(hip_y=0.48, ankle_y=0.82), "YOGA_CHILD_HIPS_NOT_BACK"),
    (YogaBoatPoseRule(), make_yoga_boat_frames(ankle_y=0.66), "YOGA_BOAT_FEET_LOW"),
    (YogaBoatPoseRule(), make_yoga_boat_frames(torso_lean=12), "YOGA_BOAT_TORSO_TOO_UPRIGHT"),
    (YogaBoatPoseRule(), make_yoga_boat_frames(ankle_y=0.66), "YOGA_BOAT_KNEES_TOO_BENT"),
    (YogaLowLungeRule(), make_yoga_low_lunge_frames(knee_deg=150), "YOGA_LUNGE_FRONT_KNEE_SHALLOW"),
    (YogaLowLungeRule(), make_yoga_low_lunge_frames(knee_offset=0.11), "YOGA_LUNGE_KNEE_VALGUS"),
])
def test_yoga_pose_issues(rule, frames, code):
    eval_ = rule.evaluate_rep(_rep_for(frames), frames)
    assert code in {issue.code for issue in eval_.issues}


@pytest.mark.parametrize(("rule", "rep"), [
    (BadmintonBackhandClearRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonClearRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonDefensiveBlockRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonForwardBackwardFootworkRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonFrontCornersFootworkRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonHeavyRacketRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonHighServeRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonIntervalRunRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonJuggleRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonJumpRopeRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonJumpSmashRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonLiftShotRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonLowServeRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonSmashRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonDropShotRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonDriveRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonLungeRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonMidCornersFootworkRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonMultiPointFootworkRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonMultiShuttleRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonNetKillRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonNetShotRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonPushShotRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonRearCornersFootworkRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonServeRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonSplitStepRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (BadmintonWallRallyRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (OverheadPressRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (PullUpRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (LungeRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (PlankRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (LateralRaiseRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (YogaBoatPoseRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (YogaBridgePoseRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (YogaChairPoseRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (YogaChildPoseRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (YogaCobraPoseRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (YogaDownwardDogRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (YogaLowLungeRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (YogaTreePoseRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (YogaTrianglePoseRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (YogaWarriorIIRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
])
def test_new_rules_missing_keypoints_inconclusive(rule, rep):
    eval_ = rule.evaluate_rep(rep, [Frame(index=0, timestamp_ms=0, skeleton=None)])
    assert eval_.inconclusive is True
    assert eval_.score is None

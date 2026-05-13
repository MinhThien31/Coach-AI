"""Pure geometry helpers operating on Keypoints. All angles in degrees."""
import math

from sport_companion_ai.pose.schema import Keypoint, Skeleton


def angle_3pt(a: Keypoint, b: Keypoint, c: Keypoint) -> float:
    """Angle at vertex b formed by rays b→a and b→c. NaN if degenerate."""
    bax, bay = a.x - b.x, a.y - b.y
    bcx, bcy = c.x - b.x, c.y - b.y
    mag_ba = math.hypot(bax, bay)
    mag_bc = math.hypot(bcx, bcy)
    if mag_ba == 0 or mag_bc == 0:
        return float("nan")
    cos = (bax * bcx + bay * bcy) / (mag_ba * mag_bc)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def angle_with_vertical(p1: Keypoint, p2: Keypoint) -> float:
    """Angle of the line p1→p2 from the vertical axis (always non-negative)."""
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def knee_angle(skel: Skeleton, side: str = "left") -> float:
    return angle_3pt(
        skel.keypoints[f"{side}_hip"],
        skel.keypoints[f"{side}_knee"],
        skel.keypoints[f"{side}_ankle"],
    )


def elbow_angle(skel: Skeleton, side: str = "left") -> float:
    return angle_3pt(
        skel.keypoints[f"{side}_shoulder"],
        skel.keypoints[f"{side}_elbow"],
        skel.keypoints[f"{side}_wrist"],
    )


def hip_angle(skel: Skeleton, side: str = "left") -> float:
    return angle_3pt(
        skel.keypoints[f"{side}_shoulder"],
        skel.keypoints[f"{side}_hip"],
        skel.keypoints[f"{side}_knee"],
    )


def shoulder_angle(skel: Skeleton, side: str = "left") -> float:
    return angle_3pt(
        skel.keypoints[f"{side}_hip"],
        skel.keypoints[f"{side}_shoulder"],
        skel.keypoints[f"{side}_elbow"],
    )


def back_angle(skel: Skeleton) -> float:
    """Angle of the torso (mid-hip → mid-shoulder) from vertical."""
    lh = skel.keypoints["left_hip"]
    rh = skel.keypoints["right_hip"]
    ls = skel.keypoints["left_shoulder"]
    rs = skel.keypoints["right_shoulder"]
    mid_hip = Keypoint(x=(lh.x + rh.x) / 2, y=(lh.y + rh.y) / 2)
    mid_sh = Keypoint(x=(ls.x + rs.x) / 2, y=(ls.y + rs.y) / 2)
    return angle_with_vertical(mid_hip, mid_sh)


def torso_alignment_offset(skel: Skeleton) -> float:
    """Hip offset from the shoulder-ankle line. Positive means hip sag."""
    ls = skel.keypoints["left_shoulder"]
    rs = skel.keypoints["right_shoulder"]
    lh = skel.keypoints["left_hip"]
    rh = skel.keypoints["right_hip"]
    la = skel.keypoints["left_ankle"]
    ra = skel.keypoints["right_ankle"]
    sh_y = (ls.y + rs.y) / 2
    hip_y = (lh.y + rh.y) / 2
    ankle_y = (la.y + ra.y) / 2
    expected_hip_y = (sh_y + ankle_y) / 2
    return hip_y - expected_hip_y


def knee_valgus_ratio(skel: Skeleton) -> float:
    """Mean horizontal knee deviation from the hip-ankle midline,
    normalized by hip width. 0.0 = neutral, ~0.3+ = visible valgus."""
    lh, lk, la = skel.keypoints["left_hip"], skel.keypoints["left_knee"], skel.keypoints["left_ankle"]
    rh, rk, ra = skel.keypoints["right_hip"], skel.keypoints["right_knee"], skel.keypoints["right_ankle"]

    def deviation(hip: Keypoint, knee: Keypoint, ankle: Keypoint) -> float:
        expected_x = (hip.x + ankle.x) / 2
        return abs(knee.x - expected_x)

    hip_width = abs(rh.x - lh.x)
    if hip_width == 0:
        return 0.0
    return (deviation(lh, lk, la) + deviation(rh, rk, ra)) / 2 / hip_width

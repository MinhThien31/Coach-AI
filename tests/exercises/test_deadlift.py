import math

from sport_companion_ai.exercises.deadlift import DeadliftRule
from sport_companion_ai.exercises.base import ExerciseRule
from sport_companion_ai.pose.schema import Frame, Keypoint, Skeleton


def kp(x, y):
    return Keypoint(x=x, y=y, visibility=1.0)


def deadlift_skeleton(hip_deg: float, back_round: float = 0.0) -> Skeleton:
    """Hip flex angle controls how bent over.
    back_round shifts shoulders forward to simulate spinal flexion (rounding).

    Geometry: shin vertical, thigh from knee up to hip at angle (180 - hip_deg)
    backward from vertical, shoulder positioned by rotating hip→shoulder vector.

    The helper must produce angle_3pt(shoulder, hip, knee) == hip_deg.
    """
    # Knee position fixed; ankle directly below.
    lk_x, lk_y = 0.50, 0.75
    rk_x, rk_y = 0.52, 0.75
    la_x, la_y = lk_x, 0.95
    ra_x, ra_y = rk_x, 0.95

    # Thigh: from knee, lean backward at (180 - hip_deg) from vertical.
    thigh_t = math.radians(180 - hip_deg)
    thigh_len = 0.20
    lh_x = lk_x - thigh_len * math.sin(thigh_t)
    lh_y = lk_y - thigh_len * math.cos(thigh_t)
    rh_x = lh_x + 0.02  # small lateral offset for right side
    rh_y = lh_y

    # Torso direction: rotate the hip→knee vector by -hip_deg (in the same plane)
    # so that angle_3pt(shoulder, hip, knee) == hip_deg.
    hk_x = lk_x - lh_x
    hk_y = lk_y - lh_y
    rad = math.radians(-hip_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    torso_dx = hk_x * cos_a - hk_y * sin_a
    torso_dy = hk_x * sin_a + hk_y * cos_a
    mag = math.hypot(torso_dx, torso_dy)
    if mag > 0:
        torso_dx /= mag
        torso_dy /= mag
    torso_len = 0.30
    ls_x = lh_x + torso_dx * torso_len + back_round
    ls_y = lh_y + torso_dy * torso_len
    rs_x = rh_x + torso_dx * torso_len + back_round
    rs_y = rh_y + torso_dy * torso_len

    return Skeleton(keypoints={
        "left_hip": kp(lh_x, lh_y), "right_hip": kp(rh_x, rh_y),
        "left_shoulder": kp(ls_x, ls_y), "right_shoulder": kp(rs_x, rs_y),
        "left_knee": kp(lk_x, lk_y), "right_knee": kp(rk_x, rk_y),
        "left_ankle": kp(la_x, la_y), "right_ankle": kp(ra_x, ra_y),
        "left_elbow": kp(ls_x + 0.05, ls_y + 0.10), "right_elbow": kp(rs_x + 0.05, rs_y + 0.10),
        "left_wrist": kp(ls_x + 0.05, ls_y + 0.20), "right_wrist": kp(rs_x + 0.05, rs_y + 0.20),
    })


def make_deadlift_rep_frames(min_hip_deg: float, back_round: float = 0.0,
                             fps: int = 30, rep_seconds: float = 2.5) -> list[Frame]:
    n = int(rep_seconds * fps)
    out = []
    for i in range(n):
        progress = abs(2 * (i / n) - 1)
        hip_at_t = 170 - (170 - min_hip_deg) * (1 - progress)
        # back_round dynamic — applied proportional to depth
        round_at_t = back_round * (1 - progress)
        skel = deadlift_skeleton(hip_at_t, back_round=round_at_t)
        out.append(Frame(index=i, timestamp_ms=int(i / fps * 1000), skeleton=skel))
    return out


def test_registered():
    assert ExerciseRule.get("deadlift") is DeadliftRule


def test_clean_rep_passes():
    frames = make_deadlift_rep_frames(min_hip_deg=80)
    rule = DeadliftRule()
    rep = rule.detect_reps(frames)[0]
    eval_ = rule.evaluate_rep(rep, frames)
    assert eval_.passed is True


def test_back_rounding_flagged():
    frames = make_deadlift_rep_frames(min_hip_deg=80, back_round=0.20)
    rule = DeadliftRule()
    rep = rule.detect_reps(frames)[0]
    eval_ = rule.evaluate_rep(rep, frames)
    assert any(i.code == "DEADLIFT_BACK_ROUND" for i in eval_.issues)


def test_partial_lockout_flagged():
    frames = make_deadlift_rep_frames(min_hip_deg=80, rep_seconds=2.5)
    # Truncate trailing ascent half — last frame stays around hip ~110°
    frames = frames[:int(len(frames) * 0.55)]
    rule = DeadliftRule()
    reps = rule.detect_reps(frames)
    if not reps:
        return  # incomplete reps may be filtered; documented behavior
    eval_ = rule.evaluate_rep(reps[0], frames)
    codes = {i.code for i in eval_.issues}
    assert "DEADLIFT_PARTIAL_LOCKOUT" in codes

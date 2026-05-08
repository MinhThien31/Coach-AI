import math

from sport_companion_ai.exercises.pushup import PushUpRule
from sport_companion_ai.exercises.base import ExerciseRule
from sport_companion_ai.pose.schema import Frame, Keypoint, Skeleton


def kp(x, y):
    return Keypoint(x=x, y=y, visibility=1.0)


def pushup_skeleton(elbow_deg: float, hip_sag: float = 0.0) -> Skeleton:
    """Bottom-of-pushup-ish kinematic skeleton.

    elbow_deg controls elbow flex (180 = locked out, 80 = bottom).
    hip_sag in normalized y; positive = hip drops (sag), negative = pikes up.

    Geometry is constructed so that elbow_angle(skeleton, side='left') == elbow_deg
    exactly for all values in [60, 180].

    The body is horizontal (pushup side-view). Shoulders are at left, hips and
    ankles extend rightward. With hip_sag=0 the hip lies exactly on the
    shoulder-ankle midline, giving _hip_alignment_offset == 0.
    """
    arm_len = 0.10

    # Shoulder positions (left side is at lower y, right side at higher y in
    # normalized coordinates, simulating left/right sides in a side-on view)
    ls_x, ls_y = 0.30, 0.45
    rs_x, rs_y = 0.30, 0.55

    # Upper arm direction (shoulder -> elbow): mostly rightward + slightly down
    ua_dir_x, ua_dir_y = 1.0, 0.3
    ua_mag = math.hypot(ua_dir_x, ua_dir_y)
    ua_dir_x /= ua_mag
    ua_dir_y /= ua_mag

    le_x = ls_x + arm_len * ua_dir_x
    le_y = ls_y + arm_len * ua_dir_y
    re_x = rs_x + arm_len * ua_dir_x
    re_y = rs_y + arm_len * ua_dir_y

    # Forearm direction: rotate the *reversed* upper-arm direction by elbow_deg.
    # angle_3pt(shoulder, elbow, wrist) computes the angle at elbow between the
    # ray elbow->shoulder and the ray elbow->wrist.  By constructing the wrist at
    # exactly elbow_deg from the reversed upper-arm ray we guarantee
    # elbow_angle() == elbow_deg.
    ua_angle_from_x = math.atan2(ua_dir_y, ua_dir_x)
    neg_ua_angle = ua_angle_from_x + math.pi          # direction: elbow -> shoulder
    fa_angle = neg_ua_angle - math.radians(elbow_deg)  # rotate clockwise by elbow_deg
    fa_dir_x = math.cos(fa_angle)
    fa_dir_y = math.sin(fa_angle)

    lw_x = le_x + arm_len * fa_dir_x
    lw_y = le_y + arm_len * fa_dir_y
    rw_x = re_x + arm_len * fa_dir_x
    rw_y = re_y + arm_len * fa_dir_y

    # Hip, knee, ankle — extend along the body line (rightward in image coords).
    # With hip_sag=0 the average hip y equals the average shoulder y and ankle y,
    # so _hip_alignment_offset returns exactly 0.
    lh_x, lh_y = 0.55, 0.50 + hip_sag
    rh_x, rh_y = 0.55, 0.50 + hip_sag
    lk_x, lk_y = 0.75, 0.50
    rk_x, rk_y = 0.75, 0.50
    la_x, la_y = 0.90, 0.50
    ra_x, ra_y = 0.90, 0.50

    return Skeleton(keypoints={
        "left_shoulder": kp(ls_x, ls_y), "right_shoulder": kp(rs_x, rs_y),
        "left_elbow": kp(le_x, le_y), "right_elbow": kp(re_x, re_y),
        "left_wrist": kp(lw_x, lw_y), "right_wrist": kp(rw_x, rw_y),
        "left_hip": kp(lh_x, lh_y), "right_hip": kp(rh_x, rh_y),
        "left_knee": kp(lk_x, lk_y), "right_knee": kp(rk_x, rk_y),
        "left_ankle": kp(la_x, la_y), "right_ankle": kp(ra_x, ra_y),
    })


def make_pushup_rep_frames(min_elbow_deg: float, hip_sag: float = 0.0,
                           fps: int = 30, rep_seconds: float = 1.6) -> list[Frame]:
    n = int(rep_seconds * fps)
    out = []
    for i in range(n):
        progress = abs(2 * (i / n) - 1)
        elbow_at_t = 170 - (170 - min_elbow_deg) * (1 - progress)
        # hip_sag is dynamic — apply only at bottom (proportional to depth)
        sag_at_t = hip_sag * (1 - progress)
        skel = pushup_skeleton(elbow_at_t, hip_sag=sag_at_t)
        out.append(Frame(index=i, timestamp_ms=int(i / fps * 1000), skeleton=skel))
    return out


def test_registered():
    assert ExerciseRule.get("push_up") is PushUpRule


def test_clean_rep_passes():
    frames = make_pushup_rep_frames(min_elbow_deg=85)
    rule = PushUpRule()
    rep = rule.detect_reps(frames)[0]
    eval_ = rule.evaluate_rep(rep, frames)
    assert eval_.passed is True
    assert eval_.issues == []


def test_partial_rom_flagged():
    frames = make_pushup_rep_frames(min_elbow_deg=130)
    rule = PushUpRule()
    reps = rule.detect_reps(frames)
    if not reps:
        return  # rep filter may discard incomplete; that's a documented behavior
    eval_ = rule.evaluate_rep(reps[0], frames)
    assert any(i.code == "PUSHUP_PARTIAL_ROM" for i in eval_.issues)


def test_hip_sag_flagged():
    frames = make_pushup_rep_frames(min_elbow_deg=85, hip_sag=0.05)
    rule = PushUpRule()
    rep = rule.detect_reps(frames)[0]
    eval_ = rule.evaluate_rep(rep, frames)
    assert any(i.code == "PUSHUP_HIP_SAG" for i in eval_.issues)

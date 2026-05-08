import math

from sport_companion_ai.exercises.bicep_curl import BicepCurlRule
from sport_companion_ai.exercises.base import ExerciseRule
from sport_companion_ai.pose.schema import Frame, Keypoint, Skeleton


def kp(x, y):
    return Keypoint(x=x, y=y, visibility=1.0)


def curl_skeleton(elbow_deg: float, elbow_drift: float = 0.0) -> Skeleton:
    """Standing curl. elbow_drift > 0 moves elbow forward of shoulder.

    Geometry: shoulder above elbow above wrist (wrist position computed by
    rotating the elbow→shoulder direction by elbow_deg so that
    angle_3pt(shoulder, elbow, wrist) == elbow_deg).
    """
    ls_x, ls_y = 0.50, 0.30
    le_x, le_y = ls_x + elbow_drift, ls_y + 0.18

    # Vector from elbow to shoulder
    es_x = ls_x - le_x
    es_y = ls_y - le_y
    # Rotate by elbow_deg around the elbow to get the forearm direction.
    # In screen coords (y down), rotating the upper-arm ray by +elbow_deg
    # gives the forearm pointing roughly forward-down for typical curl angles.
    rad = math.radians(elbow_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    forearm_dx = es_x * cos_a - es_y * sin_a
    forearm_dy = es_x * sin_a + es_y * cos_a
    # Normalize to a fixed forearm length
    mag = math.hypot(forearm_dx, forearm_dy)
    if mag > 0:
        forearm_dx /= mag
        forearm_dy /= mag
    forearm_len = 0.15
    lw_x = le_x + forearm_dx * forearm_len
    lw_y = le_y + forearm_dy * forearm_len

    return Skeleton(keypoints={
        "left_shoulder": kp(ls_x, ls_y), "left_elbow": kp(le_x, le_y), "left_wrist": kp(lw_x, lw_y),
        "right_shoulder": kp(0.55, 0.30), "right_elbow": kp(0.55, 0.48), "right_wrist": kp(0.55, 0.62),
        "left_hip": kp(0.48, 0.55), "right_hip": kp(0.55, 0.55),
        "left_knee": kp(0.48, 0.75), "right_knee": kp(0.55, 0.75),
        "left_ankle": kp(0.48, 0.95), "right_ankle": kp(0.55, 0.95),
    })


def make_curl_rep_frames(min_elbow_deg: float, elbow_drift: float = 0.0,
                         fps: int = 30, rep_seconds: float = 1.6) -> list[Frame]:
    n = int(rep_seconds * fps)
    out = []
    for i in range(n):
        progress = abs(2 * (i / n) - 1)
        elbow_at_t = 170 - (170 - min_elbow_deg) * (1 - progress)
        # elbow_drift dynamic — applied proportional to depth (peak drift at bottom)
        drift_at_t = elbow_drift * (1 - progress)
        skel = curl_skeleton(elbow_at_t, elbow_drift=drift_at_t)
        out.append(Frame(index=i, timestamp_ms=int(i / fps * 1000), skeleton=skel))
    return out


def test_registered():
    assert ExerciseRule.get("bicep_curl") is BicepCurlRule


def test_clean_rep_passes():
    frames = make_curl_rep_frames(min_elbow_deg=45)
    rule = BicepCurlRule()
    rep = rule.detect_reps(frames)[0]
    eval_ = rule.evaluate_rep(rep, frames)
    assert eval_.passed is True


def test_short_rom_flagged():
    frames = make_curl_rep_frames(min_elbow_deg=110)
    rule = BicepCurlRule()
    reps = rule.detect_reps(frames)
    if not reps:
        return
    eval_ = rule.evaluate_rep(reps[0], frames)
    assert any(i.code == "CURL_PARTIAL_ROM" for i in eval_.issues)


def test_elbow_drift_flagged():
    frames = make_curl_rep_frames(min_elbow_deg=45, elbow_drift=0.05)
    rule = BicepCurlRule()
    rep = rule.detect_reps(frames)[0]
    eval_ = rule.evaluate_rep(rep, frames)
    assert any(i.code == "CURL_ELBOW_DRIFT" for i in eval_.issues)

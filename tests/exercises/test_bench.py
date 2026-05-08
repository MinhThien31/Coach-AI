import math

from sport_companion_ai.exercises.bench import BenchRule
from sport_companion_ai.exercises.base import ExerciseRule
from sport_companion_ai.pose.schema import Frame, Keypoint, Skeleton


def kp(x, y):
    return Keypoint(x=x, y=y, visibility=1.0)


def bench_skeleton(elbow_deg: float, flare: float = 0.0) -> Skeleton:
    """Side-on bench press view (lying down). elbow_deg controls flex.
    flare > 0 = elbow further out from torso (wider flare).

    Geometry: shoulders fixed; elbow position offset laterally by `flare` plus
    upper arm length × sin/cos of half-angle; wrist computed by rotating the
    elbow→shoulder ray by elbow_deg so angle_3pt(shoulder, elbow, wrist) == elbow_deg.
    """
    ls_x, ls_y = 0.45, 0.50
    rs_x, rs_y = 0.55, 0.50

    # Left arm: shoulder→elbow ray goes outward (negative x for left side) and downward
    upper_arm_len = 0.18
    le_x = ls_x - 0.12 - flare       # outward
    le_y = ls_y - 0.05               # slightly above shoulder y when bar is up
    re_x = rs_x + 0.12 + flare
    re_y = rs_y - 0.05

    def wrist_pos(sh_x, sh_y, el_x, el_y, deg):
        """Rotate elbow→shoulder by `deg` to get forearm direction."""
        es_x = sh_x - el_x
        es_y = sh_y - el_y
        rad = math.radians(deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        fx = es_x * cos_a - es_y * sin_a
        fy = es_x * sin_a + es_y * cos_a
        mag = math.hypot(fx, fy)
        if mag > 0:
            fx /= mag
            fy /= mag
        forearm_len = 0.15
        return el_x + fx * forearm_len, el_y + fy * forearm_len

    lw_x, lw_y = wrist_pos(ls_x, ls_y, le_x, le_y, elbow_deg)
    rw_x, rw_y = wrist_pos(rs_x, rs_y, re_x, re_y, -elbow_deg)  # mirror

    return Skeleton(keypoints={
        "left_shoulder": kp(ls_x, ls_y), "right_shoulder": kp(rs_x, rs_y),
        "left_elbow": kp(le_x, le_y), "right_elbow": kp(re_x, re_y),
        "left_wrist": kp(lw_x, lw_y), "right_wrist": kp(rw_x, rw_y),
        "left_hip": kp(0.45, 0.65), "right_hip": kp(0.55, 0.65),
        "left_knee": kp(0.40, 0.85), "right_knee": kp(0.60, 0.85),
        "left_ankle": kp(0.40, 0.95), "right_ankle": kp(0.60, 0.95),
    })


def make_bench_rep_frames(min_elbow_deg: float, flare: float = 0.0,
                          fps: int = 30, rep_seconds: float = 1.8) -> list[Frame]:
    n = int(rep_seconds * fps)
    out = []
    for i in range(n):
        progress = abs(2 * (i / n) - 1)
        elbow_at_t = 170 - (170 - min_elbow_deg) * (1 - progress)
        # flare proportional to depth (zero at top, peak at bottom)
        flare_at_t = flare * (1 - progress)
        skel = bench_skeleton(elbow_at_t, flare=flare_at_t)
        out.append(Frame(index=i, timestamp_ms=int(i / fps * 1000), skeleton=skel))
    return out


def test_registered():
    assert ExerciseRule.get("bench_press") is BenchRule


def test_clean_rep_passes():
    frames = make_bench_rep_frames(min_elbow_deg=85)
    rule = BenchRule()
    rep = rule.detect_reps(frames)[0]
    eval_ = rule.evaluate_rep(rep, frames)
    assert eval_.passed is True


def test_partial_rom_flagged():
    frames = make_bench_rep_frames(min_elbow_deg=125)
    rule = BenchRule()
    reps = rule.detect_reps(frames)
    if not reps:
        return
    eval_ = rule.evaluate_rep(reps[0], frames)
    assert any(i.code == "BENCH_PARTIAL_ROM" for i in eval_.issues)


def test_elbow_flare_flagged():
    frames = make_bench_rep_frames(min_elbow_deg=85, flare=0.10)
    rule = BenchRule()
    rep = rule.detect_reps(frames)[0]
    eval_ = rule.evaluate_rep(rep, frames)
    assert any(i.code == "BENCH_ELBOW_FLARE" for i in eval_.issues)

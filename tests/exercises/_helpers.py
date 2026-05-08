"""Shared helpers for building fake skeletons in rule tests."""
import math

from sport_companion_ai.pose.schema import Frame, Keypoint, Skeleton


def kp(x: float, y: float, vis: float = 1.0) -> Keypoint:
    return Keypoint(x=max(0.0, min(1.0, x)), y=max(0.0, min(1.0, y)), visibility=vis)


def squat_skeleton(min_knee_deg: float, back_deg: float = 40.0,
                   knee_offset: float = 0.0) -> Skeleton:
    """Build a squat skeleton at the bottom position.

    Design:
    - Shin is vertical: ankle is directly below the *neutral* knee position (x=lk_x0).
    - Thigh leans backward at angle ``(180 - min_knee_deg)`` from vertical, giving
      ``angle_3pt(hip, knee, ankle) == min_knee_deg`` when ``knee_offset == 0``.
    - ``knee_offset`` shifts the knee *only* inward (toward centre) without moving
      the hip or ankle, simulating valgus collapse.  This keeps the neutral
      ``knee_valgus_ratio`` small while a non-zero offset pushes it above the 0.15
      threshold.

    Args:
        min_knee_deg: Desired knee-joint angle (degrees) when ``knee_offset == 0``.
            Smaller → deeper squat (e.g. 88 ≈ parallel), larger → shallower (e.g. 115).
        back_deg: Desired torso-lean angle from vertical for the ``back_angle()``
            helper.  Should pass straight through (back_angle returns exactly
            ``back_deg`` for ``knee_offset == 0``).
        knee_offset: Normalised-coordinate x-shift applied to both knees toward the
            midline (positive = inward valgus).  Hip and ankle positions are
            *not* affected, so the midline stays fixed while the knee moves.
    """
    thigh_len = 0.15
    shin_len = 0.22

    # Neutral (zero-offset) knee positions — wide symmetric stance.
    lk_x0, lk_y0 = 0.25, 0.60
    rk_x0, rk_y0 = 0.75, 0.60

    # Thigh backward-lean angle: arcsin gives knee angle == min_knee_deg.
    thigh_t = math.radians(180 - min_knee_deg)

    # Hip positions derived from the *neutral* knee (independent of knee_offset).
    lh_x = lk_x0 - thigh_len * math.sin(thigh_t)
    lh_y = lk_y0 - thigh_len * math.cos(thigh_t)
    rh_x = rk_x0 + thigh_len * math.sin(thigh_t)
    rh_y = rk_y0 - thigh_len * math.cos(thigh_t)

    # Ankle positions: directly below the *neutral* knee (vertical shin).
    la_x, la_y = lk_x0, lk_y0 + shin_len
    ra_x, ra_y = rk_x0, rk_y0 + shin_len

    # Shifted knee positions (knee_offset moves knees inward for valgus simulation).
    lk_x = lk_x0 + knee_offset   # left knee → rightward (toward centre)
    lk_y = lk_y0
    rk_x = rk_x0 - knee_offset   # right knee → leftward (toward centre)
    rk_y = rk_y0

    # Shoulder positions for back_angle check.
    mid_hip_x = (lh_x + rh_x) / 2
    mid_hip_y = (lh_y + rh_y) / 2
    torso_len = 0.28
    back_r = math.radians(back_deg)
    mid_sh_x = mid_hip_x + torso_len * math.sin(back_r)  # forward lean → +x
    mid_sh_y = mid_hip_y - torso_len * math.cos(back_r)  # upward → −y
    sh_half_w = 0.08

    return Skeleton(keypoints={
        "left_hip": kp(lh_x, lh_y), "right_hip": kp(rh_x, rh_y),
        "left_knee": kp(lk_x, lk_y), "right_knee": kp(rk_x, rk_y),
        "left_ankle": kp(la_x, la_y), "right_ankle": kp(ra_x, ra_y),
        "left_shoulder": kp(mid_sh_x - sh_half_w, mid_sh_y),
        "right_shoulder": kp(mid_sh_x + sh_half_w, mid_sh_y),
    })


def make_squat_rep_frames(
    min_knee_deg: float, back_deg: float = 40.0, knee_offset: float = 0.0,
    fps: int = 30, rep_seconds: float = 2.0,
) -> list[Frame]:
    """Build a list of frames for one squat rep: descend, hit bottom, ascend.

    The helper angle parameter ``knee_at_t`` sweeps from 170° (standing) down to
    ``min_knee_deg`` (bottom) and back to 170°.  Because the new ``squat_skeleton``
    geometry maps the helper degree directly to the computed ``knee_angle()``, the
    minimum of the resulting angle series equals ``min_knee_deg``.
    """
    n = int(rep_seconds * fps)
    frames = []
    for i in range(n):
        progress = abs(2 * (i / n) - 1)  # 1 (top) → 0 (bottom) → 1 (top)
        knee_at_t = 170 - (170 - min_knee_deg) * (1 - progress)
        # Valgus is dynamic — knees collapse only as depth increases. Scale
        # knee_offset by (1 - progress) so it's zero at the top of the rep and
        # full at the bottom, preserving rep_threshold_high=160.
        offset_at_t = knee_offset * (1 - progress)
        skel = squat_skeleton(knee_at_t, back_deg=back_deg, knee_offset=offset_at_t)
        frames.append(Frame(index=i, timestamp_ms=int(i / fps * 1000), skeleton=skel))
    return frames

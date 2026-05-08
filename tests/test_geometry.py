import math
import pytest

from sport_companion_ai.geometry import (
    angle_3pt, angle_with_vertical,
    knee_angle, elbow_angle, hip_angle, back_angle, knee_valgus_ratio,
)
from sport_companion_ai.pose.schema import Keypoint, Skeleton


def kp(x, y):
    return Keypoint(x=x, y=y, visibility=1.0)


def test_angle_3pt_right_angle():
    a = kp(0.0, 0.0)
    b = kp(0.0, 0.5)
    c = kp(0.5, 0.5)
    assert angle_3pt(a, b, c) == pytest.approx(90.0, abs=0.1)


def test_angle_3pt_straight():
    a = kp(0.0, 0.5)
    b = kp(0.5, 0.5)
    c = kp(1.0, 0.5)
    assert angle_3pt(a, b, c) == pytest.approx(180.0, abs=0.1)


def test_angle_3pt_zero_length_returns_nan():
    a = kp(0.5, 0.5)
    b = kp(0.5, 0.5)
    c = kp(0.5, 0.5)
    assert math.isnan(angle_3pt(a, b, c))


def test_angle_with_vertical_zero_for_vertical_line():
    p1 = kp(0.5, 0.2)
    p2 = kp(0.5, 0.8)
    assert angle_with_vertical(p1, p2) == pytest.approx(0.0, abs=0.1)


def test_angle_with_vertical_45_deg():
    p1 = kp(0.2, 0.2)
    p2 = kp(0.5, 0.5)
    assert angle_with_vertical(p1, p2) == pytest.approx(45.0, abs=0.1)


def make_skeleton(**points):
    return Skeleton(keypoints={name: kp(x, y) for name, (x, y) in points.items()})


def test_knee_angle_extended_leg_is_180():
    skel = make_skeleton(
        left_hip=(0.5, 0.4), left_knee=(0.5, 0.6), left_ankle=(0.5, 0.8),
    )
    assert knee_angle(skel, side="left") == pytest.approx(180.0, abs=0.5)


def test_knee_angle_squat_bottom_is_about_90():
    skel = make_skeleton(
        left_hip=(0.5, 0.5), left_knee=(0.3, 0.5), left_ankle=(0.3, 0.7),
    )
    assert knee_angle(skel, side="left") == pytest.approx(90.0, abs=1.0)


def test_back_angle_vertical_torso():
    skel = make_skeleton(
        left_hip=(0.4, 0.6), right_hip=(0.6, 0.6),
        left_shoulder=(0.4, 0.4), right_shoulder=(0.6, 0.4),
    )
    assert back_angle(skel) == pytest.approx(0.0, abs=0.5)


def test_back_angle_leaning_forward():
    skel = make_skeleton(
        left_hip=(0.4, 0.6), right_hip=(0.6, 0.6),
        left_shoulder=(0.6, 0.4), right_shoulder=(0.8, 0.4),
    )
    assert back_angle(skel) == pytest.approx(45.0, abs=1.0)


def test_knee_valgus_ratio_neutral_is_zero():
    skel = make_skeleton(
        left_hip=(0.4, 0.4), left_knee=(0.4, 0.6), left_ankle=(0.4, 0.8),
        right_hip=(0.6, 0.4), right_knee=(0.6, 0.6), right_ankle=(0.6, 0.8),
    )
    assert knee_valgus_ratio(skel) == pytest.approx(0.0, abs=0.01)


def test_knee_valgus_ratio_inward_collapse():
    skel = make_skeleton(
        left_hip=(0.4, 0.4), left_knee=(0.5, 0.6), left_ankle=(0.4, 0.8),
        right_hip=(0.6, 0.4), right_knee=(0.5, 0.6), right_ankle=(0.6, 0.8),
    )
    # Each knee deviates by 0.1 toward midline; hip width = 0.2 → ratio = 0.5
    assert knee_valgus_ratio(skel) == pytest.approx(0.5, abs=0.01)

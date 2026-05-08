import pytest
from pydantic import ValidationError

from sport_companion_ai.pose.schema import Keypoint, Skeleton, Frame, KEYPOINT_NAMES, SKELETON_EDGES


def test_keypoint_normalized_coords_required():
    Keypoint(x=0.5, y=0.5)  # OK
    with pytest.raises(ValidationError):
        Keypoint(x=1.5, y=0.5)
    with pytest.raises(ValidationError):
        Keypoint(x=0.5, y=-0.1)


def test_keypoint_defaults():
    kp = Keypoint(x=0.5, y=0.5)
    assert kp.z == 0.0
    assert kp.visibility == 0.0


def test_skeleton_holds_named_keypoints():
    skel = Skeleton(keypoints={"nose": Keypoint(x=0.5, y=0.3)})
    assert skel.keypoints["nose"].x == 0.5


def test_frame_skeleton_optional():
    f = Frame(index=0, timestamp_ms=0, skeleton=None)
    assert f.skeleton is None


def test_keypoint_names_includes_required_set():
    required = {
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    }
    assert required.issubset(set(KEYPOINT_NAMES))


def test_skeleton_edges_reference_known_names():
    names = set(KEYPOINT_NAMES)
    for a, b in SKELETON_EDGES:
        assert a in names
        assert b in names

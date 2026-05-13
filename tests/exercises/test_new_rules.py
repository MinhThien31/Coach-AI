import math

import pytest

from sport_companion_ai.exercises.base import ExerciseRule
from sport_companion_ai.exercises.badminton import BadmintonRule
from sport_companion_ai.exercises.lateral_raise import LateralRaiseRule
from sport_companion_ai.exercises.lunge import LungeRule
from sport_companion_ai.exercises.overhead_press import OverheadPressRule
from sport_companion_ai.exercises.plank import PlankRule
from sport_companion_ai.exercises.pullup import PullUpRule
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


def _rep_for(frames: list[Frame]) -> Rep:
    return Rep(rep_index=0, start_idx=0, peak_idx=len(frames) // 2, end_idx=len(frames) - 1)


@pytest.mark.parametrize(("name", "rule"), [
    ("badminton", BadmintonRule),
    ("overhead_press", OverheadPressRule),
    ("pull_up", PullUpRule),
    ("lunge", LungeRule),
    ("plank", PlankRule),
    ("lateral_raise", LateralRaiseRule),
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


def test_badminton_clean_and_issues():
    rule = BadmintonRule()
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


@pytest.mark.parametrize(("rule", "rep"), [
    (BadmintonRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (OverheadPressRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (PullUpRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (LungeRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (PlankRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
    (LateralRaiseRule(), Rep(rep_index=0, start_idx=0, peak_idx=0, end_idx=0)),
])
def test_new_rules_missing_keypoints_inconclusive(rule, rep):
    eval_ = rule.evaluate_rep(rep, [Frame(index=0, timestamp_ms=0, skeleton=None)])
    assert eval_.inconclusive is True
    assert eval_.score is None

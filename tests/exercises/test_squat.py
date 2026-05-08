from sport_companion_ai.exercises.squat import SquatRule
from sport_companion_ai.exercises.base import ExerciseRule
from sport_companion_ai.report import Rep
from tests.exercises._helpers import make_squat_rep_frames


def test_registered():
    assert ExerciseRule.get("squat") is SquatRule


def test_detect_reps_finds_one_rep():
    frames = make_squat_rep_frames(min_knee_deg=88)
    rule = SquatRule()
    reps = rule.detect_reps(frames)
    assert len(reps) == 1


def test_clean_rep_passes():
    frames = make_squat_rep_frames(min_knee_deg=88, back_deg=42)
    rule = SquatRule()
    rep = rule.detect_reps(frames)[0]
    eval_ = rule.evaluate_rep(rep, frames)
    assert eval_.passed is True
    assert eval_.score >= 90
    assert eval_.issues == []


def test_shallow_rep_fails_with_depth_issue():
    frames = make_squat_rep_frames(min_knee_deg=115)
    rule = SquatRule()
    rep = rule.detect_reps(frames)[0] if rule.detect_reps(frames) else Rep(
        rep_index=0, start_idx=0, peak_idx=len(frames) // 2, end_idx=len(frames) - 1)
    eval_ = rule.evaluate_rep(rep, frames)
    assert eval_.passed is False
    codes = {i.code for i in eval_.issues}
    assert "SQUAT_DEPTH_INSUFFICIENT" in codes


def test_excessive_forward_lean_flagged():
    frames = make_squat_rep_frames(min_knee_deg=90, back_deg=70)
    rule = SquatRule()
    rep = rule.detect_reps(frames)[0]
    eval_ = rule.evaluate_rep(rep, frames)
    codes = {i.code for i in eval_.issues}
    assert "SQUAT_FORWARD_LEAN" in codes


def test_knee_valgus_flagged():
    frames = make_squat_rep_frames(min_knee_deg=90, back_deg=42, knee_offset=0.06)
    rule = SquatRule()
    rep = rule.detect_reps(frames)[0]
    eval_ = rule.evaluate_rep(rep, frames)
    codes = {i.code for i in eval_.issues}
    assert "SQUAT_KNEE_VALGUS" in codes


def test_metrics_reported():
    frames = make_squat_rep_frames(min_knee_deg=88)
    rule = SquatRule()
    rep = rule.detect_reps(frames)[0]
    eval_ = rule.evaluate_rep(rep, frames)
    assert "min_knee_angle" in eval_.metrics
    assert "back_angle_at_bottom" in eval_.metrics
    assert "knee_valgus_ratio" in eval_.metrics
    assert "rep_duration_ms" in eval_.metrics

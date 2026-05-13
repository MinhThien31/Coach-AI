import json

from sport_companion_ai.pose.schema import Frame
from sport_companion_ai.report import (
    AnalysisReport, Issue, Rep, RepEvaluation,
    SkeletonSchema, VideoMeta, AnalysisWarning, Severity,
)


def test_issue_severity_literal():
    issue = Issue(code="X", severity="HIGH", message_vi="...")
    assert issue.severity == "HIGH"


def test_rep_evaluation_inconclusive_defaults():
    e = RepEvaluation(rep_index=0, score=None, passed=None, inconclusive=True)
    assert e.issues == []
    assert e.metrics == {}
    assert e.keyframes == {}


def test_analysis_report_serializes_to_json():
    report = AnalysisReport(
        exercise="squat",
        pose_model="mediapipe-blazepose-full",
        video=VideoMeta(width=1080, height=1920, fps=30, duration_ms=6000),
        skeleton_schema=SkeletonSchema(keypoint_names=["nose"], edges=[]),
    )
    text = report.model_dump_json()
    parsed = json.loads(text)
    assert parsed["exercise"] == "squat"
    assert parsed["enriched"] is False
    assert parsed["session_summary"] is None
    assert parsed["ai_feedback"] is None
    assert parsed["frames"] == []


def test_warning_minimal():
    w = AnalysisWarning(code="LOW_FPS")
    assert w.code == "LOW_FPS"
    assert w.message_vi == ""

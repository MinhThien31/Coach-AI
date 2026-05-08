from sport_companion_ai.feedback.template import TemplateEnricher
from sport_companion_ai.feedback.enricher import FeedbackEnricher
from sport_companion_ai.report import (
    AnalysisReport, SkeletonSchema, VideoMeta, RepEvaluation, Issue,
)


def base_report() -> AnalysisReport:
    return AnalysisReport(
        exercise="squat", pose_model="x",
        video=VideoMeta(width=1080, height=1920, fps=30, duration_ms=6000),
        skeleton_schema=SkeletonSchema(keypoint_names=["nose"], edges=[]),
        reps=[RepEvaluation(
            rep_index=0, score=80, passed=True,
            issues=[Issue(code="X", severity="LOW", message_vi="orig")],
        )],
    )


def test_protocol():
    assert isinstance(TemplateEnricher(), FeedbackEnricher)


def test_template_is_noop():
    rep_in = base_report()
    rep_out = TemplateEnricher().enrich(rep_in)
    assert rep_out is rep_in
    assert rep_out.enriched is False
    assert rep_out.session_summary is None
    assert rep_out.reps[0].issues[0].message_vi == "orig"

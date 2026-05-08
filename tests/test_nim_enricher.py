import httpx
import pytest

from sport_companion_ai.feedback.nim import NvidiaNimEnricher
from sport_companion_ai.report import (
    AnalysisReport, SkeletonSchema, VideoMeta, RepEvaluation, Issue,
)


def base_report() -> AnalysisReport:
    return AnalysisReport(
        exercise="squat", pose_model="x",
        video=VideoMeta(width=1080, height=1920, fps=30, duration_ms=6000),
        skeleton_schema=SkeletonSchema(keypoint_names=["nose"], edges=[]),
        total_reps=1, passed_reps=1, avg_score=80,
        reps=[RepEvaluation(
            rep_index=0, score=80, passed=True,
            issues=[Issue(code="X", severity="LOW", message_vi="orig", recommendation="r")],
        )],
    )


def _ok_response(text: str = "Buổi tập tốt!"):
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": text}}]},
        request=httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions"),
    )


def test_success_sets_summary_and_enriched(mocker):
    mock_client = mocker.MagicMock()
    mock_client.post.return_value = _ok_response("Tóm tắt buổi tập")
    mock_client.__enter__ = lambda s: s
    mock_client.__exit__ = lambda s, *a: None
    mocker.patch("httpx.Client", return_value=mock_client)

    enricher = NvidiaNimEnricher(api_key="nvapi-fake")
    out = enricher.enrich(base_report())

    assert out.enriched is True
    assert out.session_summary == "Tóm tắt buổi tập"


def test_invariants_score_metrics_codes_unchanged(mocker):
    mock_client = mocker.MagicMock()
    mock_client.post.return_value = _ok_response("ok")
    mock_client.__enter__ = lambda s: s
    mock_client.__exit__ = lambda s, *a: None
    mocker.patch("httpx.Client", return_value=mock_client)

    rep_in = base_report()
    out = NvidiaNimEnricher(api_key="nvapi-fake").enrich(rep_in)
    assert out.reps[0].score == 80
    assert out.reps[0].issues[0].code == "X"
    assert out.reps[0].metrics == rep_in.reps[0].metrics
    assert out.passed_reps == 1


def test_timeout_falls_back_silently(mocker):
    mock_client = mocker.MagicMock()
    mock_client.post.side_effect = httpx.TimeoutException("slow")
    mock_client.__enter__ = lambda s: s
    mock_client.__exit__ = lambda s, *a: None
    mocker.patch("httpx.Client", return_value=mock_client)
    mocker.patch("time.sleep", return_value=None)  # don't actually sleep in retry

    enricher = NvidiaNimEnricher(api_key="nvapi-fake", max_retries=1)
    out = enricher.enrich(base_report())

    assert out.enriched is False
    assert out.session_summary is None
    assert any(w.code == "ENRICHMENT_FAILED" for w in out.warnings)


def test_http_error_falls_back(mocker):
    mock_client = mocker.MagicMock()
    mock_client.post.return_value = httpx.Response(
        500, request=httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions"))
    mock_client.__enter__ = lambda s: s
    mock_client.__exit__ = lambda s, *a: None
    mocker.patch("httpx.Client", return_value=mock_client)
    mocker.patch("time.sleep", return_value=None)

    out = NvidiaNimEnricher(api_key="nvapi-fake", max_retries=1).enrich(base_report())
    assert out.enriched is False
    assert any(w.code == "ENRICHMENT_FAILED" for w in out.warnings)

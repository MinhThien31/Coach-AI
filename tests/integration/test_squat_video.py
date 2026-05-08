"""Integration tests against fixture videos. Skipped if video files missing."""
from pathlib import Path

import pytest
import yaml

from sport_companion_ai import VideoAnalyzer


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def load_manifest():
    with open(FIXTURES_DIR / "manifest.yaml") as f:
        return yaml.safe_load(f)


def fixture_ids():
    manifest = load_manifest()
    return list(manifest.keys())


@pytest.mark.integration
@pytest.mark.parametrize("fixture_id", fixture_ids())
def test_fixture_video(fixture_id: str):
    manifest = load_manifest()
    spec = manifest[fixture_id]
    video_path = FIXTURES_DIR / spec["path"]
    if not video_path.exists():
        pytest.skip(f"fixture missing: {video_path}")

    analyzer = VideoAnalyzer()
    report = analyzer.analyze(str(video_path), exercise=spec["exercise"])

    rmin, rmax = spec["expected_reps"]["min"], spec["expected_reps"]["max"]
    assert rmin <= report.total_reps <= rmax, (
        f"expected {rmin}-{rmax} reps, got {report.total_reps}")

    if "expected_passed_reps" in spec:
        pmin, pmax = spec["expected_passed_reps"]["min"], spec["expected_passed_reps"]["max"]
        assert pmin <= report.passed_reps <= pmax

    if "expected_avg_score" in spec:
        smin, smax = spec["expected_avg_score"]["min"], spec["expected_avg_score"]["max"]
        assert smin <= report.avg_score <= smax

    issues_seen = {i.code for r in report.reps for i in r.issues}
    for must_absent in spec.get("required_issues_absent", []):
        assert must_absent not in issues_seen, f"unexpected {must_absent} in {fixture_id}"
    for must_present in spec.get("required_issues_present", []):
        assert must_present in issues_seen, f"missing {must_present} in {fixture_id}"

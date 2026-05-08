"""Integration tests for the HTTP API against real fixture videos.

Marked `integration` so PR CI skips them. They share the Phase 1
fixture set and manifest (tests/fixtures/).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def load_manifest():
    with open(FIXTURES_DIR / "manifest.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def api_client():
    # Real lifespan -> loads MediaPipe once for the whole module.
    from api.main import app
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("fixture_id", list(load_manifest().keys()))
def test_analyze_endpoint_against_fixture(api_client, fixture_id):
    spec = load_manifest()[fixture_id]
    video_path = FIXTURES_DIR / spec["path"]
    if not video_path.exists():
        pytest.skip(f"fixture missing: {video_path}")

    with open(video_path, "rb") as fh:
        files = {"video": (video_path.name, fh, "video/mp4")}
        data = {"exercise": spec["exercise"], "skeleton_output": "keyframes"}
        r = api_client.post("/analyze", files=files, data=data)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["exercise"] == spec["exercise"]
    rmin, rmax = spec["expected_reps"]["min"], spec["expected_reps"]["max"]
    assert rmin <= body["total_reps"] <= rmax


@pytest.mark.requires_nim_key
def test_analyze_endpoint_with_real_enrichment(api_client):
    if not os.getenv("NVIDIA_API_KEY"):
        pytest.skip("NVIDIA_API_KEY not set")

    fixture = next(
        (k for k, v in load_manifest().items() if v["exercise"] == "bicep_curl"),
        None,
    )
    assert fixture, "expected at least one bicep_curl fixture"
    spec = load_manifest()[fixture]
    video_path = FIXTURES_DIR / spec["path"]
    if not video_path.exists():
        pytest.skip(f"fixture missing: {video_path}")

    with open(video_path, "rb") as fh:
        files = {"video": (video_path.name, fh, "video/mp4")}
        data = {"exercise": spec["exercise"], "enrich": "true"}
        r = api_client.post("/analyze", files=files, data=data)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["enriched"] is True
    assert body["session_summary"] and len(body["session_summary"]) > 0

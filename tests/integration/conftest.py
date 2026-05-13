"""Integration-test artifact reporter.

Renders peak-rep skeleton PNGs per fixture and writes `_artifacts/results.md`
with a pass/fail review table. Runs even when assertions fail.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sport_companion_ai.viz.render import render_skeleton_png

ARTIFACTS_DIR = Path(__file__).parent / "_artifacts"

_results: list[dict[str, Any]] = []


@pytest.fixture
def record_fixture_result(request):
    captured: dict[str, Any] = {}

    def _record(fixture_id: str, spec: dict, report) -> None:
        captured["fixture_id"] = fixture_id
        captured["spec"] = spec
        captured["report"] = report

    yield _record

    if not captured:
        return

    fixture_id: str = captured["fixture_id"]
    spec: dict = captured["spec"]
    report = captured["report"]

    case_dir = ARTIFACTS_DIR / fixture_id
    case_dir.mkdir(parents=True, exist_ok=True)

    available_indices = {f.index for f in report.frames}
    image_paths: list[Path] = []
    for rep_eval in report.reps:
        peak_idx = rep_eval.keyframes.get("peak")
        if peak_idx is None or peak_idx not in available_indices:
            continue
        out_path = case_dir / f"rep-{rep_eval.rep_index:02d}-peak.png"
        try:
            render_skeleton_png(report, frame_index=peak_idx, output=str(out_path))
            image_paths.append(out_path)
        except Exception as e:
            print(f"[results] failed to render {fixture_id} rep {rep_eval.rep_index}: {e}")

    call_rep = getattr(request.node, "rep_call", None)
    passed = call_rep is not None and call_rep.passed
    failure_msg = None if passed or call_rep is None else str(call_rep.longrepr)

    _results.append({
        "fixture_id": fixture_id,
        "spec": spec,
        "report": report,
        "passed": passed,
        "image_paths": image_paths,
        "failure_msg": failure_msg,
    })


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call":
        item.rep_call = rep


def pytest_sessionfinish(session, exitstatus):
    if not _results:
        return
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / "results.md"

    n_pass = sum(1 for r in _results if r["passed"])
    n_total = len(_results)

    lines: list[str] = [
        "# Integration test results",
        "",
        f"**{n_pass} / {n_total} passed**",
        "",
        "| Fixture | Exercise | Status | Reps (got / expected) | Passed reps | Avg score | Issues seen | Skeletons (peak) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in _results:
        spec = r["spec"]
        report = r["report"]
        rmin, rmax = spec["expected_reps"]["min"], spec["expected_reps"]["max"]
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        issues = sorted({i.code for re in report.reps for i in re.issues})
        issues_str = ", ".join(issues) if issues else "—"
        img_links = (
            " ".join(
                f"[r{p.stem.split('-')[1]}]({p.relative_to(ARTIFACTS_DIR).as_posix()})"
                for p in r["image_paths"]
            )
            or "—"
        )
        lines.append(
            f"| `{r['fixture_id']}` | {spec['exercise']} | {status} | "
            f"{report.total_reps} / {rmin}-{rmax} | {report.passed_reps} | "
            f"{report.avg_score} | {issues_str} | {img_links} |"
        )

    failures = [r for r in _results if not r["passed"]]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            lines.append(f"### `{r['fixture_id']}`")
            lines.append("")
            lines.append("```")
            lines.append(r["failure_msg"] or "(no failure message captured)")
            lines.append("```")
            lines.append("")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[results] wrote {out} ({n_pass}/{n_total} passed)")

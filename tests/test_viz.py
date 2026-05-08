from pathlib import Path

from sport_companion_ai.pose.schema import Frame, Keypoint, Skeleton
from sport_companion_ai.report import (
    AnalysisReport, SkeletonSchema, VideoMeta, RepEvaluation,
)
from sport_companion_ai.viz.render import render_skeleton_png


def _report_with_frame() -> AnalysisReport:
    skel = Skeleton(keypoints={
        "left_shoulder": Keypoint(x=0.45, y=0.40, visibility=1.0),
        "left_elbow": Keypoint(x=0.40, y=0.50, visibility=1.0),
        "left_wrist": Keypoint(x=0.35, y=0.60, visibility=1.0),
        "right_shoulder": Keypoint(x=0.55, y=0.40, visibility=1.0),
        "left_hip": Keypoint(x=0.45, y=0.60, visibility=1.0),
        "right_hip": Keypoint(x=0.55, y=0.60, visibility=1.0),
    })
    return AnalysisReport(
        exercise="squat", pose_model="x",
        video=VideoMeta(width=1080, height=1920, fps=30, duration_ms=1000),
        skeleton_schema=SkeletonSchema(
            keypoint_names=["left_shoulder", "left_elbow", "left_wrist",
                            "right_shoulder", "left_hip", "right_hip"],
            edges=[("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
                   ("left_shoulder", "left_hip"), ("left_hip", "right_hip")],
        ),
        frames=[Frame(index=0, timestamp_ms=0, skeleton=skel)],
        reps=[],
    )


def test_render_skeleton_png_writes_file(tmp_path: Path):
    out = tmp_path / "skel.png"
    render_skeleton_png(_report_with_frame(), frame_index=0, output=str(out))
    assert out.exists()
    assert out.stat().st_size > 0

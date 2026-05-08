"""Dev/test helper to render a frame's skeleton as a PNG.

Pure stick-figure on white background. Optional — not part of the production pipeline.
"""
from __future__ import annotations

from sport_companion_ai.report import AnalysisReport


def render_skeleton_png(
    report: AnalysisReport,
    frame_index: int,
    output: str,
    width: int | None = None,
    height: int | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = next((f for f in report.frames if f.index == frame_index), None)
    if target is None or target.skeleton is None:
        raise ValueError(f"frame {frame_index} not present in report (or has no skeleton)")

    w = width or report.video.width
    h = height or report.video.height

    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_axis_off()

    kps = target.skeleton.keypoints
    for a, b in report.skeleton_schema.edges:
        if a in kps and b in kps:
            ka, kb = kps[a], kps[b]
            ax.plot([ka.x * w, kb.x * w], [ka.y * h, kb.y * h], "-", linewidth=3, color="#1976d2")

    for name, kp in kps.items():
        alpha = max(0.2, kp.visibility)
        ax.plot(kp.x * w, kp.y * h, "o", markersize=6, color="#d32f2f", alpha=alpha)

    fig.savefig(output, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

"""Push-up form evaluator.

Rules:
- PUSHUP_PARTIAL_ROM (MEDIUM) — min elbow angle > 110° (not deep enough)
- PUSHUP_HIP_SAG (HIGH)       — hip y deviates above shoulder/ankle line by > 0.03
- PUSHUP_HIP_PIKE (MEDIUM)    — hip y above shoulder/ankle line (pushed up)
"""
import math

from sport_companion_ai.exercises.base import ExerciseRule, register_rule
from sport_companion_ai.geometry import elbow_angle
from sport_companion_ai.pose.schema import Frame
from sport_companion_ai.report import Issue, Rep, RepEvaluation


def _hip_alignment_offset(skel) -> float:
    """Positive = hip below shoulder-ankle line (sag). Negative = pike up."""
    ls = skel.keypoints["left_shoulder"]; rs = skel.keypoints["right_shoulder"]
    lh = skel.keypoints["left_hip"]; rh = skel.keypoints["right_hip"]
    la = skel.keypoints["left_ankle"]; ra = skel.keypoints["right_ankle"]
    sh_y = (ls.y + rs.y) / 2
    hip_y = (lh.y + rh.y) / 2
    ank_y = (la.y + ra.y) / 2
    expected_hip_y = (sh_y + ank_y) / 2
    return hip_y - expected_hip_y


@register_rule
class PushUpRule(ExerciseRule):
    name = "push_up"
    primary_angle = "elbow"
    rep_threshold_low = 110.0
    rep_threshold_high = 160.0

    ROM_TARGET = 110.0
    HIP_SAG_THRESHOLD = 0.03
    HIP_PIKE_THRESHOLD = -0.03

    def _primary_angle_series(self, frames: list[Frame]) -> list[float]:
        out: list[float] = []
        for f in frames:
            if f.skeleton is None:
                out.append(float("nan"))
                continue
            try:
                out.append(elbow_angle(f.skeleton, side="left"))
            except KeyError:
                out.append(float("nan"))
        return out

    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation:
        rep_frames = [f for f in frames[rep.start_idx:rep.end_idx + 1] if f.skeleton is not None]
        if not rep_frames:
            return RepEvaluation(
                rep_index=rep.rep_index, score=None, passed=None,
                inconclusive=True, inconclusive_reason="MISSING_KEYPOINTS",
                keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
            )

        elbow_series = [elbow_angle(f.skeleton, side="left") for f in rep_frames]
        elbow_series = [v for v in elbow_series if not math.isnan(v)]
        min_elbow = min(elbow_series) if elbow_series else float("nan")
        bottom_frame = frames[rep.peak_idx]
        hip_offset = _hip_alignment_offset(bottom_frame.skeleton) if bottom_frame.skeleton else 0.0

        issues: list[Issue] = []
        score = 100

        if not math.isnan(min_elbow) and min_elbow > self.ROM_TARGET:
            issues.append(Issue(
                code="PUSHUP_PARTIAL_ROM",
                severity="MEDIUM",
                message_vi=f"Khuỷu chỉ gập đến {min_elbow:.0f}°, chưa xuống đủ sâu",
                frame_indices=[rep.peak_idx],
                recommendation="Hạ ngực gần chạm sàn rồi đẩy lên",
            ))
            score -= 15

        if hip_offset > self.HIP_SAG_THRESHOLD:
            issues.append(Issue(
                code="PUSHUP_HIP_SAG",
                severity="HIGH",
                message_vi=f"Hông sụp xuống (offset {hip_offset:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Siết bụng và mông giữ thân thẳng plank",
            ))
            score -= 25
        elif hip_offset < self.HIP_PIKE_THRESHOLD:
            issues.append(Issue(
                code="PUSHUP_HIP_PIKE",
                severity="MEDIUM",
                message_vi=f"Hông đẩy cao (offset {hip_offset:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Hạ hông xuống thẳng hàng với vai và gót",
            ))
            score -= 15

        score = max(0, score)
        passed = score >= 70 and not any(i.severity == "HIGH" for i in issues)

        return RepEvaluation(
            rep_index=rep.rep_index,
            score=score, passed=passed, inconclusive=False,
            issues=issues,
            metrics={
                "min_elbow_angle": round(min_elbow, 1) if not math.isnan(min_elbow) else None,
                "hip_alignment_offset": round(hip_offset, 3),
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )

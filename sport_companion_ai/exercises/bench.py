"""Bench press form evaluator.

Rules:
- BENCH_PARTIAL_ROM (MEDIUM)   — min elbow angle > 110° (no chest contact)
- BENCH_ELBOW_FLARE (MEDIUM)   — elbow x deviates outward from shoulder by > 0.16
- BENCH_ASYMMETRY (LOW)        — left/right elbow angles differ by > 15°
"""
import math

from sport_companion_ai.exercises.base import ExerciseRule, register_rule
from sport_companion_ai.geometry import elbow_angle
from sport_companion_ai.pose.schema import Frame
from sport_companion_ai.report import Issue, Rep, RepEvaluation


def _elbow_flare(skel) -> float:
    le = skel.keypoints["left_elbow"]; ls = skel.keypoints["left_shoulder"]
    re = skel.keypoints["right_elbow"]; rs = skel.keypoints["right_shoulder"]
    return max(abs(le.x - ls.x), abs(re.x - rs.x))


@register_rule
class BenchRule(ExerciseRule):
    name = "bench_press"
    display_name_vi = "Bench press"
    category = "upper_body_push"
    equipment = ["barbell", "bench"]
    primary_joints = ["shoulder", "elbow", "wrist"]
    issue_codes = [
        "BENCH_PARTIAL_ROM",
        "BENCH_ELBOW_FLARE",
        "BENCH_ASYMMETRY",
    ]
    primary_angle = "elbow"
    rep_threshold_low = 100.0
    rep_threshold_high = 160.0

    ROM_TARGET = 110.0
    FLARE_THRESHOLD = 0.16
    ASYMMETRY_THRESHOLD = 15.0

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

        left_series = [elbow_angle(f.skeleton, side="left") for f in rep_frames]
        right_series = [elbow_angle(f.skeleton, side="right") for f in rep_frames]
        left_series = [v for v in left_series if not math.isnan(v)]
        right_series = [v for v in right_series if not math.isnan(v)]
        min_elbow = min(left_series) if left_series else float("nan")
        max_flare = max(_elbow_flare(f.skeleton) for f in rep_frames)
        asymmetry = abs(min(left_series) - min(right_series)) if left_series and right_series else 0.0

        issues: list[Issue] = []
        score = 100

        if not math.isnan(min_elbow) and min_elbow > self.ROM_TARGET:
            issues.append(Issue(
                code="BENCH_PARTIAL_ROM", severity="MEDIUM",
                message_vi=f"Khuỷu chỉ gập đến {min_elbow:.0f}°, chưa đụng ngực",
                frame_indices=[rep.peak_idx],
                recommendation="Hạ tạ chạm nhẹ ngực rồi đẩy lên",
            ))
            score -= 15

        if max_flare > self.FLARE_THRESHOLD:
            issues.append(Issue(
                code="BENCH_ELBOW_FLARE", severity="MEDIUM",
                message_vi=f"Khuỷu xòe quá rộng (flare {max_flare:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Khuỷu nên ở khoảng 45–60° so với thân, không vuông góc",
            ))
            score -= 15

        if asymmetry > self.ASYMMETRY_THRESHOLD:
            issues.append(Issue(
                code="BENCH_ASYMMETRY", severity="LOW",
                message_vi=f"Hai bên khuỷu lệch nhau {asymmetry:.0f}°",
                frame_indices=[rep.peak_idx],
                recommendation="Kiểm tra grip cân và đẩy đều hai tay",
            ))
            score -= 5

        score = max(0, score)
        passed = score >= 70 and not any(i.severity == "HIGH" for i in issues)

        return RepEvaluation(
            rep_index=rep.rep_index,
            score=score, passed=passed, inconclusive=False,
            issues=issues,
            metrics={
                "min_elbow_angle_left": round(min_elbow, 1) if not math.isnan(min_elbow) else None,
                "max_elbow_flare": round(max_flare, 3),
                "left_right_asymmetry": round(asymmetry, 1),
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )

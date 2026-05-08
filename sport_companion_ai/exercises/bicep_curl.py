"""Bicep curl form evaluator.

Rules:
- CURL_PARTIAL_ROM (MEDIUM)   — min elbow angle > 70° (not flexed enough)
- CURL_ELBOW_DRIFT (MEDIUM)   — elbow x deviates from shoulder x > 0.04
- CURL_TOO_FAST (LOW)         — eccentric < 600 ms
"""
import math

from sport_companion_ai.exercises.base import ExerciseRule, register_rule
from sport_companion_ai.geometry import elbow_angle
from sport_companion_ai.pose.schema import Frame
from sport_companion_ai.report import Issue, Rep, RepEvaluation


def _elbow_drift(skel) -> float:
    return abs(skel.keypoints["left_elbow"].x - skel.keypoints["left_shoulder"].x)


@register_rule
class BicepCurlRule(ExerciseRule):
    name = "bicep_curl"
    primary_angle = "elbow"
    rep_threshold_low = 70.0
    rep_threshold_high = 150.0

    ROM_TARGET = 70.0
    ELBOW_DRIFT_THRESHOLD = 0.04
    REP_TOO_FAST_MS = 600

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
        max_drift = max(_elbow_drift(f.skeleton) for f in rep_frames)
        rep_duration_ms = int((rep.end_idx - rep.start_idx) / self.fps * 1000)

        issues: list[Issue] = []
        score = 100

        if not math.isnan(min_elbow) and min_elbow > self.ROM_TARGET:
            issues.append(Issue(
                code="CURL_PARTIAL_ROM", severity="MEDIUM",
                message_vi=f"Khuỷu chỉ gập đến {min_elbow:.0f}°, chưa hết tầm",
                frame_indices=[rep.peak_idx],
                recommendation="Curl hết biên độ, đưa tạ gần vai",
            ))
            score -= 15

        if max_drift > self.ELBOW_DRIFT_THRESHOLD:
            issues.append(Issue(
                code="CURL_ELBOW_DRIFT", severity="MEDIUM",
                message_vi=f"Khuỷu di chuyển ra trước (drift {max_drift:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Giữ khuỷu cố định sát thân, không vung vai",
            ))
            score -= 15

        if rep_duration_ms < self.REP_TOO_FAST_MS:
            issues.append(Issue(
                code="CURL_TOO_FAST", severity="LOW",
                message_vi=f"Rep quá nhanh ({rep_duration_ms} ms)",
                frame_indices=[rep.start_idx, rep.end_idx],
                recommendation="Hạ tạ trong 2 giây để kiểm soát eccentric",
            ))
            score -= 5

        score = max(0, score)
        passed = score >= 70 and not any(i.severity == "HIGH" for i in issues)

        return RepEvaluation(
            rep_index=rep.rep_index,
            score=score, passed=passed, inconclusive=False,
            issues=issues,
            metrics={
                "min_elbow_angle": round(min_elbow, 1) if not math.isnan(min_elbow) else None,
                "max_elbow_drift": round(max_drift, 3),
                "rep_duration_ms": float(rep_duration_ms),
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )

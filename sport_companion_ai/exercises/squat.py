"""Squat form evaluator.

Rules implemented (codes):
- SQUAT_DEPTH_INSUFFICIENT (HIGH/MED) — min knee angle > 95°
- SQUAT_BACK_TOO_VERTICAL (LOW)       — back angle < 30° at bottom
- SQUAT_FORWARD_LEAN (MED)            — back angle > 60° at bottom
- SQUAT_KNEE_VALGUS (HIGH)            — knee valgus ratio > 0.15
- SQUAT_TOO_FAST (LOW)                — rep duration < 800 ms
"""
import math

from sport_companion_ai.exercises.base import ExerciseRule, register_rule
from sport_companion_ai.geometry import back_angle, knee_angle, knee_valgus_ratio
from sport_companion_ai.pose.schema import Frame
from sport_companion_ai.report import Issue, Rep, RepEvaluation


@register_rule
class SquatRule(ExerciseRule):
    name = "squat"
    display_name_vi = "Squat"
    category = "lower_body"
    equipment = ["bodyweight", "barbell"]
    primary_joints = ["hip", "knee", "ankle"]
    issue_codes = [
        "SQUAT_DEPTH_INSUFFICIENT",
        "SQUAT_BACK_TOO_VERTICAL",
        "SQUAT_FORWARD_LEAN",
        "SQUAT_KNEE_VALGUS",
        "SQUAT_TOO_FAST",
    ]
    primary_angle = "knee"
    rep_threshold_low = 100.0
    rep_threshold_high = 160.0

    # ── Form-rule thresholds (MUST match spec exactly) ──────────────────────
    DEPTH_TARGET = 95.0
    DEPTH_HIGH_SEVERITY_THRESHOLD = 110.0
    BACK_ANGLE_MIN = 30.0
    BACK_ANGLE_MAX = 60.0
    KNEE_VALGUS_THRESHOLD = 0.15
    REP_TOO_FAST_MS = 800

    def _primary_angle_series(self, frames: list[Frame]) -> list[float]:
        out: list[float] = []
        for f in frames:
            if f.skeleton is None:
                out.append(float("nan"))
                continue
            try:
                out.append(knee_angle(f.skeleton, side="left"))
            except KeyError:
                out.append(float("nan"))
        return out

    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation:
        rep_frames = [f for f in frames[rep.start_idx:rep.end_idx + 1] if f.skeleton is not None]
        if not rep_frames:
            return RepEvaluation(
                rep_index=rep.rep_index,
                score=None, passed=None,
                inconclusive=True,
                inconclusive_reason="MISSING_KEYPOINTS",
                keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
            )

        knee_series = [knee_angle(f.skeleton, side="left") for f in rep_frames]
        knee_series = [v for v in knee_series if not math.isnan(v)]
        if not knee_series:
            return RepEvaluation(
                rep_index=rep.rep_index, score=None, passed=None,
                inconclusive=True, inconclusive_reason="MISSING_KEYPOINTS",
                keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
            )

        min_knee = min(knee_series)
        max_knee = max(knee_series)

        bottom_frame = frames[rep.peak_idx]
        back_at_bottom = (
            back_angle(bottom_frame.skeleton)
            if bottom_frame.skeleton is not None
            else float("nan")
        )
        valgus = (
            knee_valgus_ratio(bottom_frame.skeleton)
            if bottom_frame.skeleton is not None
            else 0.0
        )

        fps = self.fps
        rep_duration_ms = int((rep.end_idx - rep.start_idx) / fps * 1000)

        issues: list[Issue] = []
        score = 100

        # ── Rule 1: Depth ──────────────────────────────────────────────────
        if min_knee > self.DEPTH_TARGET:
            high_sev = min_knee > self.DEPTH_HIGH_SEVERITY_THRESHOLD
            severity = "HIGH" if high_sev else "MEDIUM"
            penalty = 25 if high_sev else 10
            issues.append(Issue(
                code="SQUAT_DEPTH_INSUFFICIENT",
                severity=severity,
                message_vi=(
                    f"Hạ chưa đủ sâu, đầu gối chỉ gập {min_knee:.0f}°"
                    f" (cần ≤ {self.DEPTH_TARGET:.0f}°)"
                ),
                frame_indices=[rep.peak_idx],
                recommendation="Hạ thấp hông hơn cho đến khi đùi song song mặt đất",
            ))
            score -= penalty

        # ── Rule 2 & 3: Back angle ─────────────────────────────────────────
        if not math.isnan(back_at_bottom):
            if back_at_bottom < self.BACK_ANGLE_MIN:
                issues.append(Issue(
                    code="SQUAT_BACK_TOO_VERTICAL",
                    severity="LOW",
                    message_vi=(
                        f"Lưng dựng đứng quá ({back_at_bottom:.0f}°),"
                        " có thể dồn áp lực gối"
                    ),
                    frame_indices=[rep.peak_idx],
                    recommendation="Cho phép thân hơi nghiêng về trước khoảng 35–50°",
                ))
                score -= 5
            elif back_at_bottom > self.BACK_ANGLE_MAX:
                issues.append(Issue(
                    code="SQUAT_FORWARD_LEAN",
                    severity="MEDIUM",
                    message_vi=(
                        f"Thân nghiêng về trước quá nhiều ({back_at_bottom:.0f}°)"
                    ),
                    frame_indices=[rep.peak_idx],
                    recommendation="Giữ ngực lên, cốt lõi căng để giảm forward lean",
                ))
                score -= 15

        # ── Rule 4: Knee valgus ────────────────────────────────────────────
        if valgus > self.KNEE_VALGUS_THRESHOLD:
            issues.append(Issue(
                code="SQUAT_KNEE_VALGUS",
                severity="HIGH",
                message_vi=(
                    f"Đầu gối quặp vào trong"
                    f" (ratio {valgus:.2f}, ngưỡng {self.KNEE_VALGUS_THRESHOLD:.2f})"
                ),
                frame_indices=[rep.peak_idx],
                recommendation="Đẩy đầu gối ra ngoài cùng hướng mũi chân khi xuống và lên",
            ))
            score -= 20

        # ── Rule 5: Rep tempo ──────────────────────────────────────────────
        if rep_duration_ms < self.REP_TOO_FAST_MS:
            issues.append(Issue(
                code="SQUAT_TOO_FAST",
                severity="LOW",
                message_vi=f"Nhịp rep quá nhanh ({rep_duration_ms} ms)",
                frame_indices=[rep.start_idx, rep.end_idx],
                recommendation="Hạ chậm 2–3 giây và lên có kiểm soát",
            ))
            score -= 5

        score = max(0, score)
        passed = score >= 70 and not any(i.severity == "HIGH" for i in issues)

        return RepEvaluation(
            rep_index=rep.rep_index,
            score=score,
            passed=passed,
            inconclusive=False,
            issues=issues,
            metrics={
                "min_knee_angle": round(min_knee, 1),
                "max_knee_angle": round(max_knee, 1),
                "back_angle_at_bottom": (
                    round(back_at_bottom, 1) if not math.isnan(back_at_bottom) else None
                ),
                "knee_valgus_ratio": round(valgus, 3),
                "rep_duration_ms": float(rep_duration_ms),
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )

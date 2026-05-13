"""Plank hold form evaluator."""
from __future__ import annotations

from sport_companion_ai.exercises._common import frames_with_keypoints, inconclusive
from sport_companion_ai.exercises.base import ExerciseRule, register_rule
from sport_companion_ai.geometry import torso_alignment_offset
from sport_companion_ai.pose.schema import Frame
from sport_companion_ai.report import Issue, Rep, RepEvaluation


REQUIRED = (
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_ankle", "right_ankle",
)


@register_rule
class PlankRule(ExerciseRule):
    name = "plank"
    display_name_vi = "Plank"
    category = "core"
    equipment = ["bodyweight"]
    movement_type = "hold"
    primary_joints = ["shoulder", "hip", "ankle", "core"]
    issue_codes = ["PLANK_HIP_SAG", "PLANK_HIP_PIKE", "PLANK_SHORT_HOLD"]
    primary_angle = "torso_alignment"
    rep_threshold_low = 0.0
    rep_threshold_high = 0.0

    HIP_SAG_THRESHOLD = 0.035
    HIP_PIKE_THRESHOLD = -0.035
    MIN_HOLD_MS = 10000

    def _primary_angle_series(self, frames: list[Frame]) -> list[float]:
        return []

    def detect_reps(self, frames: list[Frame], fps: int = 30) -> list[Rep]:
        if not frames:
            return []
        return [Rep(rep_index=0, start_idx=0, peak_idx=len(frames) // 2, end_idx=len(frames) - 1)]

    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation:
        rep_frames = frames_with_keypoints(frames, rep, REQUIRED)
        if not rep_frames:
            return inconclusive(rep)

        offsets = [torso_alignment_offset(f.skeleton) for f in rep_frames]
        avg_offset = sum(offsets) / len(offsets)
        worst_sag = max(offsets)
        worst_pike = min(offsets)
        hold_ms = int((rep.end_idx - rep.start_idx) / self.fps * 1000)

        issues: list[Issue] = []
        score = 100

        if worst_sag > self.HIP_SAG_THRESHOLD:
            issues.append(Issue(
                code="PLANK_HIP_SAG",
                severity="HIGH",
                message_vi=f"Hong bi vong xuong (offset {worst_sag:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Siet bung va mong de dua hong ve thang hang vai-got.",
            ))
            score -= 30
        elif worst_pike < self.HIP_PIKE_THRESHOLD:
            issues.append(Issue(
                code="PLANK_HIP_PIKE",
                severity="MEDIUM",
                message_vi=f"Hong day qua cao (offset {worst_pike:.2f})",
                frame_indices=[rep.peak_idx],
                recommendation="Ha hong xuong den khi than nguoi thanh mot duong thang.",
            ))
            score -= 15

        if hold_ms < self.MIN_HOLD_MS:
            issues.append(Issue(
                code="PLANK_SHORT_HOLD",
                severity="LOW",
                message_vi=f"Thoi gian giu plank ngan ({hold_ms} ms)",
                frame_indices=[rep.start_idx, rep.end_idx],
                recommendation="Tang dan thoi gian giu len toi thieu 10 giay voi form on dinh.",
            ))
            score -= 5

        score = max(0, score)
        return RepEvaluation(
            rep_index=rep.rep_index,
            score=score,
            passed=score >= 70 and not any(i.severity == "HIGH" for i in issues),
            issues=issues,
            metrics={
                "avg_torso_alignment_offset": round(avg_offset, 3),
                "worst_hip_sag": round(worst_sag, 3),
                "worst_hip_pike": round(worst_pike, 3),
                "hold_duration_ms": float(hold_ms),
            },
            keyframes={"start": rep.start_idx, "peak": rep.peak_idx, "end": rep.end_idx},
        )


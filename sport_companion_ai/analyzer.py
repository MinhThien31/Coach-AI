"""Top-level orchestration. Reads video → pose → rule eval → enriched report."""
from __future__ import annotations

from sport_companion_ai import exercises  # noqa: F401  (registers rules)
from sport_companion_ai.exercises.base import ExerciseRule
from sport_companion_ai.feedback.enricher import FeedbackEnricher
from sport_companion_ai.feedback.template import TemplateEnricher
from sport_companion_ai.pose.extractor import MediaPipeExtractor, PoseExtractor
from sport_companion_ai.pose.schema import KEYPOINT_NAMES, SKELETON_EDGES
from sport_companion_ai.pose.video_reader import read_video
from sport_companion_ai.report import (
    AnalysisReport, SkeletonSchema,
)
from sport_companion_ai.sampling import SkeletonOutputMode, select_frames_for_output
from sport_companion_ai.warnings import detect_warnings


class VideoAnalyzer:
    def __init__(
        self,
        pose_extractor: PoseExtractor | None = None,
        enricher: FeedbackEnricher | None = None,
    ):
        self.pose_extractor = pose_extractor or MediaPipeExtractor()
        self.enricher = enricher or TemplateEnricher()

    def analyze(
        self,
        video_path: str,
        exercise: str,
        skeleton_output: str | SkeletonOutputMode = SkeletonOutputMode.KEYFRAMES,
    ) -> AnalysisReport:
        rule_cls = ExerciseRule.get(exercise)
        rule = rule_cls()

        images, meta = read_video(video_path)
        rule.fps = meta.fps  # propagate so rep_duration_ms is correct

        frames = self.pose_extractor.extract_batch(images)
        reps = rule.detect_reps(frames, fps=meta.fps)
        evaluations = [rule.evaluate_rep(rep, frames) for rep in reps]

        passed = sum(1 for e in evaluations if e.passed)
        scores = [e.score for e in evaluations if e.score is not None]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        warnings = detect_warnings(frames, meta, n_reps=len(reps))

        mode = skeleton_output if isinstance(skeleton_output, SkeletonOutputMode) else SkeletonOutputMode(skeleton_output)
        out_frames = select_frames_for_output(frames, evaluations, mode, fps=meta.fps)

        report = AnalysisReport(
            exercise=exercise,
            pose_model=self.pose_extractor.model_id,
            video=meta,
            skeleton_schema=SkeletonSchema(
                keypoint_names=KEYPOINT_NAMES,
                edges=list(SKELETON_EDGES),
            ),
            frames=out_frames,
            total_reps=len(evaluations),
            passed_reps=passed,
            avg_score=round(avg_score, 1),
            reps=evaluations,
            warnings=warnings,
        )

        return self.enricher.enrich(report)

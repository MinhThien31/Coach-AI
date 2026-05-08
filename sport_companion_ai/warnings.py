"""Soft warnings collected post-pose-extraction."""
from sport_companion_ai.pose.schema import Frame
from sport_companion_ai.report import AnalysisWarning, VideoMeta


LOW_VIS_THRESHOLD = 0.5
LOW_VIS_FRAME_RATIO = 0.30
SHORT_VIDEO_MS = 3000
LONG_VIDEO_MS = 5 * 60 * 1000
LOW_FPS_THRESHOLD = 15


def _avg_visibility(frame: Frame) -> float:
    if frame.skeleton is None:
        return 0.0
    vals = [kp.visibility for kp in frame.skeleton.keypoints.values()]
    return sum(vals) / len(vals) if vals else 0.0


def detect_warnings(frames: list[Frame], meta: VideoMeta, n_reps: int) -> list[AnalysisWarning]:
    warns: list[AnalysisWarning] = []

    low_vis_count = sum(1 for f in frames if _avg_visibility(f) < LOW_VIS_THRESHOLD)
    if frames and low_vis_count / len(frames) > LOW_VIS_FRAME_RATIO:
        warns.append(AnalysisWarning(
            code="LOW_POSE_CONFIDENCE",
            message_vi=f"Khoảng {low_vis_count / len(frames) * 100:.0f}% frames có pose detection yếu",
            affected_frame_count=low_vis_count,
        ))

    if n_reps == 0:
        warns.append(AnalysisWarning(
            code="NO_REPS_DETECTED",
            message_vi="Không phát hiện rep nào — kiểm tra góc quay và vị trí người",
        ))

    if meta.duration_ms < SHORT_VIDEO_MS:
        warns.append(AnalysisWarning(
            code="VIDEO_TOO_SHORT",
            message_vi=f"Video quá ngắn ({meta.duration_ms} ms) để đánh giá đáng tin cậy",
        ))
    if meta.duration_ms > LONG_VIDEO_MS:
        warns.append(AnalysisWarning(
            code="VIDEO_TOO_LONG",
            message_vi=f"Video dài ({meta.duration_ms / 1000:.0f}s) — cảnh báo memory/thời gian xử lý",
        ))

    if meta.fps < LOW_FPS_THRESHOLD:
        warns.append(AnalysisWarning(
            code="LOW_FPS",
            message_vi=f"FPS thấp ({meta.fps}) — độ chính xác thời gian rep giảm",
        ))

    return warns

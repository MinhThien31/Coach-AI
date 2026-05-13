"""Output schema. Designed to be JSON-serializable for FE/API consumption."""
from typing import Literal

from pydantic import BaseModel

from sport_companion_ai.pose.schema import Frame


Severity = Literal["LOW", "MEDIUM", "HIGH"]
AiFeedbackStatus = Literal["good", "warning", "critical", "unknown"]


class Issue(BaseModel):
    code: str
    severity: Severity
    message_vi: str
    frame_indices: list[int] = []
    recommendation: str = ""


class Rep(BaseModel):
    rep_index: int
    start_idx: int
    peak_idx: int
    end_idx: int


class RepEvaluation(BaseModel):
    rep_index: int
    score: int | None
    passed: bool | None
    inconclusive: bool = False
    inconclusive_reason: str | None = None
    issues: list[Issue] = []
    metrics: dict[str, float | None] = {}
    keyframes: dict[str, int] = {}


class AnalysisWarning(BaseModel):
    code: str
    message_vi: str = ""
    affected_frame_count: int | None = None


class VideoMeta(BaseModel):
    width: int
    height: int
    fps: int
    duration_ms: int


class SkeletonSchema(BaseModel):
    keypoint_names: list[str]
    edges: list[tuple[str, str]]
    coordinate_space: str = "normalized"


class AiAspectFeedback(BaseModel):
    key: str
    title: str
    status: AiFeedbackStatus = "unknown"
    score: int | None = None
    actual: str = ""
    ideal: str = ""
    message: str = ""
    recommendation: str = ""


class AiJointFeedback(BaseModel):
    key: str
    title: str
    status: AiFeedbackStatus = "unknown"
    confidence: int | None = None
    message: str = ""


class AiPriorityItem(BaseModel):
    title: str
    message: str = ""
    recommendation: str = ""


class AiFeedback(BaseModel):
    overall_summary: str = ""
    aspects: list[AiAspectFeedback] = []
    joint_analysis: list[AiJointFeedback] = []
    priority_items: list[AiPriorityItem] = []
    suggestions: list[str] = []


class AnalysisReport(BaseModel):
    exercise: str
    version: str = "0.1.0"
    pose_model: str
    enriched: bool = False
    video: VideoMeta
    skeleton_schema: SkeletonSchema
    frames: list[Frame] = []
    total_reps: int = 0
    passed_reps: int = 0
    avg_score: float = 0.0
    session_summary: str | None = None
    ai_feedback: AiFeedback | None = None
    reps: list[RepEvaluation] = []
    warnings: list[AnalysisWarning] = []

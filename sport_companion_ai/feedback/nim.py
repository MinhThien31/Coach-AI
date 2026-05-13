"""NVIDIA NIM API-backed enricher.

Adds structured AI feedback for the UI and marks enriched=True. The enricher
never modifies score, passed, metrics, issue codes, or other rule outputs.
"""
from __future__ import annotations

import json
import re
import time

import httpx
from pydantic import ValidationError

from sport_companion_ai.report import AiFeedback, AnalysisReport, AnalysisWarning


NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NIM_MODEL = "qwen/qwen3-next-80b-a3b-instruct"

JOINTS_FOR_PROMPT: list[tuple[str, str, tuple[str, ...]]] = [
    ("head_neck", "Đầu & Cổ", ("nose", "left_ear", "right_ear")),
    ("right_shoulder", "Vai phải", ("right_shoulder",)),
    ("left_shoulder", "Vai trái", ("left_shoulder",)),
    ("right_elbow", "Cùi chỏ phải", ("right_elbow",)),
    ("left_elbow", "Cùi chỏ trái", ("left_elbow",)),
    ("right_wrist", "Cổ tay phải", ("right_wrist",)),
    ("left_wrist", "Cổ tay trái", ("left_wrist",)),
    ("right_hip", "Hông phải", ("right_hip",)),
    ("left_hip", "Hông trái", ("left_hip",)),
    ("right_knee", "Đầu gối phải", ("right_knee",)),
    ("left_knee", "Đầu gối trái", ("left_knee",)),
    ("right_ankle", "Cổ chân phải", ("right_ankle",)),
    ("left_ankle", "Cổ chân trái", ("left_ankle",)),
]


def _joint_visibility_summary(report: AnalysisReport) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for key, title, names in JOINTS_FOR_PROMPT:
        values: list[float] = []
        for frame in report.frames:
            if frame.skeleton is None:
                continue
            for name in names:
                point = frame.skeleton.keypoints.get(name)
                if point is not None:
                    values.append(point.visibility)
        confidence = int(round(sum(values) / len(values) * 100)) if values else None
        out.append({"key": key, "title": title, "confidence": confidence})
    return out


def _build_prompt(report: AnalysisReport) -> str:
    reps_payload = [
        {
            "rep_index": rep.rep_index,
            "score": rep.score,
            "passed": rep.passed,
            "inconclusive": rep.inconclusive,
            "metrics": rep.metrics,
            "issues": [
                {
                    "code": issue.code,
                    "severity": issue.severity,
                    "message_vi": issue.message_vi,
                    "recommendation": issue.recommendation,
                    "frame_indices": issue.frame_indices,
                }
                for issue in rep.issues
            ],
        }
        for rep in report.reps
    ]
    payload = {
        "exercise": report.exercise,
        "total_reps": report.total_reps,
        "passed_reps": report.passed_reps,
        "avg_score": report.avg_score,
        "reps": reps_payload,
        "warnings": [warning.model_dump() for warning in report.warnings],
        "joint_visibility": _joint_visibility_summary(report),
    }

    return (
        "Bạn là AI huấn luyện viên thể hình tiếng Việt. Hãy phân tích kỹ dữ liệu "
        "pose/rule engine bên dưới và trả về DUY NHẤT một JSON object hợp lệ, không markdown.\n\n"
        "Yêu cầu:\n"
        "- Không sao chép máy móc issue message; hãy diễn giải tự nhiên, cụ thể theo số liệu.\n"
        "- aspects là các mục kỹ thuật chính cần hiển thị trong tab Tổng quan.\n"
        "- joint_analysis là nhận xét từng khớp dựa trên joint_visibility và issue liên quan.\n"
        "- priority_items và suggestions là đề xuất sửa lỗi có thể hành động ngay.\n"
        "- status chỉ dùng một trong: good, warning, critical, unknown.\n"
        "- score là số 0-100 hoặc null; confidence là số 0-100 hoặc null.\n\n"
        "Schema JSON bắt buộc:\n"
        "{"
        "\"overall_summary\":\"string\","
        "\"aspects\":[{\"key\":\"string\",\"title\":\"string\",\"status\":\"good|warning|critical|unknown\","
        "\"score\":80,\"actual\":\"string\",\"ideal\":\"string\",\"message\":\"string\",\"recommendation\":\"string\"}],"
        "\"joint_analysis\":[{\"key\":\"string\",\"title\":\"string\",\"status\":\"good|warning|critical|unknown\","
        "\"confidence\":90,\"message\":\"string\"}],"
        "\"priority_items\":[{\"title\":\"string\",\"message\":\"string\",\"recommendation\":\"string\"}],"
        "\"suggestions\":[\"string\"]"
        "}\n\n"
        f"Dữ liệu phân tích:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cleaned)
    if fenced:
        cleaned = fenced.group(1)
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("NIM response did not contain a JSON object")

    decoder = json.JSONDecoder()
    try:
        data, _ = decoder.raw_decode(cleaned[start:])
    except json.JSONDecodeError:
        end = cleaned.rfind("}")
        if end == -1 or end <= start:
            raise
        data = json.loads(cleaned[start:end + 1])
    if isinstance(data, str):
        data = _extract_json_object(data)
    if not isinstance(data, dict):
        raise ValueError("NIM response JSON root must be an object")
    return data


class NvidiaNimEnricher:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_NIM_MODEL,
        timeout_s: float = 90.0,
        max_retries: int = 1,
        backoff_s: float = 1.0,
    ):
        if not api_key:
            raise ValueError("api_key required")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s

    def enrich(self, report: AnalysisReport) -> AnalysisReport:
        prompt = _build_prompt(report)
        text = self._call_with_retry(prompt)
        if text is None:
            report.warnings.append(AnalysisWarning(
                code="ENRICHMENT_FAILED",
                message_vi="LLM enrichment thất bại - không có phân tích AI",
            ))
            return report

        try:
            feedback = AiFeedback.model_validate(_extract_json_object(text))
        except (ValueError, json.JSONDecodeError, ValidationError):
            report.session_summary = text.strip()
            report.enriched = True
            return report

        report.ai_feedback = feedback
        report.session_summary = feedback.overall_summary or None
        report.enriched = True
        return report

    def _call_with_retry(self, prompt: str) -> str | None:
        attempts = 0
        while attempts <= self.max_retries:
            try:
                return self._call_once(prompt)
            except (httpx.TimeoutException, httpx.HTTPError, ValueError):
                attempts += 1
                if attempts <= self.max_retries:
                    time.sleep(self.backoff_s * (2 ** (attempts - 1)))
        return None

    def _call_once(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 4000,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(f"{NIM_BASE_URL}/chat/completions", headers=headers, json=body)
            if r.status_code == 400:
                body.pop("response_format", None)
                r = client.post(f"{NIM_BASE_URL}/chat/completions", headers=headers, json=body)
            if r.status_code >= 400:
                raise httpx.HTTPError(f"NIM returned {r.status_code}")
            data = r.json()
            try:
                message = data["choices"][0]["message"]
                content = message.get("content") or message.get("reasoning_content")
                if not content:
                    raise ValueError("empty NIM response content")
                return content
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError("unexpected NIM response shape") from exc

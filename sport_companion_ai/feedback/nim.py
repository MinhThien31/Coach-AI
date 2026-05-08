"""NVIDIA NIM API-backed enricher. Adds session_summary, marks enriched=True.

Falls back silently with ENRICHMENT_FAILED warning on any failure. Never
modifies score / passed / metrics / issue codes.
"""
import time

import httpx

from sport_companion_ai.report import AnalysisReport, AnalysisWarning


NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


def _build_prompt(report: AnalysisReport) -> str:
    issue_lines = []
    for rep in report.reps:
        for issue in rep.issues:
            issue_lines.append(
                f"- rep {rep.rep_index} ({issue.severity}): {issue.code} — metrics: {rep.metrics}"
            )
    issues_block = "\n".join(issue_lines) if issue_lines else "(không có lỗi)"

    return (
        f"Bạn là HLV gym tiếng Việt. Dựa trên dữ liệu sau, viết 2-4 câu tóm tắt "
        f"buổi tập, ghi nhận điều tốt và đề xuất 1-2 cải thiện cụ thể. "
        f"Tránh số liệu thô, dùng giọng thân thiện.\n\n"
        f"Bài: {report.exercise}\n"
        f"Tổng rep: {report.total_reps}, đạt: {report.passed_reps}, điểm TB: {report.avg_score:.0f}\n"
        f"Lỗi:\n{issues_block}"
    )


class NvidiaNimEnricher:
    def __init__(
        self,
        api_key: str,
        model: str = "meta/llama-3.3-70b-instruct",
        timeout_s: float = 10.0,
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
                message_vi="LLM enrichment thất bại — fallback về template",
            ))
            return report
        report.session_summary = text.strip()
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
            "temperature": 0.4,
            "max_tokens": 400,
            "stream": False,
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(f"{NIM_BASE_URL}/chat/completions", headers=headers, json=body)
            if r.status_code >= 400:
                raise httpx.HTTPError(f"NIM returned {r.status_code}")
            data = r.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError("unexpected NIM response shape") from exc

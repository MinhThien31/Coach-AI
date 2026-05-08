"""Feedback enricher Protocol. Default implementation in template.py."""
from typing import Protocol, runtime_checkable

from sport_companion_ai.report import AnalysisReport


@runtime_checkable
class FeedbackEnricher(Protocol):
    def enrich(self, report: AnalysisReport) -> AnalysisReport: ...

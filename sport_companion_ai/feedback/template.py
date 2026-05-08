"""Default no-op enricher. Vietnamese strings already populated by exercise rules."""
from sport_companion_ai.report import AnalysisReport


class TemplateEnricher:
    """Pass-through. enriched=False, session_summary=None remain unchanged."""

    def enrich(self, report: AnalysisReport) -> AnalysisReport:
        return report

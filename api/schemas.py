"""API-specific request DTOs. Response uses Phase 1's AnalysisReport directly."""
from __future__ import annotations

from typing import Literal

SkeletonOutputLiteral = Literal["full", "sampled", "keyframes", "none"]

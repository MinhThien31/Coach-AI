"""Exception hierarchy. Catch SportCompanionError to handle any library failure."""


class SportCompanionError(Exception):
    """Base class for all errors raised by sport-companion-ai."""


class VideoReadError(SportCompanionError):
    """Raised when a video file cannot be opened or decoded."""


class UnsupportedExerciseError(SportCompanionError):
    """Raised when an exercise name is not registered."""


class PoseExtractionError(SportCompanionError):
    """Raised when the pose extractor fails irrecoverably."""


class EnricherError(SportCompanionError):
    """Raised when an enricher must signal hard failure (rare; usually we soft-fail)."""

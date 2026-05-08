"""Base class and registry for exercise-specific rules."""
from abc import ABC, abstractmethod
from typing import ClassVar

from sport_companion_ai.errors import UnsupportedExerciseError
from sport_companion_ai.pose.schema import Frame
from sport_companion_ai.rep_detector import detect_reps_by_peaks
from sport_companion_ai.report import Rep, RepEvaluation


EXERCISE_REGISTRY: dict[str, type["ExerciseRule"]] = {}


def register_rule(cls: type["ExerciseRule"]) -> type["ExerciseRule"]:
    """Class decorator (or callable) registering an ExerciseRule subclass."""
    EXERCISE_REGISTRY[cls.name] = cls
    return cls


class ExerciseRule(ABC):
    """Strategy class for one exercise (e.g., squat).

    Subclasses must:
    - set class-level `name`, `primary_angle`, `rep_threshold_low/high`
    - implement `_primary_angle_series(frames)` to extract the angle series
    - implement `evaluate_rep(rep, frames)` to score a single rep
    """

    name: ClassVar[str]
    primary_angle: ClassVar[str]
    rep_threshold_low: ClassVar[float]
    rep_threshold_high: ClassVar[float]
    fps: ClassVar[int] = 30

    @classmethod
    def get(cls, name: str) -> type["ExerciseRule"]:
        try:
            return EXERCISE_REGISTRY[name]
        except KeyError as exc:
            raise UnsupportedExerciseError(f"Unknown exercise: {name!r}") from exc

    @abstractmethod
    def _primary_angle_series(self, frames: list[Frame]) -> list[float]:
        """Return the angle (in degrees) for each frame; NaN if skeleton missing."""

    def detect_reps(self, frames: list[Frame], fps: int = 30) -> list[Rep]:
        series = self._primary_angle_series(frames)
        return detect_reps_by_peaks(
            series,
            low_thresh=self.rep_threshold_low,
            high_thresh=self.rep_threshold_high,
            fps=fps,
        )

    @abstractmethod
    def evaluate_rep(self, rep: Rep, frames: list[Frame]) -> RepEvaluation: ...

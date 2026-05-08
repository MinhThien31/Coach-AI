import pytest

from sport_companion_ai.errors import (
    SportCompanionError,
    VideoReadError,
    UnsupportedExerciseError,
    PoseExtractionError,
    EnricherError,
)


def test_subclass_hierarchy():
    for cls in (VideoReadError, UnsupportedExerciseError, PoseExtractionError, EnricherError):
        assert issubclass(cls, SportCompanionError)


def test_can_raise_and_catch_as_base():
    with pytest.raises(SportCompanionError):
        raise VideoReadError("bad codec")

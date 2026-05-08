import pytest

from sport_companion_ai.exercises.base import ExerciseRule, EXERCISE_REGISTRY, register_rule
from sport_companion_ai.errors import UnsupportedExerciseError


def test_registry_lookup_unknown_raises():
    with pytest.raises(UnsupportedExerciseError):
        ExerciseRule.get("does_not_exist")


def test_register_and_retrieve():
    class Dummy(ExerciseRule):
        name = "dummy"
        primary_angle = "knee"
        rep_threshold_low = 100
        rep_threshold_high = 160

        def _primary_angle_series(self, frames):
            return []

        def evaluate_rep(self, rep, frames):
            raise NotImplementedError

    register_rule(Dummy)
    assert ExerciseRule.get("dummy") is Dummy
    EXERCISE_REGISTRY.pop("dummy", None)

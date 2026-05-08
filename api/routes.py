"""HTTP route handlers for /analyze, /exercises (etc)."""
from __future__ import annotations

from fastapi import APIRouter

# Import side-effect: registers all rule classes into EXERCISE_REGISTRY.
import sport_companion_ai.exercises  # noqa: F401
from sport_companion_ai.exercises.base import EXERCISE_REGISTRY

router = APIRouter()


@router.get("/exercises", tags=["meta"])
async def list_exercises() -> dict[str, list[str]]:
    return {"exercises": sorted(EXERCISE_REGISTRY.keys())}

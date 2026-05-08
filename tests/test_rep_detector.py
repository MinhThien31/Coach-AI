import math

from sport_companion_ai.rep_detector import detect_reps_by_peaks


def synthesize_reps(n_reps: int, fps: int = 30, top: float = 170.0, bottom: float = 90.0,
                   rep_seconds: float = 2.0) -> list[float]:
    """Build a sinusoidal angle series simulating reps."""
    samples_per_rep = int(rep_seconds * fps)
    series = []
    for _ in range(n_reps):
        for i in range(samples_per_rep):
            t = i / samples_per_rep
            v = top - (top - bottom) * (1 - math.cos(2 * math.pi * t)) / 2
            series.append(v)
    return series


def test_detects_three_reps():
    series = synthesize_reps(3, rep_seconds=2.0)
    reps = detect_reps_by_peaks(series, low_thresh=100, high_thresh=160, fps=30)
    assert len(reps) == 3
    assert all(r.start_idx < r.peak_idx < r.end_idx for r in reps)


def test_short_input_returns_empty():
    assert detect_reps_by_peaks([170.0, 170.0], 100, 160, fps=30) == []


def test_filters_reps_shorter_than_min_duration():
    # Quick reps (0.4s each) should be filtered out at default 500ms min
    series = synthesize_reps(5, rep_seconds=0.4)
    reps = detect_reps_by_peaks(series, low_thresh=100, high_thresh=160, fps=30,
                                min_rep_duration_ms=500)
    assert len(reps) == 0


def test_handles_nan_values():
    series = synthesize_reps(2)
    series[10] = float("nan")
    series[50] = float("nan")
    reps = detect_reps_by_peaks(series, low_thresh=100, high_thresh=160, fps=30)
    assert len(reps) == 2

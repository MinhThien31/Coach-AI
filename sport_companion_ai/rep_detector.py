"""Detect repetitions in an angle series. Pure function — no IO, no state."""
import numpy as np
from scipy.signal import find_peaks, savgol_filter

from sport_companion_ai.report import Rep


def detect_reps_by_peaks(
    angle_series: list[float],
    low_thresh: float,
    high_thresh: float,
    fps: int = 30,
    min_rep_duration_ms: int = 500,
) -> list[Rep]:
    """Find reps in a primary-joint angle series.

    A rep is a pattern: angle starts above `high_thresh`, drops below `low_thresh`
    (the rep peak is the minimum), and returns above `high_thresh`.

    Smooths input with Savitzky-Golay (window=11, polyorder=2) when long enough,
    interpolates NaNs from neighbors, and filters peaks closer than
    `min_rep_duration_ms` apart.
    """
    if len(angle_series) < 5:
        return []

    arr = np.array(angle_series, dtype=float)
    nans = np.isnan(arr)
    if nans.all():
        return []
    if nans.any():
        arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])

    window = min(11, (len(arr) // 2) * 2 + 1)
    if window >= 5 and len(arr) >= window:
        arr = savgol_filter(arr, window_length=window, polyorder=2)

    min_distance = max(1, int(min_rep_duration_ms / 1000 * fps))
    inverted = -arr
    peak_indices, _ = find_peaks(inverted, distance=min_distance, height=-low_thresh)

    reps: list[Rep] = []
    for peak_idx in peak_indices:
        start_idx = 0
        for j in range(int(peak_idx), -1, -1):
            if arr[j] >= high_thresh:
                start_idx = j
                break
        end_idx = len(arr) - 1
        for j in range(int(peak_idx), len(arr)):
            if arr[j] >= high_thresh:
                end_idx = j
                break

        duration_ms = (end_idx - start_idx) / fps * 1000
        if duration_ms < min_rep_duration_ms:
            continue

        reps.append(Rep(
            rep_index=len(reps),
            start_idx=int(start_idx),
            peak_idx=int(peak_idx),
            end_idx=int(end_idx),
        ))

    return reps

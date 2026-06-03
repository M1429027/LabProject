"""Temporal consistency helpers for cross-view matching."""

from __future__ import annotations

from typing import Any

import numpy as np


def score_temporal_consistency(track_a: dict[str, Any], track_b: dict[str, Any]) -> float:
    """Score temporal agreement between two tracklets.

    The first runnable baseline uses the amount of overlapping time as the
    primary temporal signal, normalized by the shorter track length.
    """

    start_a = int(track_a.get("start_frame", 0))
    end_a = int(track_a.get("end_frame", -1))
    start_b = int(track_b.get("start_frame", 0))
    end_b = int(track_b.get("end_frame", -1))

    overlap_start = max(start_a, start_b)
    overlap_end = min(end_a, end_b)
    overlap = max(0, overlap_end - overlap_start + 1)
    if overlap <= 0:
        return 0.0

    len_a = max(end_a - start_a + 1, 1)
    len_b = max(end_b - start_b + 1, 1)
    return float(overlap / max(min(len_a, len_b), 1))


def score_motion_consistency(
    energy_series_a: list[float],
    energy_series_b: list[float],
) -> float:
    """Compare motion-energy profiles between two synchronized tracklets."""

    if not energy_series_a or not energy_series_b:
        return 0.0
    length = min(len(energy_series_a), len(energy_series_b))
    if length <= 0:
        return 0.0
    diff = np.abs(np.asarray(energy_series_a[:length]) - np.asarray(energy_series_b[:length]))
    return float(max(0.0, 1.0 - min(float(np.mean(diff)) / 0.5, 1.0)))

"""Appearance-based similarity helpers for cross-view association."""

from __future__ import annotations

from typing import Any


def score_appearance(feature_a: Any, feature_b: Any) -> float:
    """Return a conservative appearance similarity score.

    Stage 3B currently does not compute appearance embeddings because karate
    uniforms are highly similar and we do not yet have a reliable crop-based
    encoder in the pipeline. The helper still exists so the matcher can expose
    a stable field in its score breakdown.
    """

    if feature_a is None or feature_b is None:
        return 0.0
    return 0.0

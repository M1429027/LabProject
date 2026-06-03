"""CLI for geometry-based validation of Stage 3B identity hypotheses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .hypothesis_geometry import build_track_frame_index, evaluate_hypothesis_geometry


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Score cross-view identity hypotheses with fundamental-matrix geometry."
    )
    parser.add_argument("--hypotheses-json", required=True, help="Path to global_assignment_hypotheses.json")
    parser.add_argument("--track-jsons", nargs="+", required=True, help="Per-view tracking JSON files")
    parser.add_argument("--view-ids", nargs="+", required=True, help="View ids matching --track-jsons")
    parser.add_argument("--output-dir", required=True, help="Directory for geometry validation outputs")
    parser.add_argument("--min-confidence", type=float, default=0.3, help="Minimum keypoint confidence")
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth frame")
    parser.add_argument("--max-points-per-pair", type=int, default=2000, help="Maximum correspondences per view pair")
    parser.add_argument("--ransac-threshold", type=float, default=3.0, help="RANSAC reprojection threshold in pixels")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    """Run geometry validation for all global identity hypotheses."""

    args = parse_args()
    if len(args.track_jsons) != len(args.view_ids):
        raise ValueError("track-jsons and view-ids must have the same length.")

    hypotheses_payload = load_json(args.hypotheses_json)
    frame_indexes = {
        view_id: build_track_frame_index(load_json(track_json))
        for view_id, track_json in zip(args.view_ids, args.track_jsons)
    }

    evaluations = []
    for hypothesis in hypotheses_payload.get("hypotheses", []):
        evaluations.append(
            evaluate_hypothesis_geometry(
                hypothesis=hypothesis,
                frame_indexes=frame_indexes,
                view_ids=list(args.view_ids),
                min_confidence=float(args.min_confidence),
                frame_step=int(args.frame_step),
                max_points_per_pair=int(args.max_points_per_pair),
                ransac_threshold=float(args.ransac_threshold),
            )
        )

    evaluations.sort(
        key=lambda item: (
            item["mean_geometry_score"],
            item["mean_inlier_ratio"],
            -(item["mean_median_sampson_error"] or 1e9),
        ),
        reverse=True,
    )
    for geometry_rank, item in enumerate(evaluations):
        item["geometry_rank"] = geometry_rank

    summary = {
        "stage": "hypothesis_geometry_validation",
        "scoring_note": "Higher geometry_score/inlier_ratio is better; lower Sampson error is better.",
        "settings": {
            "min_confidence": float(args.min_confidence),
            "frame_step": int(args.frame_step),
            "max_points_per_pair": int(args.max_points_per_pair),
            "ransac_threshold": float(args.ransac_threshold),
        },
        "hypotheses": evaluations,
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "hypothesis_geometry_scores.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    selected = evaluations[0] if evaluations else None
    if selected is not None:
        selected_payload = {
            "stage": "selected_hypothesis",
            "selection_method": "geometry_rank_0",
            "selection_reason": "Highest mean geometry score, then higher inlier ratio, then lower Sampson error.",
            "selected_hypothesis": selected,
        }
        with (output_dir / "selected_hypothesis.json").open("w", encoding="utf-8") as handle:
            json.dump(selected_payload, handle, ensure_ascii=False, indent=2)

    compact = {
        "stage": summary["stage"],
        "settings": summary["settings"],
        "selected_hypothesis": {
            "geometry_rank": selected["geometry_rank"],
            "original_rank": selected["rank"],
            "hypothesis_id": selected["hypothesis_id"],
            "mean_geometry_score": selected["mean_geometry_score"],
            "mean_inlier_ratio": selected["mean_inlier_ratio"],
            "mean_median_sampson_error": selected["mean_median_sampson_error"],
            "original_mean_pair_score": selected["original_mean_pair_score"],
        }
        if selected is not None
        else None,
        "top_hypotheses": [
            {
                "geometry_rank": item["geometry_rank"],
                "original_rank": item["rank"],
                "hypothesis_id": item["hypothesis_id"],
                "mean_geometry_score": item["mean_geometry_score"],
                "mean_inlier_ratio": item["mean_inlier_ratio"],
                "mean_median_sampson_error": item["mean_median_sampson_error"],
                "original_mean_pair_score": item["original_mean_pair_score"],
            }
            for item in evaluations[:8]
        ],
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(compact, handle, ensure_ascii=False, indent=2)
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

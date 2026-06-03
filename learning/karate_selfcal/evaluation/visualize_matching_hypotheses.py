"""Visualize Stage 3B two-person global identity hypotheses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .visualize_matches import build_people_index, load_json
from .visualize_matching_groups import render_group_video


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for global hypothesis visualization."""

    parser = argparse.ArgumentParser(
        description="Render multi-view review videos for Stage 3B global identity hypotheses."
    )
    parser.add_argument("--hypotheses-json", required=True, help="Path to global_assignment_hypotheses.json")
    parser.add_argument("--track-jsons", nargs="+", required=True, help="Per-view tracking JSON files")
    parser.add_argument("--source-videos", nargs="+", required=True, help="Per-view source videos")
    parser.add_argument("--view-ids", nargs="+", required=True, help="View ids matching the input order")
    parser.add_argument("--output-dir", required=True, help="Directory for rendered hypothesis videos")
    parser.add_argument("--top-k", type=int, default=4, help="Number of ranked hypotheses to render")
    parser.add_argument("--ranks", nargs="*", type=int, default=None, help="Optional specific hypothesis ranks to render")
    parser.add_argument("--frame-step", type=int, default=1, help="Render every Nth frame")
    parser.add_argument("--max-output-width", type=int, default=1920, help="Maximum output video width")
    parser.add_argument("--max-output-height", type=int, default=1080, help="Maximum output video height")
    return parser.parse_args()


def main() -> None:
    """Render identity groups for the top global assignment hypotheses."""

    args = parse_args()
    if len(args.track_jsons) != len(args.view_ids) or len(args.source_videos) != len(args.view_ids):
        raise ValueError("track-jsons, source-videos, and view-ids must have the same length.")

    hypotheses_payload = load_json(args.hypotheses_json)
    track_payloads = {
        view_id: load_json(track_json)
        for view_id, track_json in zip(args.view_ids, args.track_jsons)
    }
    frame_maps = {
        view_id: build_people_index(track_payload)
        for view_id, track_payload in track_payloads.items()
    }
    video_paths = {
        view_id: Path(video_path).resolve()
        for view_id, video_path in zip(args.view_ids, args.source_videos)
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    renders = []
    all_hypotheses = hypotheses_payload.get("hypotheses", [])
    if args.ranks:
        wanted_ranks = set(int(rank) for rank in args.ranks)
        hypotheses = [hypothesis for hypothesis in all_hypotheses if int(hypothesis["rank"]) in wanted_ranks]
    else:
        hypotheses = all_hypotheses[: int(args.top_k)]
    for hypothesis in hypotheses:
        rank = int(hypothesis["rank"])
        hypothesis_id = int(hypothesis["hypothesis_id"])
        mean_score = float(hypothesis.get("mean_pair_score", 0.0))
        min_score = float(hypothesis.get("min_pair_score", 0.0))
        for group in hypothesis.get("groups", []):
            group_id = int(group["group_id"])
            output_path = output_dir / f"hypothesis_{rank:02d}_identity_{group_id:02d}.mp4"
            renders.append(
                render_group_video(
                    group=group,
                    view_ids=list(args.view_ids),
                    frame_maps=frame_maps,
                    video_paths=video_paths,
                    output_path=output_path,
                    max_output_size=(int(args.max_output_width), int(args.max_output_height)),
                    title_lines=[
                        f"Stage 3B hypothesis rank {rank} (id {hypothesis_id}) identity {group_id}",
                        f"mean score {mean_score:.3f} | min score {min_score:.3f}",
                    ],
                    frame_step=int(args.frame_step),
                )
            )

    summary = {
        "stage": "visualize_matching_hypotheses",
        "num_hypotheses": len(hypotheses),
        "num_renders": len(renders),
        "renders": renders,
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

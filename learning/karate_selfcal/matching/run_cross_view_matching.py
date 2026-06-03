"""Entry point for Stage 3B cross-view candidate matching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .cross_view_matcher import CrossViewMatcher, build_tracklet_descriptors


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for cross-view matching."""

    parser = argparse.ArgumentParser(
        description="Run Stage 3B cross-view candidate matching on tracked views."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="learning/karate_selfcal/configs/matching.yaml",
        help="Path to the matching config YAML.",
    )
    parser.add_argument(
        "--input-tracks",
        nargs="+",
        default=None,
        help="Per-view track JSON files. Overrides config when provided.",
    )
    parser.add_argument(
        "--input-lifted",
        nargs="+",
        default=None,
        help="Optional per-view lifted JSON files aligned with --input-tracks.",
    )
    parser.add_argument(
        "--view-ids",
        nargs="+",
        default=None,
        help="Optional view ids. Must match input track count.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for cross-view matching outputs.",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def merge_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply CLI overrides onto the matching config."""

    merged = dict(config)
    merged["inputs"] = dict(config.get("inputs", {}))
    merged["outputs"] = dict(config.get("outputs", {}))
    merged["weights"] = dict(config.get("weights", {}))

    if args.input_tracks:
        merged["inputs"]["track_jsons"] = list(args.input_tracks)
    if args.input_lifted is not None:
        merged["inputs"]["lifted_jsons"] = list(args.input_lifted)
    if args.view_ids:
        merged["inputs"]["view_ids"] = list(args.view_ids)
    if args.output_dir:
        merged["outputs"]["matching_output_dir"] = args.output_dir

    return merged


def resolve_paths(config: dict[str, Any], repo_root: Path) -> tuple[list[Path], list[str], list[Path | None], Path]:
    """Resolve track, lifted, and output paths."""

    inputs_cfg = config.get("inputs", {})
    raw_tracks = inputs_cfg.get("track_jsons", [])
    if not raw_tracks:
        raise ValueError("No track JSON inputs provided in config or CLI.")

    track_paths = []
    for raw in raw_tracks:
        path = Path(raw)
        if not path.is_absolute():
            path = repo_root / path
        track_paths.append(path.resolve())

    raw_view_ids = inputs_cfg.get("view_ids")
    if raw_view_ids:
        if len(raw_view_ids) != len(track_paths):
            raise ValueError("view_ids count must match track_jsons count.")
        view_ids = [str(v) for v in raw_view_ids]
    else:
        view_ids = []
        for path in track_paths:
            stem = path.stem
            if stem.startswith("tracks_"):
                stem = stem[len("tracks_") :]
            view_ids.append(stem)

    raw_lifted = inputs_cfg.get("lifted_jsons", [])
    lifted_paths: list[Path | None] = []
    if raw_lifted:
        if len(raw_lifted) != len(track_paths):
            raise ValueError("lifted_jsons count must match track_jsons count.")
        for raw in raw_lifted:
            if raw in ("", None):
                lifted_paths.append(None)
                continue
            path = Path(raw)
            if not path.is_absolute():
                path = repo_root / path
            lifted_paths.append(path.resolve())
    else:
        lifted_paths = [None for _ in track_paths]

    output_dir = Path(config.get("outputs", {}).get("matching_output_dir", "outputs/karate_selfcal/cross_view_matching"))
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()

    return track_paths, view_ids, lifted_paths, output_dir


def load_json(path: Path | None) -> dict[str, Any] | None:
    """Load JSON when a path is provided."""

    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_summary(pair_results: dict[str, Any]) -> dict[str, Any]:
    """Build a compact run summary for candidate matching."""

    pairs = {}
    for pair_key, payload in pair_results["pair_results"].items():
        best_candidates = payload.get("candidates", [])[:3]
        pairs[pair_key] = {
            "num_candidates": len(payload.get("candidates", [])),
            "top_candidates": best_candidates,
        }
    return {
        "stage": "cross_view_matching",
        "pairs": pairs,
        "matching_graph": pair_results.get("matching_graph", {}),
        "global_assignment_hypotheses": pair_results.get("global_assignment_hypotheses", {}),
    }


def main() -> None:
    """Run Stage 3B cross-view candidate matching."""

    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    config = merge_cli_overrides(load_yaml(args.config), args)
    track_paths, view_ids, lifted_paths, output_dir = resolve_paths(config, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    descriptors_by_view = {}
    for track_path, lifted_path, view_id in zip(track_paths, lifted_paths, view_ids):
        track_payload = load_json(track_path)
        lifted_payload = load_json(lifted_path)
        descriptors_by_view[view_id] = build_tracklet_descriptors(track_payload, lifted_payload)

    matcher = CrossViewMatcher(
        weights=dict(config.get("weights", {})),
        min_keypoint_confidence=float(config.get("matching", {}).get("min_keypoint_confidence", 0.1)),
    )
    pair_results = matcher.match(descriptors_by_view)
    with (output_dir / "candidate_matches.json").open("w", encoding="utf-8") as handle:
        json.dump(pair_results["pair_results"], handle, ensure_ascii=False, indent=2)
    with (output_dir / "matching_graph.json").open("w", encoding="utf-8") as handle:
        json.dump(pair_results["matching_graph"], handle, ensure_ascii=False, indent=2)
    with (output_dir / "global_assignment_hypotheses.json").open("w", encoding="utf-8") as handle:
        json.dump(pair_results["global_assignment_hypotheses"], handle, ensure_ascii=False, indent=2)

    summary = build_summary(pair_results)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

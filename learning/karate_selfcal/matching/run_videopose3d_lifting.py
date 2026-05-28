"""Entry point for Stage 3A learned single-view lifting with VideoPose3D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .videopose3d_lifting import (
    PoseLiftingConfig,
    lift_tracked_view_with_videopose3d,
    load_json,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for learned pose lifting."""

    parser = argparse.ArgumentParser(
        description="Run MMPose VideoPose3D lifting on tracked single-view skeletons."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="learning/karate_selfcal/configs/lifting_videopose3d.yaml",
        help="Path to learned lifting config YAML.",
    )
    parser.add_argument(
        "--input-tracks",
        nargs="+",
        default=None,
        help="Per-view track JSON files. Overrides config when provided.",
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
        help="Directory for learned lifting outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override, e.g. cuda:0 or cpu.",
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
    """Apply CLI overrides to the loaded config."""

    merged = dict(config)
    merged["inputs"] = dict(config.get("inputs", {}))
    merged["outputs"] = dict(config.get("outputs", {}))
    merged["pose_lifting"] = dict(config.get("pose_lifting", {}))

    if args.input_tracks:
        merged["inputs"]["track_jsons"] = list(args.input_tracks)
    if args.view_ids:
        merged["inputs"]["view_ids"] = list(args.view_ids)
    if args.output_dir:
        merged["outputs"]["lifting_output_dir"] = args.output_dir
    if args.device:
        merged["pose_lifting"]["device"] = args.device
    return merged


def resolve_paths(config: dict[str, Any], repo_root: Path) -> tuple[list[Path], list[str], Path]:
    """Resolve input track paths and output directory."""

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

    outputs_cfg = config.get("outputs", {})
    output_dir = Path(outputs_cfg.get("lifting_output_dir", "outputs/karate_selfcal/learned_lifting"))
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()

    return track_paths, view_ids, output_dir


def build_lifting_config(config: dict[str, Any], repo_root: Path) -> PoseLiftingConfig:
    """Resolve PoseLiftingConfig from YAML content."""

    lifting_cfg = dict(config.get("pose_lifting", {}))

    def _resolve(raw: str) -> Path:
        path = Path(raw)
        if not path.is_absolute():
            path = repo_root / path
        return path.resolve()

    return PoseLiftingConfig(
        config_path=_resolve(
            lifting_cfg.get(
                "config_path",
                "/home/yp8700/amass/.venv/lib/python3.8/site-packages/mmpose/.mim/configs/body_3d_keypoint/video_pose_lift/h36m/video-pose-lift_tcn-243frm-supv_8xb128-160e_h36m.py",
            )
        ),
        checkpoint_path=_resolve(
            lifting_cfg.get(
                "checkpoint_path",
                "learning/karate_selfcal/checkpoints/videopose3d_h36m/videopose_h36m_243frames_fullconv_supervised-880bea25_20210527.pth",
            )
        ),
        device=str(lifting_cfg.get("device", "cuda:0")),
        detector_dataset_name=str(lifting_cfg.get("detector_dataset_name", "coco")),
        norm_pose_2d=bool(lifting_cfg.get("norm_pose_2d", True)),
        rebase_keypoint=bool(lifting_cfg.get("rebase_keypoint", True)),
    )


def build_summary(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build compact run summary for all lifted views."""

    views = {}
    total_frames = 0
    total_tracks = 0
    for view_id, payload in results.items():
        frames = payload.get("frames", [])
        track_descriptors = payload.get("track_descriptors", [])
        total_frames += len(frames)
        total_tracks += len(track_descriptors)
        views[view_id] = {
            "total_frames": len(frames),
            "num_tracks": len(track_descriptors),
            "lifting_method": payload.get("metadata", {}).get("lifting_method"),
            "seq_len": payload.get("metadata", {}).get("seq_len"),
        }
    return {
        "stage": "videopose3d_lifting",
        "views": views,
        "aggregate": {
            "num_views": len(results),
            "total_frames_processed": total_frames,
            "total_track_descriptors": total_tracks,
        },
    }


def main() -> None:
    """Run Stage 3A learned lifting."""

    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    config = merge_cli_overrides(load_yaml(args.config), args)
    track_paths, view_ids, output_dir = resolve_paths(config, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    lifting_cfg = build_lifting_config(config, repo_root)

    results = {}
    for track_path, view_id in zip(track_paths, view_ids):
        track_payload = load_json(track_path)
        lifted = lift_tracked_view_with_videopose3d(track_payload, lifting_cfg)
        results[view_id] = lifted
        with (output_dir / f"lifted_{view_id}.json").open("w", encoding="utf-8") as handle:
            json.dump(lifted, handle, ensure_ascii=False, indent=2)

    summary = build_summary(results)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

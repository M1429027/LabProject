"""Validate and merge one or more Stage A observation caches."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


SAMPLE_PATTERN = re.compile(r"^(?P<sequence>.+)_frame_(?P<frame>\d+)$")
REQUIRED_KEYS = {
    "camera_intrinsics",
    "camera_origins",
    "camera_rotations",
    "camera_translations",
    "image_size",
    "target_3d",
    "target_3d_root_relative",
    "views",
    "yolo_keypoints_2d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", action="append", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--require-no-occlusion", action="store_true")
    parser.add_argument("--min-frames-per-sequence", type=int, default=1)
    parser.add_argument("--namespace-source-sequences", action="store_true", default=True)
    parser.add_argument("--no-namespace-source-sequences", dest="namespace_source_sequences", action="store_false")
    return parser.parse_args()


def validate_sample(path: Path) -> tuple[str, int, bool]:
    match = SAMPLE_PATTERN.match(path.stem)
    if match is None:
        raise ValueError("sample name does not end in _frame_NNNNNN")
    with np.load(path, allow_pickle=False) as sample:
        missing = REQUIRED_KEYS.difference(sample.files)
        if missing:
            raise ValueError(f"missing keys: {sorted(missing)}")
        if np.asarray(sample["yolo_keypoints_2d"]).shape != (4, 17, 3):
            raise ValueError("yolo_keypoints_2d must have shape (4, 17, 3)")
        if np.asarray(sample["target_3d"]).shape != (17, 3):
            raise ValueError("target_3d must have shape (17, 3)")
        occlusion_keys = (
            "occlusion_view_mask",
            "occlusion_joint_mask",
            "occlusion_mask",
        )
        is_occluded = any(
            bool(np.asarray(sample[key]).any())
            for key in occlusion_keys
            if key in sample.files
        )
    return match.group("sequence"), int(match.group("frame")), is_occluded


def main() -> None:
    args = parse_args()
    entries_by_id: dict[str, dict[str, object]] = {}
    invalid: list[dict[str, str]] = []
    source_counts: dict[str, int] = {}
    skipped_occluded = 0

    for source_index, cache_dir_value in enumerate(args.cache_dir):
        cache_dir = Path(cache_dir_value)
        source_namespace = f"source_{source_index:02d}"
        source_count = 0
        for path in sorted((cache_dir / "samples").glob("*.npz")):
            try:
                sequence, frame_id, is_occluded = validate_sample(path)
            except (OSError, ValueError, EOFError) as error:
                invalid.append({"path": str(path), "error": str(error)})
                continue
            if args.require_no_occlusion and is_occluded:
                skipped_occluded += 1
                continue
            sample_id = (
                f"{source_namespace}_{path.stem}"
                if args.namespace_source_sequences
                else path.stem
            )
            sequence_id = (
                f"{source_namespace}_{sequence}"
                if args.namespace_source_sequences
                else sequence
            )
            if sample_id in entries_by_id:
                continue
            entries_by_id[sample_id] = {
                "id": sample_id,
                "path": str(path),
                "sequence": sequence_id,
                "frame_id": frame_id,
                "person_id": "amass_person01",
            }
            source_count += 1
        source_counts[str(cache_dir)] = source_count

    sequence_counts: dict[str, int] = {}
    for entry in entries_by_id.values():
        sequence = str(entry["sequence"])
        sequence_counts[sequence] = sequence_counts.get(sequence, 0) + 1
    entries = [
        entry
        for entry in entries_by_id.values()
        if sequence_counts[str(entry["sequence"])] >= args.min_frames_per_sequence
    ]
    entries.sort(key=lambda item: (str(item["sequence"]), int(item["frame_id"])))
    manifest = {
        "metadata": {
            "stage": "validated_merged_observation_cache",
            "num_entries": len(entries),
            "num_sequences": len({str(entry["sequence"]) for entry in entries}),
            "source_counts_after_deduplication": source_counts,
            "invalid_sample_count": len(invalid),
            "invalid_samples": invalid,
            "require_no_occlusion": args.require_no_occlusion,
            "skipped_occluded_count": skipped_occluded,
            "min_frames_per_sequence": args.min_frames_per_sequence,
            "namespace_source_sequences": args.namespace_source_sequences,
        },
        "entries": entries,
    }
    output = Path(args.output_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["metadata"], indent=2))


if __name__ == "__main__":
    main()

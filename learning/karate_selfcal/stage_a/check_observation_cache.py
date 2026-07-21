"""Validate a clean AMASS observation cache before head-specific expansion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_ARRAYS = (
    "clean_ray_tokens",
    "target_3d",
    "camera_intrinsics",
    "camera_rotations",
    "camera_translations",
    "camera_origins",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reports = []
    failures = []

    for entry in manifest.get("entries", []):
        sample_path = Path(entry["path"])
        if not sample_path.is_absolute() and not sample_path.exists():
            sample_path = manifest_path.parent / sample_path
        sample = np.load(sample_path)
        missing = [key for key in REQUIRED_ARRAYS if key not in sample]
        if missing:
            failures.append({"id": entry["id"], "missing": missing})
            continue

        rotations = sample["camera_rotations"]
        yolo = sample["yolo_keypoints_2d"]
        valid = yolo[..., 2] >= float(args.confidence_threshold)
        pixel_heights = []
        for view_index in range(yolo.shape[0]):
            points = yolo[view_index, valid[view_index], :2]
            pixel_heights.append(float(np.ptp(points[:, 1])) if len(points) else 0.0)

        finite = bool(all(np.isfinite(sample[key]).all() for key in REQUIRED_ARRAYS))
        orthogonality = [float(np.linalg.norm(rotation.T @ rotation - np.eye(3))) for rotation in rotations]
        determinants = [float(np.linalg.det(rotation)) for rotation in rotations]
        if not finite or max(orthogonality) > 1e-4 or max(abs(value - 1.0) for value in determinants) > 1e-4:
            failures.append({"id": entry["id"], "finite": finite, "orthogonality": orthogonality, "determinants": determinants})

        reports.append({
            "id": entry["id"],
            "active_joint_views": int(valid.sum()),
            "valid_joints_per_view": valid.sum(axis=1).astype(int).tolist(),
            "person_pixel_height_per_view": [round(value, 1) for value in pixel_heights],
            "finite": finite,
            "rotation_orthogonality_max": max(orthogonality),
            "rotation_determinants": [round(value, 6) for value in determinants],
        })

    sequence_reports = []
    for sequence in manifest.get("metadata", {}).get("sequences", []):
        angles = sorted(float(camera["azimuth_degree"]) for camera in sequence["cameras"].values())
        gaps = [angles[index + 1] - angles[index] for index in range(len(angles) - 1)]
        if angles:
            gaps.append(angles[0] + 360.0 - angles[-1])
        sequence_reports.append({
            "sequence": sequence["sequence"],
            "preset": sequence["rig_metadata"]["preset"],
            "camera_azimuths_deg": [round(value, 2) for value in angles],
            "adjacent_gaps_deg": [round(value, 2) for value in gaps],
        })

    summary = {
        "manifest": str(manifest_path),
        "num_entries": len(reports),
        "num_failures": len(failures),
        "rig_fixed_per_sequence": bool(manifest.get("metadata", {}).get("rig_fixed_per_sequence", False)),
        "extrinsic_noise": bool(manifest.get("metadata", {}).get("extrinsic_noise", True)),
        "sequences": sequence_reports,
        "samples": reports,
        "failures": failures,
    }
    output_path = Path(args.output_json) if args.output_json else manifest_path.parent / "cache_validation.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

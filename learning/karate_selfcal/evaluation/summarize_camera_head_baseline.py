"""Measure the zero-correction baseline for a camera-head dataset split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [entry for entry in manifest.get("entries", []) if entry.get("split") == args.split]
    origin_errors = []
    origin_errors_m = []
    rotation_errors = []
    translation_errors = []

    for entry in entries:
        sample = np.load(entry["path"], allow_pickle=False)
        mask = np.asarray(sample["camera_origin_delta_valid"], dtype=bool)
        origin = np.linalg.norm(np.asarray(sample["camera_origin_delta"]), axis=-1)
        translation = np.linalg.norm(np.asarray(sample["camera_translation_delta"]), axis=-1)
        rotation = np.linalg.norm(np.asarray(sample["camera_rotation_delta"]), axis=-1)
        scale = float(np.asarray(sample["ray_origin_scale"]))
        origin_errors.extend(origin[mask].tolist())
        origin_errors_m.extend((origin[mask] * scale).tolist())
        translation_errors.extend(translation[mask].tolist())
        rotation_errors.extend(rotation[mask].tolist())

    summary = {
        "manifest": str(manifest_path),
        "mode": manifest.get("metadata", {}).get("mode"),
        "split": args.split,
        "num_samples": len(entries),
        "zero_correction_camera_origin_error_normalized": float(np.mean(origin_errors)),
        "zero_correction_camera_origin_error_m": float(np.mean(origin_errors_m)),
        "zero_correction_camera_translation_error_normalized": float(np.mean(translation_errors)),
        "zero_correction_camera_rotation_error_rad": float(np.mean(rotation_errors)),
        "zero_correction_camera_rotation_error_deg": float(np.degrees(np.mean(rotation_errors))),
    }
    output_path = Path(args.output_json) if args.output_json else manifest_path.parent / f"{args.split}_zero_correction_baseline.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

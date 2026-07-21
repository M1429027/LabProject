"""Evaluate detector 2D observations against synthetic GT projections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BODY_LEFT_RIGHT_PERMUTATION = np.asarray(
    [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10], dtype=np.int64
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--key", default="detector_keypoints_2d")
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    samples = [
        np.load(entry["path"], allow_pickle=False)
        for entry in manifest.get("entries", [])
    ]
    if not samples:
        raise ValueError("Manifest has no samples.")

    view_count = int(np.asarray(samples[0][args.key]).shape[0])
    per_view = []
    all_errors = []
    for view in range(view_count):
        predicted = np.stack([sample[args.key][view, 5:] for sample in samples])
        target = np.stack([sample["gt_keypoints_2d"][view, 5:] for sample in samples])
        valid = predicted[..., 2] >= args.confidence_threshold
        errors = np.linalg.norm(predicted[..., :2] - target[..., :2], axis=-1)
        swapped_errors = np.linalg.norm(
            predicted[..., :2] - target[:, BODY_LEFT_RIGHT_PERMUTATION, :2],
            axis=-1,
        )
        body_extent = np.ptp(target[..., :2], axis=1)
        valid_errors = errors[valid]
        if len(valid_errors):
            all_errors.append(valid_errors)
        per_view.append({
            "view_index": view,
            "valid_ratio": float(np.mean(valid)),
            "mean_error_px": float(np.mean(valid_errors)) if len(valid_errors) else None,
            "p90_error_px": float(np.percentile(valid_errors, 90)) if len(valid_errors) else None,
            "swap_better_ratio": float(np.mean(swapped_errors[valid] < valid_errors))
            if len(valid_errors)
            else None,
            "mean_body_width_px": float(np.mean(body_extent[:, 0])),
            "mean_body_height_px": float(np.mean(body_extent[:, 1])),
        })

    combined = np.concatenate(all_errors) if all_errors else np.empty(0)
    report = {
        "manifest": args.manifest,
        "key": args.key,
        "num_samples": len(samples),
        "confidence_threshold": args.confidence_threshold,
        "overall": {
            "valid_ratio": float(np.mean([
                item["valid_ratio"] for item in per_view
            ])),
            "mean_error_px": float(np.mean(combined)) if len(combined) else None,
            "p90_error_px": float(np.percentile(combined, 90)) if len(combined) else None,
        },
        "per_view": per_view,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


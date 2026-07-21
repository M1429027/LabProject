"""Check that a head-only dataset contains only its intended camera error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mode = str(manifest["metadata"]["mode"])
    anchor_index = int(manifest["metadata"].get("anchor_index", 0))
    maxima = {"origin": 0.0, "translation": 0.0, "rotation": 0.0, "scale": 0.0, "anchor": 0.0}
    failures = []
    noise_stats: dict[str, dict[str, list[float]]] = {}
    case_stats: list[dict[str, object]] = []
    seen_cases: set[tuple[str, int]] = set()
    diagnostics_by_key = {
        (str(item["sequence"]), int(item["variant"])): item.get("pipeline_diagnostic", {})
        for item in manifest["metadata"].get("noise", [])
    }

    for entry in manifest.get("entries", []):
        sample = np.load(entry["path"], allow_pickle=False)
        origin = np.asarray(sample["camera_origin_delta"])
        translation = np.asarray(sample["camera_translation_delta"])
        rotation = np.asarray(sample["camera_rotation_delta"])
        scale = np.asarray(sample["camera_scale_delta"])
        maxima["origin"] = max(maxima["origin"], float(np.abs(origin).max()))
        maxima["translation"] = max(maxima["translation"], float(np.abs(translation).max()))
        maxima["rotation"] = max(maxima["rotation"], float(np.abs(rotation).max()))
        maxima["scale"] = max(maxima["scale"], float(np.abs(scale).max()))
        anchor_values = np.concatenate((origin[anchor_index], translation[anchor_index], rotation[anchor_index], scale[anchor_index : anchor_index + 1]))
        maxima["anchor"] = max(maxima["anchor"], float(np.abs(anchor_values).max()))
        source = str(entry.get("noise_source", mode))
        bucket = noise_stats.setdefault(source, {"rotation_deg": [], "center_m": []})
        non_anchor = np.arange(len(origin)) != anchor_index
        origin_scale = float(np.asarray(sample["ray_origin_scale"]).reshape(()))
        bucket["rotation_deg"].extend(
            np.linalg.norm(rotation[non_anchor], axis=-1).astype(float) * (180.0 / np.pi)
        )
        bucket["center_m"].extend(
            np.linalg.norm(origin[non_anchor], axis=-1).astype(float) * origin_scale
        )
        case_key = (str(entry.get("sequence", "")), int(entry.get("variant", 0)))
        if case_key not in seen_cases:
            seen_cases.add(case_key)
            case_stats.append({
                "sequence": case_key[0],
                "variant": case_key[1],
                "split": str(entry.get("split", "")),
                "noise_source": source,
                "max_rotation_deg": float(np.linalg.norm(rotation[non_anchor], axis=-1).max() * (180.0 / np.pi)),
                "max_center_m": float(np.linalg.norm(origin[non_anchor], axis=-1).max() * origin_scale),
                "pipeline_diagnostic": diagnostics_by_key.get(case_key, {}),
            })
        # t = -R C, so rotation-only noise changes t while the physical camera
        # center C remains fixed.
        if mode == "rotation" and (np.abs(origin).max() > args.tolerance or np.abs(scale).max() > args.tolerance):
            failures.append({"id": entry["id"], "reason": "rotation dataset contains non-rotation target"})
        if mode == "center" and (np.abs(rotation).max() > args.tolerance or np.abs(scale).max() > args.tolerance):
            failures.append({"id": entry["id"], "reason": "center dataset contains rotation/scale target"})
        if mode == "mixed_rough" and np.abs(scale).max() > args.tolerance:
            failures.append({"id": entry["id"], "reason": "mixed rough dataset contains scale target"})
        if np.abs(anchor_values).max() > args.tolerance:
            failures.append({"id": entry["id"], "reason": "anchor camera is not fixed"})

    noise_summary = {
        source: {
            metric: {
                "mean": float(np.mean(values)) if values else 0.0,
                "p90": float(np.percentile(values, 90)) if values else 0.0,
                "max": float(np.max(values)) if values else 0.0,
            }
            for metric, values in metrics.items()
        }
        for source, metrics in noise_stats.items()
    }
    summary = {
        "manifest": str(manifest_path),
        "mode": mode,
        "num_entries": len(manifest.get("entries", [])),
        "num_failures": len(failures),
        "max_absolute_targets": maxima,
        "noise_distribution_by_source": noise_summary,
        "failures": failures[:20],
        "worst_center_cases": sorted(
            case_stats, key=lambda item: float(item["max_center_m"]), reverse=True
        )[:10],
        "worst_rotation_cases": sorted(
            case_stats, key=lambda item: float(item["max_rotation_deg"]), reverse=True
        )[:10],
    }
    output_path = manifest_path.parent / "dataset_validation.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

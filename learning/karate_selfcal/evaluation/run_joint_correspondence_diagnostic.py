"""Diagnose joint-level correspondence failures in triangulated sequences."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_BONES = [
    (5, 6, "shoulder_width"),
    (11, 12, "hip_width"),
    (5, 7, "left_upper_arm"),
    (7, 9, "left_forearm"),
    (6, 8, "right_upper_arm"),
    (8, 10, "right_forearm"),
    (5, 11, "left_torso"),
    (6, 12, "right_torso"),
    (11, 13, "left_thigh"),
    (13, 15, "left_shin"),
    (12, 14, "right_thigh"),
    (14, 16, "right_shin"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize which joints/bones are likely breaking multi-view triangulation."
    )
    parser.add_argument("--triangulated-json", required=True, help="Path to triangulated_3d.json")
    parser.add_argument("--output-dir", required=True, help="Directory for diagnostic outputs")
    parser.add_argument(
        "--bone-ratio-threshold",
        type=float,
        default=1.8,
        help="Flag a bone sample when length / median_length is above this value.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def point3(joint: dict[str, Any]) -> np.ndarray:
    return np.array([float(joint["x"]), float(joint["y"]), float(joint["z"])], dtype=np.float64)


def finite_mean(values: list[float]) -> float | None:
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def percentile(values: list[float], q: float) -> float | None:
    finite = sorted(float(v) for v in values if np.isfinite(v))
    if not finite:
        return None
    idx = int(round((len(finite) - 1) * q))
    return float(finite[idx])


def summarize_sequence(payload: dict[str, Any], bone_ratio_threshold: float) -> dict[str, Any]:
    frames = payload.get("frames", [])
    total_identity_frames = sum(len(frame.get("identities", [])) for frame in frames)

    joint_stats: dict[int, dict[str, Any]] = {
        idx: {
            "joint_id": idx,
            "joint_name": COCO_KEYPOINT_NAMES[idx],
            "count": 0,
            "reprojection_errors": [],
            "confidences": [],
            "num_views": [],
            "view_sets": Counter(),
            "bone_outlier_hits": 0,
            "connected_bone_samples": 0,
        }
        for idx in range(len(COCO_KEYPOINT_NAMES))
    }
    bone_lengths: dict[str, list[float]] = defaultdict(list)
    bone_samples: list[dict[str, Any]] = []

    for frame in frames:
        frame_idx = int(frame.get("frame", -1))
        for identity in frame.get("identities", []):
            identity_id = int(identity.get("identity_id", -1))
            joints = {int(joint["id"]): joint for joint in identity.get("joints", [])}
            for joint_id, joint in joints.items():
                if joint_id not in joint_stats:
                    continue
                stats = joint_stats[joint_id]
                stats["count"] += 1
                if joint.get("mean_reprojection_error_px") is not None:
                    stats["reprojection_errors"].append(float(joint["mean_reprojection_error_px"]))
                if joint.get("mean_confidence") is not None:
                    stats["confidences"].append(float(joint["mean_confidence"]))
                stats["num_views"].append(int(joint.get("num_views", 0)))
                stats["view_sets"][tuple(sorted(joint.get("views", [])))] += 1

            for joint_a, joint_b, bone_name in COCO_BONES:
                if joint_a not in joints or joint_b not in joints:
                    continue
                length = float(np.linalg.norm(point3(joints[joint_a]) - point3(joints[joint_b])))
                bone_lengths[bone_name].append(length)
                bone_samples.append(
                    {
                        "frame": frame_idx,
                        "identity_id": identity_id,
                        "bone_name": bone_name,
                        "joint_a": joint_a,
                        "joint_b": joint_b,
                        "length": length,
                    }
                )
                joint_stats[joint_a]["connected_bone_samples"] += 1
                joint_stats[joint_b]["connected_bone_samples"] += 1

    bone_medians = {
        bone_name: float(median(lengths))
        for bone_name, lengths in bone_lengths.items()
        if lengths
    }

    bone_outliers = []
    for sample in bone_samples:
        med = bone_medians.get(sample["bone_name"])
        if med is None or med <= 1e-9:
            continue
        ratio = float(sample["length"] / med)
        if ratio >= float(bone_ratio_threshold):
            sample = {**sample, "median_length": med, "length_over_median": ratio}
            bone_outliers.append(sample)
            joint_stats[int(sample["joint_a"])]["bone_outlier_hits"] += 1
            joint_stats[int(sample["joint_b"])]["bone_outlier_hits"] += 1

    joint_rows = []
    for joint_id, stats in joint_stats.items():
        count = int(stats["count"])
        connected = int(stats["connected_bone_samples"])
        outlier_hits = int(stats["bone_outlier_hits"])
        view_sets = [
            {
                "views": list(view_set),
                "count": int(view_count),
            }
            for view_set, view_count in stats["view_sets"].most_common(5)
        ]
        row = {
            "joint_id": joint_id,
            "joint_name": stats["joint_name"],
            "count": count,
            "coverage_over_identity_frames": (
                float(count / total_identity_frames) if total_identity_frames else None
            ),
            "mean_reprojection_error_px": finite_mean(stats["reprojection_errors"]),
            "p95_reprojection_error_px": percentile(stats["reprojection_errors"], 0.95),
            "mean_confidence": finite_mean(stats["confidences"]),
            "mean_num_views": finite_mean(stats["num_views"]),
            "bone_outlier_hits": outlier_hits,
            "bone_outlier_rate": float(outlier_hits / connected) if connected else 0.0,
            "top_view_sets": view_sets,
        }
        joint_rows.append(row)

    bone_rows = []
    for joint_a, joint_b, bone_name in COCO_BONES:
        lengths = bone_lengths.get(bone_name, [])
        med = bone_medians.get(bone_name)
        if not lengths or med is None:
            continue
        p95 = percentile(lengths, 0.95)
        bone_rows.append(
            {
                "bone_name": bone_name,
                "joint_a": joint_a,
                "joint_b": joint_b,
                "count": len(lengths),
                "median_length": med,
                "p95_length": p95,
                "p95_over_median": float(p95 / med) if p95 is not None and med > 1e-9 else None,
            }
        )

    suspicious_joints = sorted(
        joint_rows,
        key=lambda row: (
            float(row["bone_outlier_rate"]),
            float(row["p95_reprojection_error_px"] or 0.0),
            -float(row["coverage_over_identity_frames"] or 0.0),
        ),
        reverse=True,
    )

    return {
        "stage": "joint_correspondence_diagnostic",
        "source_metadata": payload.get("metadata", {}),
        "summary": {
            "num_frames": len(frames),
            "total_identity_frames": total_identity_frames,
            "bone_ratio_threshold": float(bone_ratio_threshold),
            "num_bone_outliers": len(bone_outliers),
        },
        "joints": joint_rows,
        "bones": bone_rows,
        "top_suspicious_joints": suspicious_joints[:10],
        "top_bone_outliers": sorted(
            bone_outliers,
            key=lambda item: float(item["length_over_median"]),
            reverse=True,
        )[:50],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Joint Correspondence Diagnostic",
        "",
        "## Summary",
        "",
        f"- Frames: {report['summary']['num_frames']}",
        f"- Identity frames: {report['summary']['total_identity_frames']}",
        f"- Bone outliers: {report['summary']['num_bone_outliers']}",
        "",
        "## Top Suspicious Joints",
        "",
        "| Joint | Count | Coverage | Mean reproj px | P95 reproj px | Bone outlier rate | Top view sets |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["top_suspicious_joints"]:
        view_sets = ", ".join(
            f"{'+'.join(item['views'])}:{item['count']}"
            for item in row["top_view_sets"][:3]
        )
        lines.append(
            "| {joint} | {count} | {coverage:.3f} | {mean_reproj} | {p95_reproj} | {outlier:.3f} | {views} |".format(
                joint=f"{row['joint_id']} {row['joint_name']}",
                count=row["count"],
                coverage=float(row["coverage_over_identity_frames"] or 0.0),
                mean_reproj=(
                    f"{row['mean_reprojection_error_px']:.3f}"
                    if row["mean_reprojection_error_px"] is not None
                    else "NA"
                ),
                p95_reproj=(
                    f"{row['p95_reprojection_error_px']:.3f}"
                    if row["p95_reprojection_error_px"] is not None
                    else "NA"
                ),
                outlier=float(row["bone_outlier_rate"]),
                views=view_sets,
            )
        )

    lines.extend(
        [
            "",
            "## Bone Stability",
            "",
            "| Bone | Count | Median | P95 | P95 / Median |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(report["bones"], key=lambda item: float(item["p95_over_median"] or 0.0), reverse=True):
        lines.append(
            "| {bone} | {count} | {median:.3f} | {p95:.3f} | {ratio:.3f} |".format(
                bone=row["bone_name"],
                count=row["count"],
                median=float(row["median_length"]),
                p95=float(row["p95_length"] or 0.0),
                ratio=float(row["p95_over_median"] or 0.0),
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = load_json(args.triangulated_json)
    report = summarize_sequence(payload, bone_ratio_threshold=float(args.bone_ratio_threshold))

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "joint_correspondence_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    write_markdown(report, output_dir / "joint_correspondence_report.md")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print("Top suspicious joints:")
    for row in report["top_suspicious_joints"][:8]:
        print(
            f"- {row['joint_id']:02d} {row['joint_name']}: "
            f"coverage={float(row['coverage_over_identity_frames'] or 0.0):.3f}, "
            f"p95_reproj={row['p95_reprojection_error_px']}, "
            f"bone_outlier_rate={row['bone_outlier_rate']:.3f}"
        )


if __name__ == "__main__":
    main()

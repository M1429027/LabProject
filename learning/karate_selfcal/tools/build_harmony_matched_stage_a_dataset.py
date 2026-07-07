"""Build Stage A residual inference samples from matched Harmony4D multi-person tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

COCO_NUM_JOINTS = 17
RAY_FEATURE_DIM = 11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build matched Harmony4D Stage A samples.")
    parser.add_argument("--selected-hypothesis-json", required=True)
    parser.add_argument("--rough-extrinsics-json", required=True)
    parser.add_argument("--track-jsons", nargs="+", required=True)
    parser.add_argument("--view-ids", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence", default="harmony4d/karate004_matched")
    parser.add_argument("--split", default="test")
    parser.add_argument("--min-confidence", type=float, default=0.05)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def root_relative(points: np.ndarray) -> np.ndarray:
    pelvis = 0.5 * (points[11] + points[12])
    return (points - pelvis.reshape(1, 3)).astype(np.float32)


def camera_origin_world(view_meta: dict[str, Any]) -> np.ndarray:
    r = np.asarray(view_meta["rotation"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(view_meta["translation"], dtype=np.float64).reshape(3)
    return (-r.T @ t).astype(np.float32)


def pixel_to_world_ray(x: float, y: float, view_meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    intr = view_meta["intrinsics"]
    k = np.asarray(intr["camera_matrix"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(intr.get("dist_coeffs", []), dtype=np.float64).reshape(-1)
    point = np.array([[[float(x), float(y)]]], dtype=np.float64)
    model = str(intr.get("model", "")).upper()
    if "FISHEYE" in model and dist.size >= 4:
        normalized = cv2.fisheye.undistortPoints(point, k, dist[:4]).reshape(2)
    else:
        normalized = cv2.undistortPoints(point, k, dist).reshape(2)
    direction_camera = np.array([normalized[0], normalized[1], 1.0], dtype=np.float64)
    direction_camera /= max(float(np.linalg.norm(direction_camera)), 1e-9)
    r = np.asarray(view_meta["rotation"], dtype=np.float64).reshape(3, 3)
    direction_world = r.T @ direction_camera
    direction_world /= max(float(np.linalg.norm(direction_world)), 1e-9)
    return camera_origin_world(view_meta), direction_world.astype(np.float32)


def triangulate_weighted_rays(origins: list[np.ndarray], directions: list[np.ndarray], weights: list[float]) -> tuple[np.ndarray, float]:
    if len(origins) < 2:
        return np.zeros(3, dtype=np.float32), 0.0
    a = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    weight_sum = 0.0
    for origin, direction, weight in zip(origins, directions, weights):
        d = np.asarray(direction, dtype=np.float64)
        d /= max(float(np.linalg.norm(d)), 1e-9)
        projection = eye - np.outer(d, d)
        w = max(float(weight), 1e-6)
        a += w * projection
        b += w * projection @ np.asarray(origin, dtype=np.float64)
        weight_sum += w
    try:
        point = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.zeros(3, dtype=np.float32), 0.0
    distances = []
    for origin, direction in zip(origins, directions):
        d = np.asarray(direction, dtype=np.float64)
        d /= max(float(np.linalg.norm(d)), 1e-9)
        offset = point - np.asarray(origin, dtype=np.float64)
        distances.append(float(np.linalg.norm(offset - np.dot(offset, d) * d)))
    mean_distance = float(np.mean(distances)) if distances else 1e6
    quality = float(len(origins)) * float(weight_sum / max(len(origins), 1)) / (1.0 + mean_distance)
    return point.astype(np.float32), quality


def track_frame_index(track_payload: dict[str, Any]) -> dict[int, dict[int, list[dict[str, Any]]]]:
    out: dict[int, dict[int, list[dict[str, Any]]]] = {}
    for frame in track_payload.get("frames", []):
        by_track = {}
        for person in frame.get("people", []):
            track_id = int(person.get("track_id", person.get("person_id", 0)))
            by_track[track_id] = person.get("keypoints", [])
        out[int(frame["frame"])] = by_track
    return out


def assignments_from_selected(selected_payload: dict[str, Any]) -> list[dict[str, int]]:
    selected = selected_payload["selected_hypothesis"]
    groups = []
    for group in selected["groups"]:
        mapping = {}
        for node in group["nodes"]:
            view, track = str(node).split(":")
            mapping[view] = int(track)
        groups.append(mapping)
    return groups


def build_sample(
    frame_id: int,
    group_mapping: dict[str, int],
    frame_indices: dict[str, dict[int, dict[int, list[dict[str, Any]]]]],
    view_ids: list[str],
    extrinsics_by_view: dict[str, Any],
    origin_center: np.ndarray,
    origin_scale: float,
    min_confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tokens = np.zeros((COCO_NUM_JOINTS, len(view_ids), RAY_FEATURE_DIM), dtype=np.float32)
    view_mask = np.zeros((len(view_ids),), dtype=np.bool_)
    joint_view_mask = np.zeros((COCO_NUM_JOINTS, len(view_ids)), dtype=np.bool_)
    confidence = np.zeros((COCO_NUM_JOINTS,), dtype=np.float32)

    ray_origins: dict[int, list[np.ndarray]] = {j: [] for j in range(COCO_NUM_JOINTS)}
    ray_dirs: dict[int, list[np.ndarray]] = {j: [] for j in range(COCO_NUM_JOINTS)}
    ray_weights: dict[int, list[float]] = {j: [] for j in range(COCO_NUM_JOINTS)}

    for view_index, view_id in enumerate(view_ids):
        track_id = group_mapping[view_id]
        points = frame_indices[view_id].get(frame_id, {}).get(track_id, [])
        if not points:
            continue
        point_by_id = {int(p["id"]): p for p in points}
        view_meta = extrinsics_by_view[view_id]
        width, height = [float(v) for v in view_meta["intrinsics"].get("image_size", [1280, 720])]
        view_mask[view_index] = True
        for joint_id in range(COCO_NUM_JOINTS):
            point = point_by_id.get(joint_id)
            if not point:
                continue
            x = float(point.get("x", np.nan))
            y = float(point.get("y", np.nan))
            conf = float(point.get("confidence", 0.0))
            if not np.isfinite(x) or not np.isfinite(y) or conf <= 0.0:
                continue
            origin, direction = pixel_to_world_ray(x, y, view_meta)
            origin_norm = (origin - origin_center) / max(float(origin_scale), 1e-6)
            x_norm = (x / width) * 2.0 - 1.0
            y_norm = (y / height) * 2.0 - 1.0
            tokens[joint_id, view_index] = [
                float(origin_norm[0]),
                float(origin_norm[1]),
                float(origin_norm[2]),
                float(direction[0]),
                float(direction[1]),
                float(direction[2]),
                float(x_norm),
                float(y_norm),
                float(conf),
                view_index / max(len(view_ids) - 1, 1),
                joint_id / max(COCO_NUM_JOINTS - 1, 1),
            ]
            confidence[joint_id] = max(confidence[joint_id], np.float32(conf))
            if conf >= min_confidence:
                joint_view_mask[joint_id, view_index] = True
                ray_origins[joint_id].append(origin)
                ray_dirs[joint_id].append(direction)
                ray_weights[joint_id].append(conf * conf)

    triangulated_3d = np.zeros((COCO_NUM_JOINTS, 3), dtype=np.float32)
    triangulated_quality = np.zeros((COCO_NUM_JOINTS,), dtype=np.float32)
    triangulated_valid = np.zeros((COCO_NUM_JOINTS,), dtype=np.bool_)
    for joint_id in range(COCO_NUM_JOINTS):
        point, quality = triangulate_weighted_rays(ray_origins[joint_id], ray_dirs[joint_id], ray_weights[joint_id])
        triangulated_3d[joint_id] = point
        triangulated_quality[joint_id] = np.float32(quality)
        triangulated_valid[joint_id] = len(ray_origins[joint_id]) >= 2 and quality > 0.0

    return tokens, view_mask, joint_view_mask, confidence, triangulated_3d, triangulated_quality, triangulated_valid


def main() -> None:
    args = parse_args()
    if len(args.track_jsons) != len(args.view_ids):
        raise ValueError("track-jsons and view-ids must have the same length")
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    selected_payload = load_json(args.selected_hypothesis_json)
    groups = assignments_from_selected(selected_payload)
    extrinsics_payload = load_json(args.rough_extrinsics_json)
    extrinsics_by_view = extrinsics_payload["extrinsics_by_view"]
    track_payloads = {view_id: load_json(path) for view_id, path in zip(args.view_ids, args.track_jsons)}
    frame_indices = {view_id: track_frame_index(payload) for view_id, payload in track_payloads.items()}

    origins = np.vstack([camera_origin_world(extrinsics_by_view[view_id]) for view_id in args.view_ids])
    origin_center = origins.mean(axis=0).astype(np.float32)
    origin_scale = max(float(np.linalg.norm(origins - origin_center.reshape(1, 3), axis=1).max()), 1.0)

    frame_sets = []
    for group in groups:
        for view_id in args.view_ids:
            track_id = group[view_id]
            frame_sets.append({fid for fid, by_track in frame_indices[view_id].items() if track_id in by_track})
    common_frames = sorted(set.intersection(*frame_sets)) if frame_sets else []
    if args.max_frames is not None:
        common_frames = common_frames[: int(args.max_frames)]

    entries = []
    dummy_target = np.zeros((COCO_NUM_JOINTS, 3), dtype=np.float32)
    for frame_id in common_frames:
        for group_index, group in enumerate(groups):
            tokens, view_mask, joint_view_mask, confidence, tri, tri_quality, tri_valid = build_sample(
                frame_id=frame_id,
                group_mapping=group,
                frame_indices=frame_indices,
                view_ids=list(args.view_ids),
                extrinsics_by_view=extrinsics_by_view,
                origin_center=origin_center,
                origin_scale=origin_scale,
                min_confidence=float(args.min_confidence),
            )
            pelvis_anchor = 0.5 * (tri[11] + tri[12])
            pelvis_anchor_valid = bool(tri_valid[11] and tri_valid[12])
            pelvis_anchor_quality = np.float32(0.5 * (tri_quality[11] + tri_quality[12])) if pelvis_anchor_valid else np.float32(0.0)
            person_id = f"karate_person{group_index + 1:02d}"
            sample_id = f"harmony4d_karate004_frame_{frame_id:06d}_{person_id}"
            rel_path = Path("samples") / f"{sample_id}.npz"
            np.savez(
                output_dir / rel_path,
                ray_tokens=tokens,
                view_mask=view_mask,
                joint_view_mask=joint_view_mask,
                target_3d=dummy_target,
                target_3d_root_relative=dummy_target,
                target_confidence=confidence,
                triangulated_3d=tri.astype(np.float32),
                triangulated_3d_root_relative=root_relative(tri),
                triangulated_quality=tri_quality.astype(np.float32),
                triangulated_valid=tri_valid.astype(np.bool_),
                pelvis_anchor=pelvis_anchor.astype(np.float32),
                pelvis_anchor_quality=np.asarray(pelvis_anchor_quality, dtype=np.float32),
                pelvis_anchor_valid=np.asarray(pelvis_anchor_valid, dtype=np.bool_),
                ray_origin_center=origin_center.astype(np.float32),
                ray_origin_scale=np.asarray(origin_scale, dtype=np.float32),
            )
            entries.append({
                "id": sample_id,
                "path": str(output_dir / rel_path),
                "split": args.split,
                "sequence": args.sequence,
                "frame_id": int(frame_id),
                "person_id": person_id,
                "identity_id": group_index,
                "active_joint_views": int(joint_view_mask.sum()),
                "triangulated_valid_joints": int(tri_valid.sum()),
                "pelvis_anchor_valid": pelvis_anchor_valid,
                "group_mapping": group,
            })

    manifest = {
        "metadata": {
            "stage": "harmony4d_matched_stage_a_dataset",
            "selected_hypothesis_json": str(Path(args.selected_hypothesis_json).resolve()),
            "rough_extrinsics_json": str(Path(args.rough_extrinsics_json).resolve()),
            "view_ids": list(args.view_ids),
            "num_entries": len(entries),
            "num_frames": len(common_frames),
            "num_identities": len(groups),
            "origin_center": origin_center.tolist(),
            "origin_scale": origin_scale,
            "min_confidence": float(args.min_confidence),
        },
        "entries": entries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(output_dir / "manifest.json"), "num_entries": len(entries), "num_frames": len(common_frames)}, indent=2))


if __name__ == "__main__":
    main()
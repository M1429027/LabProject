"""Cross-view matcher orchestration logic."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Any

import numpy as np

from .appearance import score_appearance
from .pose_similarity import score_pose_similarity, score_visibility_similarity
from .temporal_consistency import score_motion_consistency, score_temporal_consistency


@dataclass
class TrackletDescriptor:
    """Compact per-track descriptor used for cross-view matching."""

    view_id: str
    track_id: int
    start_frame: int
    end_frame: int
    num_detections: int
    mean_tracking_score: float
    mean_keypoint_confidence: float
    mean_motion_energy: float
    motion_energy_by_frame: list[tuple[int, float]]
    people_by_frame: dict[int, dict[str, Any]]
    lifted_by_frame: dict[int, dict[str, Any]]
    mean_forward_vector: list[float] | None
    mean_frontality: float | None


def _mean_keypoint_confidence(person: dict[str, Any]) -> float:
    """Return the mean keypoint confidence for one detection."""

    keypoints = person.get("keypoints", [])
    if not keypoints:
        return 0.0
    confs = [float(point.get("confidence", 0.0)) for point in keypoints]
    return float(np.mean(confs)) if confs else 0.0


def _root_xy(person: dict[str, Any]) -> tuple[float, float]:
    """Estimate a robust body root in image space."""

    keypoints = {int(point["id"]): point for point in person.get("keypoints", [])}
    if 11 in keypoints and 12 in keypoints:
        return (
            (float(keypoints[11]["x"]) + float(keypoints[12]["x"])) * 0.5,
            (float(keypoints[11]["y"]) + float(keypoints[12]["y"])) * 0.5,
        )
    bbox = person.get("bbox")
    if bbox:
        return (
            (float(bbox["x1"]) + float(bbox["x2"])) * 0.5,
            (float(bbox["y1"]) + float(bbox["y2"])) * 0.5,
        )
    return (0.0, 0.0)


def _motion_energy(track_people: list[tuple[int, dict[str, Any]]]) -> list[tuple[int, float]]:
    """Compute simple root-motion energy over time."""

    if not track_people:
        return []
    energies: list[tuple[int, float]] = []
    last_xy: tuple[float, float] | None = None
    for frame_idx, person in track_people:
        root_xy = _root_xy(person)
        if last_xy is None:
            energies.append((frame_idx, 0.0))
        else:
            dist = ((root_xy[0] - last_xy[0]) ** 2 + (root_xy[1] - last_xy[1]) ** 2) ** 0.5
            bbox = person.get("bbox")
            scale = 100.0
            if bbox:
                scale = max(
                    float(bbox["x2"]) - float(bbox["x1"]),
                    float(bbox["y2"]) - float(bbox["y1"]),
                    50.0,
                )
            energies.append((frame_idx, float(dist / scale)))
        last_xy = root_xy
    return energies


def _cosine_similarity_abs(vec_a: list[float] | None, vec_b: list[float] | None) -> float:
    """Absolute cosine similarity to tolerate front/back ambiguity."""

    if vec_a is None or vec_b is None:
        return 0.0
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0
    cos = float(np.dot(a, b) / (norm_a * norm_b))
    return abs(cos)


def _overlap_frames(track_a: TrackletDescriptor, track_b: TrackletDescriptor) -> list[int]:
    """Return temporally overlapping frame ids between two tracklets."""

    common = sorted(set(track_a.people_by_frame) & set(track_b.people_by_frame))
    return common


def _overlap_pose_similarity(
    track_a: TrackletDescriptor,
    track_b: TrackletDescriptor,
    min_conf: float,
) -> float:
    """Average normalized 2D pose similarity over the overlapping time window."""

    overlap = _overlap_frames(track_a, track_b)
    if not overlap:
        return 0.0
    scores = []
    for frame_idx in overlap:
        scores.append(
            score_pose_similarity(
                track_a.people_by_frame[frame_idx],
                track_b.people_by_frame[frame_idx],
                min_conf=min_conf,
            )
        )
    return float(np.mean(scores)) if scores else 0.0


def _overlap_motion_consistency(track_a: TrackletDescriptor, track_b: TrackletDescriptor) -> float:
    """Average motion-energy consistency on overlapping frames."""

    overlap = _overlap_frames(track_a, track_b)
    if not overlap:
        return 0.0
    energy_map_a = {frame_idx: energy for frame_idx, energy in track_a.motion_energy_by_frame}
    energy_map_b = {frame_idx: energy for frame_idx, energy in track_b.motion_energy_by_frame}
    seq_a = [energy_map_a[frame_idx] for frame_idx in overlap if frame_idx in energy_map_a and frame_idx in energy_map_b]
    seq_b = [energy_map_b[frame_idx] for frame_idx in overlap if frame_idx in energy_map_a and frame_idx in energy_map_b]
    return score_motion_consistency(seq_a, seq_b)


def _overlap_visibility_similarity(
    track_a: TrackletDescriptor,
    track_b: TrackletDescriptor,
    min_conf: float,
) -> float:
    """Average joint visibility-pattern similarity on overlapping frames."""

    overlap = _overlap_frames(track_a, track_b)
    if not overlap:
        return 0.0
    scores = []
    for frame_idx in overlap:
        scores.append(
            score_visibility_similarity(
                track_a.people_by_frame[frame_idx],
                track_b.people_by_frame[frame_idx],
                min_conf=min_conf,
            )
        )
    return float(np.mean(scores)) if scores else 0.0


def build_tracklet_descriptors(
    track_payload: dict[str, Any],
    lifted_payload: dict[str, Any] | None = None,
) -> list[TrackletDescriptor]:
    """Aggregate one view into per-track cross-view descriptors."""

    view_id = str(track_payload.get("metadata", {}).get("view_id", "unknown_view"))
    people_by_track: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for frame in track_payload.get("frames", []):
        frame_idx = int(frame["frame"])
        for person in frame.get("people", []):
            track_id = int(person["track_id"])
            people_by_track.setdefault(track_id, []).append((frame_idx, person))

    lifted_track_descriptors = {}
    lifted_by_track_frame: dict[int, dict[int, dict[str, Any]]] = {}
    if lifted_payload:
        for item in lifted_payload.get("track_descriptors", []):
            lifted_track_descriptors[int(item["track_id"])] = item
        for frame in lifted_payload.get("frames", []):
            frame_idx = int(frame["frame"])
            for person in frame.get("people", []):
                track_id = int(person["track_id"])
                lifted_by_track_frame.setdefault(track_id, {})[frame_idx] = person

    descriptors = []
    for track_summary in track_payload.get("tracks", []):
        track_id = int(track_summary["track_id"])
        detections = sorted(people_by_track.get(track_id, []), key=lambda item: item[0])
        if not detections:
            continue
        tracking_scores = [float(person.get("tracking_score", 0.0)) for _, person in detections]
        keypoint_scores = [_mean_keypoint_confidence(person) for _, person in detections]
        motion_energies = _motion_energy(detections)
        lifted_descriptor = lifted_track_descriptors.get(track_id)
        descriptors.append(
            TrackletDescriptor(
                view_id=view_id,
                track_id=track_id,
                start_frame=int(track_summary["start_frame"]),
                end_frame=int(track_summary["end_frame"]),
                num_detections=int(track_summary["num_detections"]),
                mean_tracking_score=float(np.mean(tracking_scores)) if tracking_scores else 0.0,
                mean_keypoint_confidence=float(np.mean(keypoint_scores)) if keypoint_scores else 0.0,
                mean_motion_energy=float(np.mean([energy for _, energy in motion_energies])) if motion_energies else 0.0,
                motion_energy_by_frame=motion_energies,
                people_by_frame={frame_idx: person for frame_idx, person in detections},
                lifted_by_frame=lifted_by_track_frame.get(track_id, {}),
                mean_forward_vector=(
                    list(lifted_descriptor.get("mean_forward_vector"))
                    if lifted_descriptor and lifted_descriptor.get("mean_forward_vector") is not None
                    else None
                ),
                mean_frontality=(
                    float(lifted_descriptor.get("mean_frontality"))
                    if lifted_descriptor and lifted_descriptor.get("mean_frontality") is not None
                    else None
                ),
            )
        )
    return descriptors


class CrossViewMatcher:
    """Combine multiple similarity terms into a final association decision."""

    def __init__(self, weights: dict[str, float], min_keypoint_confidence: float = 0.1) -> None:
        self.weights = {
            "appearance": float(weights.get("appearance", 0.15)),
            "pose_shape": float(weights.get("pose_shape", 0.35)),
            "temporal": float(weights.get("temporal", 0.0)),
            "motion_prior": float(weights.get("motion_prior", 0.20)),
            "visibility": float(weights.get("visibility", 0.20)),
            "skeleton_consistency": float(weights.get("skeleton_consistency", 0.05)),
        }
        self.min_keypoint_confidence = float(min_keypoint_confidence)

    def score_pair(self, track_a: TrackletDescriptor, track_b: TrackletDescriptor) -> dict[str, Any]:
        """Score one cross-view candidate pair."""

        temporal = score_temporal_consistency(
            {
                "start_frame": track_a.start_frame,
                "end_frame": track_a.end_frame,
            },
            {
                "start_frame": track_b.start_frame,
                "end_frame": track_b.end_frame,
            },
        )
        pose_shape = _overlap_pose_similarity(track_a, track_b, self.min_keypoint_confidence)
        motion_prior = _overlap_motion_consistency(track_a, track_b)
        visibility = _overlap_visibility_similarity(track_a, track_b, self.min_keypoint_confidence)
        skeleton_consistency = _cosine_similarity_abs(track_a.mean_forward_vector, track_b.mean_forward_vector)
        appearance = score_appearance(None, None)

        raw_scores = {
            "appearance": appearance,
            "pose_shape": pose_shape,
            "temporal": temporal,
            "motion_prior": motion_prior,
            "visibility": visibility,
            "skeleton_consistency": skeleton_consistency,
        }
        weighted_sum = 0.0
        weight_sum = 0.0
        for key, weight in self.weights.items():
            weighted_sum += raw_scores[key] * weight
            weight_sum += weight
        final_score = float((weighted_sum / max(weight_sum, 1e-6)) * temporal)

        overlap_frames = _overlap_frames(track_a, track_b)
        return {
            "view_a": track_a.view_id,
            "track_a": track_a.track_id,
            "view_b": track_b.view_id,
            "track_b": track_b.track_id,
            "final_score": final_score,
            "scores": raw_scores,
            "overlap_frames": {
                "count": len(overlap_frames),
                "start": int(overlap_frames[0]) if overlap_frames else None,
                "end": int(overlap_frames[-1]) if overlap_frames else None,
            },
            "summary": {
                "track_a_len": track_a.num_detections,
                "track_b_len": track_b.num_detections,
                "track_a_frontality": track_a.mean_frontality,
                "track_b_frontality": track_b.mean_frontality,
                "temporal_gate": temporal,
            },
        }

    @staticmethod
    def build_matching_graph(pair_results: dict[str, Any]) -> dict[str, Any]:
        """Build mutual-best graph edges from pairwise candidate rankings."""

        edges = []
        for pair_key, payload in pair_results.items():
            best_by_track_a = payload.get("best_by_track_a", {})
            best_by_track_b = payload.get("best_by_track_b", {})
            for track_a, candidate in best_by_track_a.items():
                track_b = str(candidate["track_b"])
                reverse_best = best_by_track_b.get(track_b)
                if reverse_best is None:
                    continue
                is_mutual = int(reverse_best["track_a"]) == int(track_a)
                edges.append(
                    {
                        "pair_key": pair_key,
                        "view_a": candidate["view_a"],
                        "track_a": candidate["track_a"],
                        "node_a": f"{candidate['view_a']}:{candidate['track_a']}",
                        "view_b": candidate["view_b"],
                        "track_b": candidate["track_b"],
                        "node_b": f"{candidate['view_b']}:{candidate['track_b']}",
                        "final_score": candidate["final_score"],
                        "is_mutual_best": is_mutual,
                        "scores": candidate["scores"],
                    }
                )

        edges.sort(key=lambda item: item["final_score"], reverse=True)
        parent: dict[str, str] = {}

        def find(node: str) -> str:
            parent.setdefault(node, node)
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: str, right: str) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        def group_views(node: str) -> set[str]:
            root = find(node)
            views = set()
            for candidate_node in parent:
                if find(candidate_node) == root:
                    views.add(candidate_node.split(":", 1)[0])
            return views

        for edge in edges:
            edge["used_for_group"] = False
            if not edge["is_mutual_best"]:
                continue
            views_a = group_views(edge["node_a"])
            views_b = group_views(edge["node_b"])
            if views_a & views_b:
                edge["blocked_reason"] = "source_view_conflict"
                continue
            union(edge["node_a"], edge["node_b"])
            edge["used_for_group"] = True

        groups_by_root: dict[str, list[str]] = {}
        for edge in edges:
            for node in (edge["node_a"], edge["node_b"]):
                groups_by_root.setdefault(find(node), []).append(node)

        groups = []
        for group_id, nodes in enumerate(groups_by_root.values()):
            unique_nodes = sorted(set(nodes))
            groups.append(
                {
                    "group_id": group_id,
                    "nodes": unique_nodes,
                    "num_views": len({node.split(":", 1)[0] for node in unique_nodes}),
                }
            )

        return {
            "edges": edges,
            "groups": groups,
        }

    @staticmethod
    def build_global_assignment_hypotheses(
        pair_results: dict[str, Any],
        descriptors_by_view: dict[str, list[TrackletDescriptor]],
        max_hypotheses: int = 16,
    ) -> dict[str, Any]:
        """Enumerate two-person global identity hypotheses across all views.

        Pairwise matching can collapse into per-view near/far ordering. For the
        karate two-player case, we keep explicit global assignment hypotheses so
        review and later geometry/self-calibration can choose between them.
        """

        views = sorted(descriptors_by_view)
        track_ids_by_view = {
            view_id: sorted(int(track.track_id) for track in descriptors_by_view[view_id])
            for view_id in views
        }
        unsupported = {
            view_id: track_ids
            for view_id, track_ids in track_ids_by_view.items()
            if len(track_ids) != 2
        }
        if unsupported:
            return {
                "status": "unsupported",
                "reason": "global_assignment_hypotheses currently expects exactly two tracks per view",
                "track_ids_by_view": track_ids_by_view,
                "hypotheses": [],
            }

        def candidate_score(view_a: str, track_a: int, view_b: str, track_b: int) -> float | None:
            pair_key = f"{view_a}__{view_b}"
            reverse = False
            if pair_key not in pair_results:
                pair_key = f"{view_b}__{view_a}"
                reverse = True
            payload = pair_results.get(pair_key)
            if payload is None:
                return None
            for candidate in payload.get("candidates", []):
                if not reverse:
                    is_match = int(candidate["track_a"]) == int(track_a) and int(candidate["track_b"]) == int(track_b)
                else:
                    is_match = int(candidate["track_a"]) == int(track_b) and int(candidate["track_b"]) == int(track_a)
                if is_match:
                    return float(candidate["final_score"])
            return None

        hypotheses = []
        anchor_view = views[0] if views else None
        flip_options = list(product([False, True], repeat=max(len(views) - 1, 0)))
        for hypothesis_idx, flips_tail in enumerate(flip_options):
            flips = {anchor_view: False} if anchor_view is not None else {}
            flips.update({view_id: flip for view_id, flip in zip(views[1:], flips_tail)})

            assignments: dict[str, dict[str, int]] = {}
            groups = [
                {"group_id": 0, "nodes": [], "num_views": len(views)},
                {"group_id": 1, "nodes": [], "num_views": len(views)},
            ]
            for view_id in views:
                local_tracks = track_ids_by_view[view_id]
                if flips[view_id]:
                    global_tracks = {0: local_tracks[1], 1: local_tracks[0]}
                else:
                    global_tracks = {0: local_tracks[0], 1: local_tracks[1]}
                assignments[view_id] = {
                    "identity_0": global_tracks[0],
                    "identity_1": global_tracks[1],
                    "flipped_from_local_order": bool(flips[view_id]),
                }
                groups[0]["nodes"].append(f"{view_id}:{global_tracks[0]}")
                groups[1]["nodes"].append(f"{view_id}:{global_tracks[1]}")

            pair_scores = []
            pair_details = []
            for view_a, view_b in combinations(views, 2):
                for identity_id in (0, 1):
                    track_a = assignments[view_a][f"identity_{identity_id}"]
                    track_b = assignments[view_b][f"identity_{identity_id}"]
                    score = candidate_score(view_a, track_a, view_b, track_b)
                    if score is None:
                        continue
                    pair_scores.append(score)
                    pair_details.append(
                        {
                            "identity_id": identity_id,
                            "view_a": view_a,
                            "track_a": track_a,
                            "view_b": view_b,
                            "track_b": track_b,
                            "score": score,
                        }
                    )

            mean_score = float(np.mean(pair_scores)) if pair_scores else 0.0
            min_score = float(np.min(pair_scores)) if pair_scores else 0.0
            hypotheses.append(
                {
                    "hypothesis_id": hypothesis_idx,
                    "anchor_view": anchor_view,
                    "assignments": assignments,
                    "groups": groups,
                    "mean_pair_score": mean_score,
                    "min_pair_score": min_score,
                    "num_pair_scores": len(pair_scores),
                    "pair_details": pair_details,
                }
            )

        hypotheses.sort(key=lambda item: (item["mean_pair_score"], item["min_pair_score"]), reverse=True)
        for rank, hypothesis in enumerate(hypotheses):
            hypothesis["rank"] = rank
        return {
            "status": "ok",
            "note": "These are two-person global identity hypotheses; high pairwise score alone may still prefer near/far ordering before geometry is added.",
            "track_ids_by_view": track_ids_by_view,
            "hypotheses": hypotheses[:max_hypotheses],
        }

    def match(self, descriptors_by_view: dict[str, list[TrackletDescriptor]]) -> dict[str, Any]:
        """Produce pairwise cross-view candidate rankings."""

        pair_results = {}
        for view_a, view_b in combinations(sorted(descriptors_by_view), 2):
            candidates = []
            descriptors_a = descriptors_by_view[view_a]
            descriptors_b = descriptors_by_view[view_b]
            for track_a in descriptors_a:
                for track_b in descriptors_b:
                    candidates.append(self.score_pair(track_a, track_b))
            candidates.sort(key=lambda item: item["final_score"], reverse=True)

            best_by_track_a = {}
            best_by_track_b = {}
            for candidate in candidates:
                best_by_track_a.setdefault(str(candidate["track_a"]), candidate)
                best_by_track_b.setdefault(str(candidate["track_b"]), candidate)

            pair_results[f"{view_a}__{view_b}"] = {
                "view_a": view_a,
                "view_b": view_b,
                "num_tracks_a": len(descriptors_a),
                "num_tracks_b": len(descriptors_b),
                "candidates": candidates,
                "best_by_track_a": best_by_track_a,
                "best_by_track_b": best_by_track_b,
            }
        return {
            "pair_results": pair_results,
            "matching_graph": self.build_matching_graph(pair_results),
            "global_assignment_hypotheses": self.build_global_assignment_hypotheses(
                pair_results,
                descriptors_by_view,
            ),
        }

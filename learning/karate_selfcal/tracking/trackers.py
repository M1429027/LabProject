"""Tracking backend adapters and shared interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


class TrackerBackend:
    """Minimal interface for a single-view tracker backend."""

    def update(self, frame_index: int, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raise NotImplementedError


@dataclass
class TrackState:
    """State for a single active track."""

    track_id: int
    last_frame: int
    bbox: dict[str, float] | None
    keypoints: list[dict[str, Any]]
    hits: int = 1
    missed: int = 0
    history_frames: list[int] = field(default_factory=list)
    history_bboxes: list[dict[str, float] | None] = field(default_factory=list)
    history_keypoints: list[list[dict[str, Any]]] = field(default_factory=list)


def _bbox_center(bbox: dict[str, float] | None) -> tuple[float, float] | None:
    """Return bbox center or None when bbox is missing."""

    if not bbox:
        return None
    return ((bbox["x1"] + bbox["x2"]) * 0.5, (bbox["y1"] + bbox["y2"]) * 0.5)


def _bbox_area(bbox: dict[str, float] | None) -> float:
    """Return bbox area or zero when bbox is missing."""

    if not bbox:
        return 0.0
    w = max(0.0, bbox["x2"] - bbox["x1"])
    h = max(0.0, bbox["y2"] - bbox["y1"])
    return w * h


def _shift_bbox(
    bbox: dict[str, float] | None,
    dx: float,
    dy: float,
) -> dict[str, float] | None:
    """Translate a bbox by the provided offsets."""

    if not bbox:
        return None
    return {
        "x1": float(bbox["x1"]) + dx,
        "y1": float(bbox["y1"]) + dy,
        "x2": float(bbox["x2"]) + dx,
        "y2": float(bbox["y2"]) + dy,
        "confidence": float(bbox.get("confidence", 0.0)),
    }


def _bbox_iou(a: dict[str, float] | None, b: dict[str, float] | None) -> float:
    """Compute IoU between two xyxy bboxes."""

    if not a or not b:
        return 0.0

    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    union = _bbox_area(a) + _bbox_area(b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _keypoint_dict(points: list[dict[str, Any]], min_conf: float) -> dict[int, tuple[float, float]]:
    """Convert a keypoint list into an id-indexed dict."""

    out = {}
    for point in points:
        if float(point.get("confidence", 0.0)) < min_conf:
            continue
        out[int(point["id"])] = (float(point["x"]), float(point["y"]))
    return out


def _pose_similarity(
    a_points: list[dict[str, Any]],
    b_points: list[dict[str, Any]],
    min_conf: float,
    normalization_scale: float,
) -> float:
    """Compute a bounded similarity from confident keypoint distances."""

    a_map = _keypoint_dict(a_points, min_conf)
    b_map = _keypoint_dict(b_points, min_conf)
    common = sorted(set(a_map) & set(b_map))
    if not common:
        return 0.0

    distances = []
    for key in common:
        ax, ay = a_map[key]
        bx, by = b_map[key]
        distances.append(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)

    mean_dist = float(np.mean(distances))
    norm = max(normalization_scale, 1.0)
    score = 1.0 - min(mean_dist / norm, 1.0)
    return max(0.0, score)


def _center_similarity(
    a_bbox: dict[str, float] | None,
    b_bbox: dict[str, float] | None,
    frame_width: int,
    frame_height: int,
) -> float:
    """Compute center proximity similarity within the frame."""

    a_center = _bbox_center(a_bbox)
    b_center = _bbox_center(b_bbox)
    if a_center is None or b_center is None:
        return 0.0

    diag = max((frame_width**2 + frame_height**2) ** 0.5, 1.0)
    dist = ((a_center[0] - b_center[0]) ** 2 + (a_center[1] - b_center[1]) ** 2) ** 0.5
    return max(0.0, 1.0 - min(dist / diag, 1.0))


def _track_velocity(track: TrackState) -> tuple[float, float]:
    """Estimate average per-frame center velocity from recent history."""

    centers: list[tuple[int, tuple[float, float]]] = []
    for frame_idx, bbox in zip(track.history_frames, track.history_bboxes):
        center = _bbox_center(bbox)
        if center is None:
            continue
        centers.append((int(frame_idx), center))

    if len(centers) < 2:
        return 0.0, 0.0

    velocities = []
    for (prev_frame, prev_center), (curr_frame, curr_center) in zip(centers[:-1], centers[1:]):
        dt = max(curr_frame - prev_frame, 1)
        velocities.append(
            (
                (curr_center[0] - prev_center[0]) / dt,
                (curr_center[1] - prev_center[1]) / dt,
            )
        )
    if not velocities:
        return 0.0, 0.0
    vx = float(np.mean([v[0] for v in velocities]))
    vy = float(np.mean([v[1] for v in velocities]))
    return vx, vy


def _predict_bbox(track: TrackState, frame_index: int) -> dict[str, float] | None:
    """Predict the bbox position for a future frame using recent velocity."""

    if not track.bbox:
        return None
    gap = max(int(frame_index) - int(track.last_frame), 0)
    if gap <= 0:
        return track.bbox
    vx, vy = _track_velocity(track)
    return _shift_bbox(track.bbox, vx * gap, vy * gap)


def _prototype_keypoints(track: TrackState) -> list[dict[str, Any]]:
    """Build a simple prototype pose from recent history."""

    aggregates: dict[int, list[tuple[float, float, float]]] = {}
    for keypoints in track.history_keypoints:
        for point in keypoints:
            confidence = float(point.get("confidence", 0.0))
            if confidence <= 0.0:
                continue
            point_id = int(point["id"])
            aggregates.setdefault(point_id, []).append(
                (float(point["x"]), float(point["y"]), confidence)
            )

    if not aggregates:
        return track.keypoints

    prototype = []
    for point_id in sorted(aggregates):
        xs = [item[0] for item in aggregates[point_id]]
        ys = [item[1] for item in aggregates[point_id]]
        cs = [item[2] for item in aggregates[point_id]]
        prototype.append(
            {
                "id": point_id,
                "x": float(np.mean(xs)),
                "y": float(np.mean(ys)),
                "confidence": float(np.mean(cs)),
            }
        )
    return prototype


class PoseAwareSingleViewTracker(TrackerBackend):
    """Simple pose-aware tracker for single-view sports clips."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        match_threshold: float = 0.45,
        lost_track_buffer: int = 30,
        min_keypoint_confidence: float = 0.1,
        iou_weight: float = 0.45,
        pose_weight: float = 0.4,
        center_weight: float = 0.15,
        history_size: int = 8,
    ) -> None:
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.match_threshold = float(match_threshold)
        self.lost_track_buffer = int(lost_track_buffer)
        self.min_keypoint_confidence = float(min_keypoint_confidence)
        self.iou_weight = float(iou_weight)
        self.pose_weight = float(pose_weight)
        self.center_weight = float(center_weight)
        self.history_size = max(int(history_size), 2)
        self._next_track_id = 0
        self._tracks: list[TrackState] = []

    def _normalization_scale(
        self,
        track_bbox: dict[str, float] | None,
        det_bbox: dict[str, float] | None,
    ) -> float:
        """Choose a pose-distance normalization scale from bbox sizes."""

        scale_a = max(_bbox_area(track_bbox), 1.0) ** 0.5
        scale_b = max(_bbox_area(det_bbox), 1.0) ** 0.5
        return max((scale_a + scale_b) * 0.5, 50.0)

    def _append_history(
        self,
        track: TrackState,
        frame_index: int,
        bbox: dict[str, float] | None,
        keypoints: list[dict[str, Any]],
    ) -> None:
        """Append a new observation into the track memory."""

        track.history_frames.append(int(frame_index))
        track.history_bboxes.append(bbox)
        track.history_keypoints.append(keypoints)
        if len(track.history_frames) > self.history_size:
            track.history_frames = track.history_frames[-self.history_size :]
            track.history_bboxes = track.history_bboxes[-self.history_size :]
            track.history_keypoints = track.history_keypoints[-self.history_size :]

    def _match_score(self, track: TrackState, detection: dict[str, Any], frame_index: int) -> float:
        """Compute overall association score between a track and detection."""

        det_bbox = detection.get("bbox")
        reference_bbox = _predict_bbox(track, frame_index)
        iou = _bbox_iou(reference_bbox, det_bbox)
        pose = _pose_similarity(
            _prototype_keypoints(track),
            detection.get("keypoints", []),
            min_conf=self.min_keypoint_confidence,
            normalization_scale=self._normalization_scale(reference_bbox, det_bbox),
        )
        center = _center_similarity(reference_bbox, det_bbox, self.frame_width, self.frame_height)
        return self.iou_weight * iou + self.pose_weight * pose + self.center_weight * center

    def _create_track(self, frame_index: int, detection: dict[str, Any]) -> TrackState:
        """Create a new track from a detection."""

        track = TrackState(
            track_id=self._next_track_id,
            last_frame=frame_index,
            bbox=detection.get("bbox"),
            keypoints=detection.get("keypoints", []),
        )
        self._append_history(
            track,
            frame_index=frame_index,
            bbox=detection.get("bbox"),
            keypoints=detection.get("keypoints", []),
        )
        self._next_track_id += 1
        self._tracks.append(track)
        return track

    def update(self, frame_index: int, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Assign persistent track ids to detections on the current frame."""

        active_tracks = [
            track for track in self._tracks if (frame_index - track.last_frame) <= self.lost_track_buffer
        ]

        if not active_tracks:
            outputs = []
            for detection in detections:
                track = self._create_track(frame_index, detection)
                enriched = dict(detection)
                enriched["track_id"] = track.track_id
                enriched["tracking_score"] = 1.0
                outputs.append(enriched)
            return outputs

        if not detections:
            for track in active_tracks:
                track.missed += 1
            return []

        score_matrix = np.zeros((len(active_tracks), len(detections)), dtype=np.float32)
        for i, track in enumerate(active_tracks):
            for j, detection in enumerate(detections):
                score_matrix[i, j] = self._match_score(track, detection, frame_index)

        cost_matrix = 1.0 - score_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_detections = set()
        outputs: list[dict[str, Any]] = []

        for row, col in zip(row_ind.tolist(), col_ind.tolist()):
            score = float(score_matrix[row, col])
            if score < self.match_threshold:
                continue

            track = active_tracks[row]
            detection = detections[col]
            track.last_frame = frame_index
            track.bbox = detection.get("bbox")
            track.keypoints = detection.get("keypoints", [])
            track.hits += 1
            track.missed = 0
            self._append_history(
                track,
                frame_index=frame_index,
                bbox=detection.get("bbox"),
                keypoints=detection.get("keypoints", []),
            )
            matched_tracks.add(track.track_id)
            matched_detections.add(col)

            enriched = dict(detection)
            enriched["track_id"] = track.track_id
            enriched["tracking_score"] = score
            outputs.append(enriched)

        for track in active_tracks:
            if track.track_id not in matched_tracks:
                track.missed += 1

        for det_idx, detection in enumerate(detections):
            if det_idx in matched_detections:
                continue
            track = self._create_track(frame_index, detection)
            enriched = dict(detection)
            enriched["track_id"] = track.track_id
            enriched["tracking_score"] = 1.0
            outputs.append(enriched)

        outputs.sort(key=lambda item: int(item["track_id"]))
        return outputs


def build_tracker(config: dict[str, Any], frame_width: int, frame_height: int) -> TrackerBackend:
    """Construct the configured single-view tracker."""

    tracker_cfg = config.get("tracker", {})
    backend = str(tracker_cfg.get("backend", "pose_single_view")).lower()

    if backend in {"pose_single_view", "pose_tracker", "single_view_pose_tracker", "bytetrack"}:
            return PoseAwareSingleViewTracker(
                frame_width=frame_width,
                frame_height=frame_height,
                match_threshold=float(tracker_cfg.get("match_threshold", 0.45)),
                lost_track_buffer=int(tracker_cfg.get("lost_track_buffer", 30)),
                min_keypoint_confidence=float(tracker_cfg.get("min_keypoint_confidence", 0.1)),
                iou_weight=float(tracker_cfg.get("iou_weight", 0.45)),
                pose_weight=float(tracker_cfg.get("pose_weight", 0.4)),
                center_weight=float(tracker_cfg.get("center_weight", 0.15)),
                history_size=int(tracker_cfg.get("history_size", 8)),
            )

    raise ValueError(f"Unsupported tracker backend: {backend}")

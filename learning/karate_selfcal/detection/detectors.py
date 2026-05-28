"""Detector backends and data adapters for 2D pose estimation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

KEYPOINT_NAMES = [
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


@dataclass
class DetectionResult:
    """Normalized per-frame pose output."""

    people: list[dict[str, Any]]
    annotated_frame: np.ndarray | None = None


class DetectorBackend:
    """Minimal interface for a detector backend."""

    supports_heatmaps = False

    def run(self, frame: np.ndarray) -> DetectionResult:
        raise NotImplementedError


def _safe_numpy(value: Any) -> np.ndarray | None:
    """Best-effort conversion from model outputs to numpy arrays."""

    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    try:
        return np.asarray(value)
    except Exception:
        return None


def _normalize_bbox_xyxy(bbox: np.ndarray, confidence: float | None = None) -> dict[str, float]:
    """Convert a raw xyxy bbox to a JSON-serializable mapping."""

    x1, y1, x2, y2 = [float(v) for v in bbox.tolist()]
    payload = {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }
    if confidence is not None:
        payload["confidence"] = float(confidence)
    return payload


class YOLOPoseDetector(DetectorBackend):
    """Run Ultralytics YOLO pose inference on a single frame."""

    def __init__(
        self,
        model_path: str | Path,
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.5,
        max_people: int = 4,
        device: str | None = None,
        draw_annotations: bool = False,
    ) -> None:
        self.model_path = str(model_path)
        self.conf_thresh = float(conf_thresh)
        self.iou_thresh = float(iou_thresh)
        self.max_people = int(max_people)
        self.device = device
        self.draw_annotations = draw_annotations
        self.model = YOLO(self.model_path)

    def run(self, frame: np.ndarray) -> DetectionResult:
        infer_kwargs: dict[str, Any] = {
            "verbose": False,
            "conf": self.conf_thresh,
            "iou": self.iou_thresh,
            "max_det": self.max_people,
        }
        if self.device and self.device != "auto":
            infer_kwargs["device"] = self.device

        results = self.model(frame, **infer_kwargs)
        if not results:
            annotated = frame.copy() if self.draw_annotations else None
            return DetectionResult(people=[], annotated_frame=annotated)

        result = results[0]
        people: list[dict[str, Any]] = []

        keypoints_xy = None
        keypoints_conf = None
        if getattr(result, "keypoints", None) is not None:
            if result.keypoints.xy is not None:
                keypoints_xy = result.keypoints.xy.cpu().numpy()
            if result.keypoints.conf is not None:
                keypoints_conf = result.keypoints.conf.cpu().numpy()

        boxes_xyxy = None
        box_conf = None
        if getattr(result, "boxes", None) is not None and result.boxes.xyxy is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            if result.boxes.conf is not None:
                box_conf = result.boxes.conf.cpu().numpy()

        if keypoints_xy is None or len(keypoints_xy) == 0:
            annotated = result.plot() if self.draw_annotations else None
            return DetectionResult(people=[], annotated_frame=annotated)

        person_count = min(len(keypoints_xy), self.max_people)
        for person_id in range(person_count):
            keypoints = []
            for joint_id in range(min(len(KEYPOINT_NAMES), keypoints_xy[person_id].shape[0])):
                x = float(keypoints_xy[person_id][joint_id][0])
                y = float(keypoints_xy[person_id][joint_id][1])
                confidence = 0.0
                if keypoints_conf is not None:
                    confidence = float(keypoints_conf[person_id][joint_id])

                keypoints.append(
                    {
                        "id": joint_id,
                        "x": x,
                        "y": y,
                        "confidence": confidence,
                    }
                )

            person: dict[str, Any] = {
                "person_id": person_id,
                "keypoints": keypoints,
            }
            if boxes_xyxy is not None and person_id < len(boxes_xyxy):
                confidence = None
                if box_conf is not None and person_id < len(box_conf):
                    confidence = float(box_conf[person_id])
                person["bbox"] = _normalize_bbox_xyxy(boxes_xyxy[person_id], confidence)

            people.append(person)

        annotated = result.plot() if self.draw_annotations else None
        if annotated is not None and annotated.dtype != np.uint8:
            annotated = annotated.astype(np.uint8)

        return DetectionResult(people=people, annotated_frame=annotated)


class YOLOHRNetTopDownDetector(DetectorBackend):
    """Run YOLO person detection followed by HRNet top-down pose estimation."""

    supports_heatmaps = True

    def __init__(
        self,
        detector_model_path: str | Path,
        pose_config_path: str | Path,
        pose_checkpoint_path: str | Path,
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.5,
        max_people: int = 4,
        device: str | None = None,
        draw_annotations: bool = False,
        return_heatmaps: bool = False,
    ) -> None:
        self.detector_model_path = str(detector_model_path)
        self.pose_config_path = str(pose_config_path)
        self.pose_checkpoint_path = str(pose_checkpoint_path)
        self.conf_thresh = float(conf_thresh)
        self.iou_thresh = float(iou_thresh)
        self.max_people = int(max_people)
        self.device = device
        self.draw_annotations = draw_annotations
        self.return_heatmaps = bool(return_heatmaps)
        self.detector = YOLO(self.detector_model_path)

        try:
            from mmpose.apis import inference_topdown, init_model
        except Exception as exc:
            raise RuntimeError(
                "YOLO->HRNet top-down backend requires MMPose to be installed in "
                "the active environment. Install mmpose/mmengine/mmcv and provide "
                "a valid HRNet config/checkpoint before using this backend."
            ) from exc

        self._inference_topdown = inference_topdown
        init_kwargs: dict[str, Any] = {"device": self._resolve_mmpose_device()}
        self.pose_model = init_model(
            self.pose_config_path,
            self.pose_checkpoint_path,
            **init_kwargs,
        )
        if self.return_heatmaps:
            self.pose_model.test_cfg["output_heatmaps"] = True

    def _resolve_mmpose_device(self) -> str:
        """Map the generic device flag to an MMPose-compatible string."""

        if not self.device or self.device == "auto":
            return "cuda:0"
        if self.device == "cpu":
            return "cpu"
        if self.device.startswith("cuda"):
            return self.device
        if self.device.isdigit():
            return f"cuda:{self.device}"
        return self.device

    def _detect_person_boxes(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Use YOLO detection to extract person bboxes for top-down pose."""

        infer_kwargs: dict[str, Any] = {
            "verbose": False,
            "conf": self.conf_thresh,
            "iou": self.iou_thresh,
            "max_det": self.max_people,
            "classes": [0],
        }
        if self.device and self.device != "auto":
            infer_kwargs["device"] = self.device

        results = self.detector(frame, **infer_kwargs)
        if not results:
            return []

        result = results[0]
        if getattr(result, "boxes", None) is None or result.boxes.xyxy is None:
            return []

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = None
        if result.boxes.conf is not None:
            confs = result.boxes.conf.cpu().numpy()

        boxes = []
        for idx in range(min(len(boxes_xyxy), self.max_people)):
            score = float(confs[idx]) if confs is not None and idx < len(confs) else None
            boxes.append(
                {
                    "bbox": boxes_xyxy[idx],
                    "bbox_score": score if score is not None else 1.0,
                    "person_id": idx,
                }
            )
        return boxes

    def _extract_heatmaps(self, sample: Any) -> np.ndarray | None:
        """Best-effort retrieval of per-person heatmaps from an MMPose sample."""

        pred_fields = getattr(sample, "pred_fields", None)
        if pred_fields is None:
            return None

        for field_name in ("heatmaps", "heatmap", "output_heatmap"):
            value = getattr(pred_fields, field_name, None)
            array = _safe_numpy(value)
            if array is not None:
                return array
        return None

    def _extract_people(self, samples: list[Any], boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert MMPose output samples into the normalized JSON structure."""

        people = []
        for person_id, sample in enumerate(samples[: self.max_people]):
            pred_instances = getattr(sample, "pred_instances", None)
            if pred_instances is None:
                continue

            keypoints_np = _safe_numpy(getattr(pred_instances, "keypoints", None))
            scores_np = _safe_numpy(getattr(pred_instances, "keypoint_scores", None))
            if keypoints_np is None:
                continue

            if keypoints_np.ndim == 3:
                keypoints_np = keypoints_np[0]
            if scores_np is not None and scores_np.ndim == 2:
                scores_np = scores_np[0]

            keypoints = []
            for joint_id in range(min(len(KEYPOINT_NAMES), keypoints_np.shape[0])):
                x = float(keypoints_np[joint_id][0])
                y = float(keypoints_np[joint_id][1])
                confidence = 0.0
                if scores_np is not None and joint_id < len(scores_np):
                    confidence = float(scores_np[joint_id])
                keypoints.append(
                    {
                        "id": joint_id,
                        "x": x,
                        "y": y,
                        "confidence": confidence,
                    }
                )

            person: dict[str, Any] = {
                "person_id": int(boxes[person_id].get("person_id", person_id))
                if person_id < len(boxes)
                else person_id,
                "keypoints": keypoints,
            }

            if person_id < len(boxes):
                bbox_xyxy = np.asarray(boxes[person_id]["bbox"])
                bbox_score = boxes[person_id].get("bbox_score")
                person["bbox"] = _normalize_bbox_xyxy(
                    bbox_xyxy,
                    float(bbox_score) if bbox_score is not None else None,
                )

            heatmaps = self._extract_heatmaps(sample)
            if heatmaps is not None:
                person["heatmaps"] = heatmaps

            people.append(person)

        return people

    def _draw_annotations(self, frame: np.ndarray, people: list[dict[str, Any]]) -> np.ndarray:
        """Draw lightweight bbox and keypoint overlays for debug videos."""

        annotated = frame.copy()
        for person in people:
            bbox = person.get("bbox")
            if bbox:
                x1 = int(bbox["x1"])
                y1 = int(bbox["y1"])
                x2 = int(bbox["x2"])
                y2 = int(bbox["y2"])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 255), 2)
                cv2.putText(
                    annotated,
                    f"id {person['person_id']}",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 180, 255),
                    2,
                    cv2.LINE_AA,
                )

            for keypoint in person.get("keypoints", []):
                if float(keypoint.get("confidence", 0.0)) <= 0.0:
                    continue
                x = int(keypoint["x"])
                y = int(keypoint["y"])
                cv2.circle(annotated, (x, y), 3, (0, 255, 0), -1)
        return annotated

    def run(self, frame: np.ndarray) -> DetectionResult:
        boxes = self._detect_person_boxes(frame)
        if not boxes:
            annotated = frame.copy() if self.draw_annotations else None
            return DetectionResult(people=[], annotated_frame=annotated)

        pose_inputs = np.asarray([box["bbox"] for box in boxes], dtype=np.float32)

        samples = self._inference_topdown(
            self.pose_model,
            frame,
            pose_inputs,
            bbox_format="xyxy",
        )

        people = self._extract_people(samples, boxes)
        annotated = self._draw_annotations(frame, people) if self.draw_annotations else None
        return DetectionResult(people=people, annotated_frame=annotated)


def build_detector(config: dict[str, Any], draw_annotations: bool = False) -> DetectorBackend:
    """Create a detector backend from a detection-stage config."""

    detector_cfg = config.get("detector", {})
    pose_cfg = config.get("pose", {})

    backend = str(detector_cfg.get("backend", "yolo_pose")).lower()
    model_path = detector_cfg.get("model_path") or config.get("model_path") or "yolo11l-pose.pt"
    conf_thresh = detector_cfg.get("confidence_threshold", 0.35)
    iou_thresh = detector_cfg.get("iou_threshold", 0.5)
    max_people = detector_cfg.get("max_people", 4)
    device = detector_cfg.get("device", "auto")

    if backend in {"yolo_pose", "yolov8_pose", "yolo11_pose"}:
        return YOLOPoseDetector(
            model_path=model_path,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            max_people=max_people,
            device=device,
            draw_annotations=draw_annotations,
        )

    if backend in {"yolo_hrnet_topdown", "yolo_topdown_hrnet", "topdown_hrnet"}:
        pose_backend = str(pose_cfg.get("backend", "hrnet")).lower()
        if pose_backend != "hrnet":
            raise ValueError(
                "The yolo_hrnet_topdown detector backend requires pose.backend=hrnet."
            )

        pose_config_path = pose_cfg.get("config_path")
        pose_checkpoint_path = pose_cfg.get("checkpoint_path")
        if not pose_config_path or not pose_checkpoint_path:
            raise ValueError(
                "HRNet top-down mode requires pose.config_path and "
                "pose.checkpoint_path to be set."
            )

        return YOLOHRNetTopDownDetector(
            detector_model_path=model_path,
            pose_config_path=pose_config_path,
            pose_checkpoint_path=pose_checkpoint_path,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            max_people=max_people,
            device=device,
            draw_annotations=draw_annotations,
            return_heatmaps=bool(pose_cfg.get("return_heatmaps", False)),
        )

    raise ValueError(f"Unsupported detector backend: {backend}")


def open_video_writer(
    output_path: str | Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    """Create a writer for annotated MP4 output."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")
    return writer

"""Build assets for the global floor-axis relabeling page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


CAMERAS = {
    "cam1": {
        "points": "point_picks_cam4_hvflip_retry/cam1out_line_points.json",
        "report": "extrinsics_all_hvflip_retry/cam1_line_extrinsics_report.json",
    },
    "cam2": {
        "points": "point_picks_cam4_hvflip_retry/cam2out_line_points.json",
        "report": "extrinsics_all_hvflip_retry/cam2_line_extrinsics_report.json",
    },
    "cam3": {
        "points": "point_picks_cam4_hvflip_retry/cam3out_line_points.json",
        "report": "extrinsics_all_hvflip_retry/cam3_line_extrinsics_report.json",
    },
    "cam4": {
        "points": "point_picks_cam4_hvflip_retry/cam4_line_points.json",
        "report": "extrinsics_cam4_hvflip_newintr_20260727/cam4_line_extrinsics_report.json",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--line-dir", default="camtest/line")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    line_dir = Path(args.line_dir)
    output_dir = Path(args.output_dir)
    assets = output_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    payload = {"spacing_m": 1.52, "cameras": {}}
    for cam_id, paths in CAMERAS.items():
        points_data = json.loads((line_dir / paths["points"]).read_text(encoding="utf-8"))
        report = json.loads((line_dir / paths["report"]).read_text(encoding="utf-8"))
        metadata = points_data["metadata"]
        video_path = Path(metadata["video_path"])
        if not video_path.is_absolute():
            video_path = line_dir / video_path.name
        frame_index = int(metadata["frame_index"])
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Cannot read {cam_id} frame {frame_index} from {video_path}")
        height, width = frame.shape[:2]
        target_width = min(width, 1200)
        if target_width != width:
            target_height = int(round(height * target_width / width))
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        image_name = f"{cam_id}.jpg"
        cv2.imwrite(str(assets / image_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 86])

        used = report["used_points"]
        markers = []
        for index, item in enumerate(used):
            xy = item["image_xy_px"]
            markers.append(
                {
                    "marker_id": f"M{index + 1:02d}",
                    "source_id": item.get("label", item.get("id", f"P{index:02d}")),
                    "current_world_label": item["world_label"],
                    "current_world_xyz_m": item["world_xyz_m"],
                    "image_xy_px": xy,
                    "x_ratio": float(xy[0]) / width,
                    "y_ratio": float(xy[1]) / height,
                }
            )
        payload["cameras"][cam_id] = {
            "image": f"assets/{image_name}",
            "image_size": [width, height],
            "frame_index": frame_index,
            "source_video": str(video_path),
            "markers": markers,
        }

    js = "window.GLOBAL_AXIS_PICKER_DATA = " + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ) + ";\n"
    (output_dir / "data.js").write_text(js, encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir.resolve()), "cameras": list(CAMERAS)}, indent=2))


if __name__ == "__main__":
    main()

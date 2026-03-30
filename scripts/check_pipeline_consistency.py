import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import yaml


DEFAULT_INTRIN_CAM1 = "camera_calibration/charuco/calib_charuco_v2/config_intrinsics_cam1.yaml"
DEFAULT_INTRIN_CAM2 = "camera_calibration/charuco/calib_charuco_v2/config_intrinsics_cam2.yaml"
DEFAULT_EXTRIN_CAM1 = "camera_calibration/charuco/calib_charuco_v2/config_extrinsics_cam1.yaml"
DEFAULT_EXTRIN_CAM2 = "camera_calibration/charuco/calib_charuco_v2/config_extrinsics_cam2.yaml"


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def unique_paths(paths):
    out = []
    seen = set()
    for p in paths:
        k = str(p.resolve()) if p.exists() else str(p)
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def candidate_paths(raw_path, cfg_dir):
    p = Path(raw_path)
    cands = []
    cands.append(p)
    cands.append(cfg_dir / p)
    raw = str(raw_path).replace("\\", "/")
    cfg_name = cfg_dir.name
    if raw.startswith(cfg_name + "/"):
        cands.append(cfg_dir.parent / p)
    if raw.startswith("./"):
        cands.append(cfg_dir / raw[2:])
    return unique_paths(cands)


def resolve_existing_path(raw_path, cfg_dir):
    cands = candidate_paths(raw_path, cfg_dir)
    for c in cands:
        if c.exists():
            return c.resolve(), cands
    return None, cands


def resolve_output_file(out_dir_raw, filename, cfg_dir):
    cands = candidate_paths(out_dir_raw, cfg_dir)
    file_cands = [c / filename for c in cands]
    for c in file_cands:
        if c.exists():
            return c.resolve(), file_cands
    return file_cands[0], file_cands


def get_video_resolution(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return {"width": w, "height": h, "fps": fps}


def load_intrinsics(path):
    with np.load(path, allow_pickle=True) as d:
        K = None
        for k in ["K", "mtx", "camera_matrix", "cameraMatrix", "intrinsic_matrix"]:
            if k in d:
                K = np.array(d[k], dtype=np.float64)
                break
        if K is None:
            raise ValueError(f"Cannot find K in {path}. Keys={list(d.keys())}")

        image_size = None
        if "image_size" in d:
            arr = np.array(d["image_size"]).reshape(-1)
            if arr.size >= 2:
                image_size = (int(arr[0]), int(arr[1]))
        return K, image_size


def load_extrinsics(path):
    with np.load(path, allow_pickle=True) as d:
        R = None
        for k in ["R", "rotation_matrix"]:
            if k in d:
                arr = np.array(d[k], dtype=np.float64)
                if arr.shape == (3, 3):
                    R = arr
                    break
        if R is None:
            for k in ["rvec", "rotation_vector", "rvecs"]:
                if k in d:
                    r = np.array(d[k], dtype=np.float64).reshape(-1)
                    if r.size == 3:
                        R, _ = cv2.Rodrigues(r.reshape(3, 1))
                        break
        if R is None:
            raise ValueError(f"Cannot find R in {path}. Keys={list(d.keys())}")

        t = None
        for k in ["t", "T", "tvec", "translation_vector", "translation", "tvecs"]:
            if k in d:
                t = np.array(d[k], dtype=np.float64).reshape(-1)
                if t.size >= 3:
                    t = t[:3].reshape(3, 1)
                    break
        if t is None:
            raise ValueError(f"Cannot find t in {path}. Keys={list(d.keys())}")
    return R, t


def norm_roi(roi):
    if roi is None:
        roi = {}
    return {
        "enabled": bool(roi.get("enabled", False)),
        "x": int(roi.get("x", 0)),
        "y": int(roi.get("y", 0)),
        "w": int(roi.get("w", 0)),
        "h": int(roi.get("h", 0)),
    }


def norm_board(board):
    if board is None:
        board = {}
    return {
        "squares_x": int(board.get("squares_x", -1)),
        "squares_y": int(board.get("squares_y", -1)),
        "square_size_mm": float(board.get("square_size_mm", -1)),
        "marker_size_mm": float(board.get("marker_size_mm", -1)),
        "aruco_dict": str(board.get("aruco_dict", "")),
        "legacy_pattern": bool(board.get("legacy_pattern", False)),
    }


class Report:
    def __init__(self):
        self.entries = []

    def add(self, level, check, message):
        self.entries.append({"level": level, "check": check, "message": message})

    def status(self):
        has_fail = any(e["level"] == "FAIL" for e in self.entries)
        has_warn = any(e["level"] == "WARN" for e in self.entries)
        if has_fail:
            return "FAIL"
        if has_warn:
            return "WARN"
        return "PASS"

    def print(self):
        print("== Phase 0 Consistency Check ==")
        for e in self.entries:
            print(f"[{e['level']}] {e['check']}: {e['message']}")
        print(f"\nOverall: {self.status()}")


def main():
    ap = argparse.ArgumentParser(description="Phase 0 consistency check for real-world 2D->3D pipeline.")
    ap.add_argument("--intrin-cam1-cfg", default=DEFAULT_INTRIN_CAM1)
    ap.add_argument("--intrin-cam2-cfg", default=DEFAULT_INTRIN_CAM2)
    ap.add_argument("--extrin-cam1-cfg", default=DEFAULT_EXTRIN_CAM1)
    ap.add_argument("--extrin-cam2-cfg", default=DEFAULT_EXTRIN_CAM2)
    ap.add_argument("--cam1-video", default="", help="Optional reconstruction video path for cam1.")
    ap.add_argument("--cam2-video", default="", help="Optional reconstruction video path for cam2.")
    ap.add_argument("--out-json", default="docs/phase0_consistency_report.json")
    args = ap.parse_args()

    report = Report()

    cfg_paths = {
        "intrin_cam1": Path(args.intrin_cam1_cfg),
        "intrin_cam2": Path(args.intrin_cam2_cfg),
        "extrin_cam1": Path(args.extrin_cam1_cfg),
        "extrin_cam2": Path(args.extrin_cam2_cfg),
    }

    for name, p in cfg_paths.items():
        if p.exists():
            report.add("PASS", "config_exists", f"{name}: {p}")
        else:
            report.add("FAIL", "config_exists", f"{name}: missing {p}")

    if report.status() == "FAIL":
        report.print()
        raise SystemExit(1)

    cfgs = {k: load_yaml(v) for k, v in cfg_paths.items()}
    cfg_dirs = {k: v.parent.resolve() for k, v in cfg_paths.items()}

    cameras = {}
    for cam in ["cam1", "cam2"]:
        intrin_key = f"intrin_{cam}"
        extrin_key = f"extrin_{cam}"
        intrin_cfg = cfgs[intrin_key]
        extrin_cfg = cfgs[extrin_key]
        intrin_dir = cfg_dirs[intrin_key]
        extrin_dir = cfg_dirs[extrin_key]

        intrin_video_raw = intrin_cfg.get("input", {}).get("video", "")
        intrin_video, _ = resolve_existing_path(intrin_video_raw, intrin_dir) if intrin_video_raw else (None, [])

        extrin_video_raw = extrin_cfg.get("input", {}).get("video", "")
        extrin_video, _ = resolve_existing_path(extrin_video_raw, extrin_dir) if extrin_video_raw else (None, [])

        prefix_i = intrin_cfg.get("output", {}).get("prefix", cam)
        out_i = intrin_cfg.get("output", {}).get("out_dir", f"calib_out_{cam}")
        intrin_npz, _ = resolve_output_file(out_i, f"{prefix_i}_intrinsics.npz", intrin_dir)

        prefix_e = extrin_cfg.get("output", {}).get("prefix", cam)
        out_e = extrin_cfg.get("output", {}).get("out_dir", f"calib_out_{cam}")
        extrin_npz, _ = resolve_output_file(out_e, f"{prefix_e}_extrinsics.npz", extrin_dir)

        ref_intrin_raw = extrin_cfg.get("intrinsics", {}).get("path", "")
        ref_intrin, _ = resolve_existing_path(ref_intrin_raw, extrin_dir) if ref_intrin_raw else (None, [])

        cameras[cam] = {
            "intrin_cfg": intrin_cfg,
            "extrin_cfg": extrin_cfg,
            "intrin_video": intrin_video,
            "extrin_video": extrin_video,
            "intrin_npz": intrin_npz,
            "extrin_npz": extrin_npz,
            "ref_intrin": ref_intrin,
            "roi_intrin": norm_roi(intrin_cfg.get("roi")),
            "roi_extrin": norm_roi(extrin_cfg.get("roi")),
            "board_intrin": norm_board(intrin_cfg.get("board")),
            "board_extrin": norm_board(extrin_cfg.get("board")),
        }

    for cam in ["cam1", "cam2"]:
        c = cameras[cam]
        for key in ["intrin_video", "extrin_video", "intrin_npz", "extrin_npz"]:
            p = c[key]
            if p and Path(p).exists():
                report.add("PASS", "path_exists", f"{cam} {key}: {p}")
            else:
                report.add("FAIL", "path_exists", f"{cam} {key}: missing")

    rois = [
        cameras["cam1"]["roi_intrin"],
        cameras["cam1"]["roi_extrin"],
        cameras["cam2"]["roi_intrin"],
        cameras["cam2"]["roi_extrin"],
    ]
    if all(r == rois[0] for r in rois):
        report.add("PASS", "roi_consistency", f"All configs share ROI {rois[0]}")
    else:
        report.add("FAIL", "roi_consistency", f"ROI mismatch detected: {rois}")

    if cameras["cam1"]["board_intrin"] == cameras["cam2"]["board_intrin"]:
        report.add("PASS", "board_intrinsics_consistency", f"Intrinsics board match: {cameras['cam1']['board_intrin']}")
    else:
        report.add("FAIL", "board_intrinsics_consistency", "cam1/cam2 intrinsics board mismatch")

    if cameras["cam1"]["board_extrin"] == cameras["cam2"]["board_extrin"]:
        report.add("PASS", "board_extrinsics_consistency", f"Extrinsics board match: {cameras['cam1']['board_extrin']}")
    else:
        report.add("FAIL", "board_extrinsics_consistency", "cam1/cam2 extrinsics board mismatch")

    for cam in ["cam1", "cam2"]:
        c = cameras[cam]
        if c["ref_intrin"] is None:
            report.add("FAIL", "intrinsics_pairing", f"{cam} extrinsics config intrinsics.path unresolved")
        else:
            expected = str(c["intrin_npz"])
            actual = str(c["ref_intrin"])
            if os.path.normcase(os.path.normpath(actual)) == os.path.normcase(os.path.normpath(expected)):
                report.add("PASS", "intrinsics_pairing", f"{cam} extrinsics uses matching intrinsics file")
            else:
                report.add("FAIL", "intrinsics_pairing", f"{cam} extrinsics points to {actual}, expected {expected}")

    for cam in ["cam1", "cam2"]:
        c = cameras[cam]
        resolutions = {}
        if c["intrin_video"] is not None:
            resolutions["intrin_video"] = get_video_resolution(c["intrin_video"])
        if c["extrin_video"] is not None:
            resolutions["extrin_video"] = get_video_resolution(c["extrin_video"])
        if Path(c["intrin_npz"]).exists():
            try:
                _, img_size = load_intrinsics(c["intrin_npz"])
                if img_size is not None:
                    resolutions["intrin_npz"] = {"width": img_size[0], "height": img_size[1]}
            except Exception as ex:
                report.add("FAIL", "intrinsics_load", f"{cam}: {ex}")

        custom_video = Path(getattr(args, f"{cam}_video"))
        if str(custom_video):
            if custom_video.exists():
                resolutions["pipeline_video"] = get_video_resolution(custom_video)
            else:
                report.add("WARN", "pipeline_video", f"{cam} pipeline video not found: {custom_video}")

        wh = {(v["width"], v["height"]) for v in resolutions.values() if v is not None}
        if len(wh) <= 1 and wh:
            report.add("PASS", "resolution_consistency", f"{cam} resolution consistent: {list(wh)[0]}")
        elif len(wh) == 0:
            report.add("WARN", "resolution_consistency", f"{cam} no readable resolution sources")
        else:
            report.add("FAIL", "resolution_consistency", f"{cam} resolution mismatch: {resolutions}")

    RT = {}
    for cam in ["cam1", "cam2"]:
        c = cameras[cam]
        if Path(c["intrin_npz"]).exists():
            try:
                K, img_size = load_intrinsics(c["intrin_npz"])
                fx = float(K[0, 0])
                fy = float(K[1, 1])
                cx = float(K[0, 2])
                cy = float(K[1, 2])
                if fx > 0 and fy > 0:
                    report.add("PASS", "intrinsics_fx_fy", f"{cam} fx={fx:.2f}, fy={fy:.2f}")
                else:
                    report.add("FAIL", "intrinsics_fx_fy", f"{cam} invalid fx/fy: {fx}, {fy}")

                if img_size is not None:
                    w, h = img_size
                    if 0 <= cx <= w and 0 <= cy <= h:
                        report.add("PASS", "principal_point", f"{cam} principal point in image bounds ({cx:.1f}, {cy:.1f})")
                    else:
                        report.add("WARN", "principal_point", f"{cam} principal point out of bounds ({cx:.1f}, {cy:.1f}) for {w}x{h}")
            except Exception as ex:
                report.add("FAIL", "intrinsics_load", f"{cam}: {ex}")

        if Path(c["extrin_npz"]).exists():
            try:
                R, t = load_extrinsics(c["extrin_npz"])
                RT[cam] = (R, t)
                det = float(np.linalg.det(R))
                if abs(det - 1.0) < 0.05:
                    report.add("PASS", "rotation_matrix", f"{cam} det(R)={det:.4f}")
                else:
                    report.add("WARN", "rotation_matrix", f"{cam} det(R)={det:.4f} (far from 1)")
            except Exception as ex:
                report.add("FAIL", "extrinsics_load", f"{cam}: {ex}")

    if "cam1" in RT and "cam2" in RT:
        R1, t1 = RT["cam1"]
        R2, t2 = RT["cam2"]
        C1 = -R1.T @ t1
        C2 = -R2.T @ t2
        baseline = float(np.linalg.norm(C1 - C2))
        if baseline < 5.0:
            report.add("FAIL", "baseline", f"baseline too small: {baseline:.2f} (units from calibration board)")
        elif baseline < 50.0:
            report.add("WARN", "baseline", f"baseline small: {baseline:.2f} (check camera spacing/world consistency)")
        elif baseline > 10000.0:
            report.add("WARN", "baseline", f"baseline very large: {baseline:.2f} (possible world mismatch)")
        else:
            report.add("PASS", "baseline", f"baseline={baseline:.2f}")
        report.add("WARN", "baseline_note", "baseline assumes cam1/cam2 extrinsics share the same world/board pose.")

    output = {
        "overall": report.status(),
        "entries": report.entries,
        "resolved_paths": {
            cam: {
                "intrin_video": str(cameras[cam]["intrin_video"]) if cameras[cam]["intrin_video"] else None,
                "extrin_video": str(cameras[cam]["extrin_video"]) if cameras[cam]["extrin_video"] else None,
                "intrin_npz": str(cameras[cam]["intrin_npz"]),
                "extrin_npz": str(cameras[cam]["extrin_npz"]),
                "ref_intrin": str(cameras[cam]["ref_intrin"]) if cameras[cam]["ref_intrin"] else None,
            }
            for cam in ["cam1", "cam2"]
        },
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    report.print()
    print(f"Saved: {out_json}")
    if report.status() == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

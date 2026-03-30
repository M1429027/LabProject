import argparse
import os
import cv2
import cv2.aruco as aruco
import numpy as np
import yaml


def load_yaml(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dicts():
    return [
        ("DICT_4X4_50", aruco.DICT_4X4_50),
        ("DICT_4X4_100", aruco.DICT_4X4_100),
        ("DICT_4X4_250", aruco.DICT_4X4_250),
        ("DICT_4X4_1000", aruco.DICT_4X4_1000),
    ]


def detect_markers(gray, dictionary):
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = aruco.detectMarkers(gray, dictionary)
    n = len(ids) if ids is not None else 0
    return corners, ids, n


def detect_charuco(gray, board, dictionary):
    if hasattr(aruco, "CharucoDetector"):
        detector = aruco.CharucoDetector(board)
        corners, ids, _, _ = detector.detectBoard(gray)
        return corners, ids
    marker_corners, marker_ids, _ = aruco.detectMarkers(gray, dictionary)
    if marker_ids is None or len(marker_ids) == 0:
        return None, None
    ret, ch_corners, ch_ids = aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board
    )
    if ret is None or ret < 4:
        return None, None
    return ch_corners, ch_ids


def make_board(sx, sy, square_size, marker_size, dictionary, legacy_pattern=None):
    board = aruco.CharucoBoard((sx, sy), square_size, marker_size, dictionary)
    if legacy_pattern is not None and hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(bool(legacy_pattern))
    return board


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="debug_camera2.mp4")
    ap.add_argument("--frame", type=int, default=30)
    ap.add_argument("--config", default="calib_charuco_v2/config_extrinsics.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config) or {}
    board_cfg = cfg.get("board", {})
    squares_x = int(board_cfg.get("squares_x", 6))
    squares_y = int(board_cfg.get("squares_y", 9))
    square_size = float(board_cfg.get("square_size_mm", 55.0))
    marker_size = float(board_cfg.get("marker_size_mm", 41.0))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to read frame")

    cv2.imwrite("debug_raw_frame.jpg", frame)
    print(f"Saved debug_raw_frame.jpg (frame {args.frame})")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    print("\n[Step 1] ArUco marker detection (4x4 dicts)")
    marker_stats = []
    for name, dict_id in get_dicts():
        dictionary = aruco.getPredefinedDictionary(dict_id)
        corners, ids, n = detect_markers(gray, dictionary)
        if ids is not None and len(ids) > 0:
            ids_flat = ids.flatten()
            id_min = int(ids_flat.min())
            id_max = int(ids_flat.max())
        else:
            id_min = None
            id_max = None
        marker_stats.append((name, n, id_min, id_max, dictionary, corners, ids))
        print(f"  {name}: markers={n}  id_range={id_min}..{id_max}")

    # Use the dict with max markers as base; tie-breaker by lower dict size
    marker_stats.sort(key=lambda x: (-x[1], x[0]))
    best_marker = marker_stats[0]
    best_name, best_n, best_id_min, best_id_max, best_dict, best_corners, best_ids = best_marker

    if best_n == 0:
        print("\nNo ArUco markers detected with any DICT_4X4_xxx.")
        return

    vis_markers = frame.copy()
    aruco.drawDetectedMarkers(vis_markers, best_corners, best_ids)
    cv2.imwrite("debug_markers_found.jpg", vis_markers)
    print(f"\nBest dict by markers: {best_name} (markers={best_n})")
    print("Saved debug_markers_found.jpg")

    # Candidate squares: as-is, +1 (if user provided inner corners), -1, swapped
    candidates = []
    base = [(squares_x, squares_y)]
    if squares_x > 2 and squares_y > 2:
        base += [(squares_x + 1, squares_y + 1), (squares_x - 1, squares_y - 1)]
    base += [(squares_y, squares_x)]
    # dedupe
    for sx, sy in base:
        if sx < 2 or sy < 2:
            continue
        if (sx, sy) not in candidates:
            candidates.append((sx, sy))

    legacy_opts = [None]
    # if OpenCV supports legacy pattern, test both True/False
    try:
        _tmp = aruco.CharucoBoard((2, 2), 10.0, 7.0, best_dict)
        if hasattr(_tmp, "setLegacyPattern"):
            legacy_opts = [False, True]
    except Exception:
        pass

    print("\n[Step 2] Charuco corner scan (dict x squares x legacyPattern)")
    best = None
    for name, dict_id in get_dicts():
        dictionary = aruco.getPredefinedDictionary(dict_id)
        for (sx, sy) in candidates:
            for legacy in legacy_opts:
                board = make_board(sx, sy, square_size, marker_size, dictionary, legacy)
                ch_corners, ch_ids = detect_charuco(gray, board, dictionary)
                n_ch = len(ch_ids) if ch_ids is not None else 0
                label = f"{name}  squares=({sx},{sy})  legacy={legacy}"
                print(f"  {label}: corners={n_ch}")
                if best is None or n_ch > best["n_ch"]:
                    best = {
                        "name": name,
                        "sx": sx,
                        "sy": sy,
                        "legacy": legacy,
                        "n_ch": n_ch,
                        "dictionary": dictionary,
                        "ch_corners": ch_corners,
                        "ch_ids": ch_ids,
                    }

    if best is None:
        print("No Charuco corners found.")
        return

    print("\n[Best Result]")
    print(f"  dict={best['name']}")
    print(f"  squares_x={best['sx']}  squares_y={best['sy']}")
    print(f"  legacyPattern={best['legacy']}")
    print(f"  corners={best['n_ch']}")

    vis_charuco = frame.copy()
    aruco.drawDetectedMarkers(vis_charuco, best_corners, best_ids)
    if best["ch_corners"] is not None and best["ch_ids"] is not None:
        cv2.aruco.drawDetectedCornersCharuco(vis_charuco, best["ch_corners"], best["ch_ids"])
    cv2.imwrite("debug_charuco_found.jpg", vis_charuco)
    print("Saved debug_charuco_found.jpg")

    print("\nSuggested config updates:")
    print(f"  aruco_dict: {best['name']}")
    print(f"  squares_x: {best['sx']}")
    print(f"  squares_y: {best['sy']}")
    if best["legacy"] is not None:
        print(f"  legacyPattern: {best['legacy']}")


if __name__ == "__main__":
    main()

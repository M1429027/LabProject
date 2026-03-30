import os
import yaml
import cv2
import cv2.aruco as aruco
import numpy as np


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_aruco_dict(name):
    if name is None:
        raise ValueError("aruco_dict is required")
    if isinstance(name, int):
        return aruco.getPredefinedDictionary(name)
    if not isinstance(name, str):
        raise ValueError("aruco_dict must be str or int")
    if not name.startswith("DICT_"):
        raise ValueError("aruco_dict must look like DICT_6X6_250")
    if not hasattr(aruco, name):
        raise ValueError(f"Unknown aruco_dict: {name}")
    return aruco.getPredefinedDictionary(getattr(aruco, name))


def make_board(board_cfg):
    squares_x = int(board_cfg["squares_x"])
    squares_y = int(board_cfg["squares_y"])
    square_size = float(board_cfg["square_size_mm"])
    marker_size = float(board_cfg["marker_size_mm"])
    dictionary = get_aruco_dict(board_cfg["aruco_dict"])
    legacy = board_cfg.get("legacy_pattern", board_cfg.get("legacyPattern", None))
    board = aruco.CharucoBoard((squares_x, squares_y), square_size, marker_size, dictionary)
    if legacy is not None and hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(bool(legacy))
    return board, dictionary

def apply_roi_mask(gray, roi_cfg):
    h, w = gray.shape[:2]
    if not roi_cfg or not roi_cfg.get("enabled", False):
        return gray, (0, 0, w, h)
    x = int(roi_cfg.get("x", 0))
    y = int(roi_cfg.get("y", 0))
    rw = int(roi_cfg.get("w", 0))
    rh = int(roi_cfg.get("h", 0))
    if rw <= 0 or rh <= 0:
        return gray, (0, 0, w, h)
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    rw = max(1, min(rw, w - x))
    rh = max(1, min(rh, h - y))
    masked = np.zeros_like(gray)
    masked[y:y + rh, x:x + rw] = gray[y:y + rh, x:x + rw]
    return masked, (x, y, rw, rh)


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


def compute_reproj_rms(obj_pts, img_pts, K, dist, rvec, tvec):
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    obs = img_pts.reshape(-1, 2)
    err = np.linalg.norm(proj - obs, axis=1)
    return float(np.sqrt(np.mean(err ** 2)))


def rotmat_to_quat(R):
    m = R
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_to_rotmat(q):
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
    ], dtype=np.float64)
    return R


def average_quaternions(quats):
    if len(quats) == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ref = quats[0]
    acc = np.zeros(4, dtype=np.float64)
    for q in quats:
        if np.dot(q, ref) < 0:
            q = -q
        acc += q
    n = np.linalg.norm(acc)
    if n < 1e-12:
        return ref
    return acc / n


def robust_filter_by_rms(rms_list, max_rms_px=None):
    rms = np.array(rms_list, dtype=np.float64)
    if rms.size == 0:
        return np.array([], dtype=np.int64)
    median = np.median(rms)
    mad = np.median(np.abs(rms - median))
    if mad < 1e-9:
        thresh = median + 1e-6
    else:
        thresh = median + 2.5 * mad
    if max_rms_px is not None and max_rms_px > 0:
        thresh = min(thresh, float(max_rms_px))
    keep = np.where(rms <= thresh)[0]
    return keep


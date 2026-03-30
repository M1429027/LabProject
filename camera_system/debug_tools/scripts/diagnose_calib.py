"""
diagnose_calib.py
=================
診斷相機標定 .npz 檔案，並做基本的 geometry sanity check。
執行方式：
    python diagnose_calib.py
"""
import numpy as np
import cv2

FILES = {
    "cam1_intrinsics": "cam1_intrinsics.npz",
    "cam1_extrinsics": "cam1_extrinsics.npz",
    "cam2_intrinsics": "cam2_intrinsics.npz",
    "cam2_extrinsics": "cam2_extrinsics.npz",
}

def load_and_print(label, path):
    print(f"\n{'='*60}")
    print(f"  {label}  →  {path}")
    print(f"{'='*60}")
    d = np.load(path)
    data = {}
    for k in d.files:
        v = d[k]
        data[k] = v
        print(f"  key='{k}':  shape={v.shape}  dtype={v.dtype}")
        with np.printoptions(precision=4, suppress=True):
            print(f"    {v}")
    return data

def try_build_P(intrin_data, extrin_data, label):
    print(f"\n--- Projection Matrix check for {label} ---")

    # 找 K
    K = None
    for k in ['mtx', 'camera_matrix', 'K', 'intrinsic_matrix', 'cameraMatrix']:
        if k in intrin_data:
            K = np.array(intrin_data[k], dtype=np.float64)
            print(f"  K from key='{k}'")
            break
    if K is None:
        for k, v in intrin_data.items():
            if np.array(v).shape == (3, 3):
                K = np.array(v, dtype=np.float64)
                print(f"  K guessed from key='{k}'")
                break
    if K is None:
        print("  ERROR: Cannot find K!")
        return

    # 找 R / rvec
    R = None
    for k in ['rvec', 'rvecs', 'rotation_vector', 'R', 'rotation_matrix']:
        if k in extrin_data:
            arr = np.array(extrin_data[k], dtype=np.float64).flatten()
            if arr.shape == (3,):
                R, _ = cv2.Rodrigues(arr)
                print(f"  R from rvec key='{k}'  → R det={np.linalg.det(R):.4f}")
            elif arr.reshape(-1).shape[0] == 9:
                R = arr.reshape(3, 3)
                print(f"  R from matrix key='{k}'  → R det={np.linalg.det(R):.4f}")
            break
    if R is None:
        print("  ERROR: Cannot find R!")
        return

    # 找 t
    t = None
    for k in ['tvec', 'tvecs', 't', 'translation', 'T', 'translation_vector']:
        if k in extrin_data:
            t = np.array(extrin_data[k], dtype=np.float64).flatten().reshape(3, 1)
            print(f"  t from key='{k}':  {t.flatten()}")
            break
    if t is None:
        print("  ERROR: Cannot find t!")
        return

    P = K @ np.hstack([R, t])
    print(f"\n  K =\n{np.round(K, 2)}")
    print(f"\n  R =\n{np.round(R, 4)}")
    print(f"\n  t = {t.flatten()}")
    print(f"\n  P (3x4) =\n{np.round(P, 2)}")

    # 相機中心 C = -R^T @ t
    C = -R.T @ t
    print(f"\n  Camera Center in World = {C.flatten()}")

    # 焦距合理性
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    print(f"\n  Focal (fx,fy) = ({fx:.1f}, {fy:.1f})")
    print(f"  Principal pt (cx,cy) = ({cx:.1f}, {cy:.1f})")
    if fx < 100 or fx > 10000:
        print("  ⚠️  WARNING: focal length looks unusual!")
    if abs(np.linalg.det(R) - 1.0) > 0.01:
        print("  ⚠️  WARNING: R is not a proper rotation matrix (det != 1)!")

    # t 的量綱診斷
    t_norm = np.linalg.norm(t)
    print(f"\n  |t| = {t_norm:.4f}  (物理距離，單位與標定板格子相同)")
    if t_norm < 1e-6:
        print("  ⚠️  WARNING: t is near zero — camera1 world origin?")
    elif t_norm > 1e5:
        print("  ⚠️  WARNING: t is very large — unit mismatch (mm vs m)?")

    return P, C

print("\n" + "="*60)
print("  Camera Calibration Diagnostic")
print("="*60)

raw = {}
for label, path in FILES.items():
    raw[label] = load_and_print(label, path)

P1, C1 = try_build_P(raw["cam1_intrinsics"], raw["cam1_extrinsics"], "CAM1") or (None, None)
P2, C2 = try_build_P(raw["cam2_intrinsics"], raw["cam2_extrinsics"], "CAM2") or (None, None)

if C1 is not None and C2 is not None:
    baseline = np.linalg.norm(C1 - C2)
    print(f"\n{'='*60}")
    print(f"  Baseline (distance between cameras) = {baseline:.4f}  (same unit as t)")
    print(f"{'='*60}")
    if baseline < 1e-4:
        print("  ❌ CRITICAL: Baseline is essentially ZERO! Both cameras at same position.")
        print("     → 三角測量爆炸的根本原因！相機等效共位，射線幾乎平行，深度無窮大。")
    elif baseline > 1e5:
        print("  ⚠️  WARNING: Baseline is very large — unit mismatch?")
    else:
        print("  ✅ Baseline looks reasonable.")

    # 計算 epipolar angle
    d1 = -R1.T @ np.array([0, 0, 1]) if 'R1' in dir() else None

    # 做一個測試三角測量
    print("\n--- Test Triangulation (mid-image points) ---")
    from scipy.optimize import least_squares
    import itertools

    def project_pt(p3d, P):
        h = P @ np.append(p3d, 1.0)
        return h[:2] / h[2]

    def reproj_err(p3d, Ps, obs):
        r = []
        for P, o in zip(Ps, obs):
            r.extend(project_pt(p3d, P) - np.array(o))
        return np.array(r)

    def svd_tri(Ps, pts):
        A = np.zeros((len(Ps)*2, 4))
        for i, (P, (x, y)) in enumerate(zip(Ps, pts)):
            A[2*i]   = x * P[2] - P[0]
            A[2*i+1] = y * P[2] - P[1]
        _, _, Vh = np.linalg.svd(A)
        X = Vh[-1]
        return X[:3] / X[3]

    # 取影像中心作為測試 2D 點
    cx1, cy1 = raw["cam1_intrinsics"].get("mtx", np.zeros((3,3)))[0,2], raw["cam1_intrinsics"].get("mtx", np.zeros((3,3)))[1,2]
    cx2, cy2 = raw["cam2_intrinsics"].get("mtx", np.zeros((3,3)))[0,2], raw["cam2_intrinsics"].get("mtx", np.zeros((3,3)))[1,2]

    test_pts = [[cx1, cy1], [cx2, cy2]]
    test_Ps  = [P1, P2]
    try:
        pt3d_init = svd_tri(test_Ps, test_pts)
        print(f"  SVD init  = {pt3d_init}")
        res = least_squares(reproj_err, pt3d_init, args=(test_Ps, test_pts), method='lm')
        pt3d_lm = res.x
        print(f"  LM refine = {pt3d_lm}")
        print(f"  |pt3d|    = {np.linalg.norm(pt3d_lm):.4f}")
        e1 = np.linalg.norm(project_pt(pt3d_lm, P1) - np.array(test_pts[0]))
        e2 = np.linalg.norm(project_pt(pt3d_lm, P2) - np.array(test_pts[1]))
        print(f"  Reproj error cam1={e1:.2f}px  cam2={e2:.2f}px")
        if np.any(np.isinf(pt3d_lm)) or np.any(np.isnan(pt3d_lm)):
            print("  ❌ CRITICAL: Triangulation diverged (inf/nan)!")
        elif np.linalg.norm(pt3d_lm) > 1e6:
            print("  ❌ CRITICAL: 3D point is at infinity — degenerate camera configuration!")
    except Exception as ex:
        print(f"  ERROR: {ex}")

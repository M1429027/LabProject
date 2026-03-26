"""
診斷腳本 v2：純 Python + json，不依賴 numpy/cv2
分析 YOLO 輸出 keypoints 和三角測量結果
"""
import json, math

print("=" * 70)
print("  1. YOLO Keypoints 診斷")
print("=" * 70)

for cam in ['cam1', 'cam2']:
    kp_path = f'real_world_output/keypoints_{cam}.json'
    with open(kp_path) as f:
        data = json.load(f)
    
    meta = data['metadata']
    frames = data['frames']
    print(f"\n[{cam}] {meta['total_frames']} frames, {meta['width']}x{meta['height']}, {meta['fps']:.1f}fps")
    
    found = 0
    for fr in frames:
        if fr['people']:
            kpts = fr['people'][0]['keypoints']
            valid_kpts = [k for k in kpts if k['confidence'] > 0.3]
            xs = [k['x'] for k in valid_kpts]
            ys = [k['y'] for k in valid_kpts]
            if xs:
                print(f"  Frame {fr['frame']:4d}: {len(xs)} joints, "
                      f"x=[{min(xs):.0f},{max(xs):.0f}], y=[{min(ys):.0f},{max(ys):.0f}]")
                nose  = kpts[0]
                ankle_l = kpts[15]
                ankle_r = kpts[16]
                print(f"    nose:        x={nose['x']:.1f},  y={nose['y']:.1f},  conf={nose['confidence']:.2f}")
                print(f"    left_ankle:  x={ankle_l['x']:.1f}, y={ankle_l['y']:.1f}, conf={ankle_l['confidence']:.2f}")
                yn = nose['y']
                ya = ankle_l['y']
                # Image coord: y=0 is top, y=max is bottom
                # Normal: nose should have SMALLER y than ankle
                print(f"    nose_y < ankle_y? {yn < ya} (True = normal, False = FLIPPED IMAGE)")
            found += 1
            if found >= 3:
                break
    
    no_person = sum(1 for fr in frames if not fr['people'])
    print(f"  Frames with NO person: {no_person}/{len(frames)} ({100*no_person/len(frames):.1f}%)")

print()
print("=" * 70)
print("  2. Camera Parameters 診斷 (純數學)")
print("=" * 70)

def mat3x3_det(M):
    return (M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
           -M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
           +M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]))

def mat_mul(A, B):
    rows_a, cols_a = len(A), len(A[0])
    rows_b, cols_b = len(B), len(B[0])
    result = [[0]*cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for k in range(cols_a):
            for j in range(cols_b):
                result[i][j] += A[i][k] * B[k][j]
    return result

def transpose(M):
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]

def project_P(P, world_xyz):
    x, y, z = world_xyz
    h = [sum(P[i][j]*v for j,v in enumerate([x,y,z,1])) for i in range(3)]
    if abs(h[2]) < 1e-8:
        return None
    return [h[0]/h[2], h[1]/h[2]]

with open('real_world_output/camera_params.json') as f:
    cams = json.load(f)

for cam_id, c in cams.items():
    K = c['K']
    R = c['R']
    t = c['t']
    P = c['P']
    
    print(f"\n[{cam_id}]")
    print(f"  K: fx={K[0][0]:.1f}, fy={K[1][1]:.1f}, cx={K[0][2]:.1f}, cy={K[1][2]:.1f}")
    print(f"  t (mm?): [{t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}]")
    
    # 相機在世界座標的位置 = -R^T @ t
    Rt = transpose(R)
    cam_pos = [-sum(Rt[i][j]*t[j] for j in range(3)) for i in range(3)]
    print(f"  Camera world pos: [{cam_pos[0]:.1f}, {cam_pos[1]:.1f}, {cam_pos[2]:.1f}]")
    
    # det(R)
    det = mat3x3_det(R)
    print(f"  det(R) = {det:.4f} (should be +1.0)")
    
    # 把世界原點 (0,0,0) 投影
    uv = project_P(P, [0, 0, 0])
    print(f"  World origin projected to: {['%.1f'%v for v in uv] if uv else 'BEHIND CAMERA'}")
    
    # 把世界 Z 軸端點 (0, 0, 100) 投影 (表示棋盤格 Z 軸藍色的頂點)
    uv_z = project_P(P, [0, 0, 100])
    print(f"  Z-axis tip (0,0,100) projected to: {['%.1f'%v for v in uv_z] if uv_z else 'BEHIND CAMERA'}")
    
    if uv and uv_z:
        # 在影像中, Z 軸 tip 的 v 應該比 origin 更小 (在影像中向上)
        v_origin = uv[1]
        v_z_tip  = uv_z[1]
        print(f"  Z-up check: origin_v={v_origin:.1f}, z_tip_v={v_z_tip:.1f}, "
              f"z_tip_above_origin? {v_z_tip < v_origin}")

print()
print("=" * 70)
print("  3. 三角測量結果診斷")
print("=" * 70)

with open('real_world_output/skeleton_3d.json') as f:
    skel = json.load(f)

frames_3d = skel['frames']
valid_frames = [fr for fr in frames_3d if fr['valid_count'] > 0]
print(f"Total frames: {len(frames_3d)}, with valid joints: {len(valid_frames)}")

nose_positions = []
for fr in valid_frames[:5]:
    kpts3d = fr['keypoints_3d']
    positions = [k['position'] for k in kpts3d if k['valid']]
    if positions:
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]
        print(f"\n  Frame {fr['frame']:4d}: {fr['valid_count']} joints")
        print(f"    X: [{min(xs):.2f}, {max(xs):.2f}]")
        print(f"    Y: [{min(ys):.2f}, {max(ys):.2f}]")
        print(f"    Z: [{min(zs):.2f}, {max(zs):.2f}]")
        
        nose  = next((k for k in kpts3d if k['id'] == 0 and k['valid']), None)
        l_hip = next((k for k in kpts3d if k['id'] == 11 and k['valid']), None)
        l_ank = next((k for k in kpts3d if k['id'] == 15 and k['valid']), None)
        if nose and l_ank:
            nz, az = nose['position'][2], l_ank['position'][2]
            print(f"    nose_z={nz:.3f}, ankle_z={az:.3f}")
            print(f"    nose ABOVE ankle? (z_nose > z_ankle if Z=up): {nz > az}")
        
        nose_positions.append(positions[0] if positions else None)

# 速度分析：找噴飛的幀
if len(valid_frames) > 10:
    all_nose = []
    for fr in valid_frames[:100]:
        nose = next((k for k in fr['keypoints_3d'] if k['id'] == 0 and k['valid']), None)
        if nose:
            all_nose.append((fr['frame'], nose['position']))
    
    if len(all_nose) > 2:
        print(f"\n  Nose joint 速度分析 (連續幀間距):")
        speeds = []
        for i in range(1, len(all_nose)):
            f1, p1 = all_nose[i-1]
            f2, p2 = all_nose[i]
            dist = math.sqrt(sum((p2[j]-p1[j])**2 for j in range(3)))
            speeds.append(dist)
        
        speeds.sort(reverse=True)
        print(f"    最大跳動: {speeds[0]:.2f}")
        print(f"    第5大跳動: {speeds[4]:.2f}")
        mean_s = sum(speeds)/len(speeds)
        print(f"    平均: {mean_s:.2f}")
        print(f"    (單位與世界座標相同，若 t 是mm則此為mm)")

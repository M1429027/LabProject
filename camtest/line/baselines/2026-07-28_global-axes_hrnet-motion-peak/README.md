# 2026-07-28 Correct Demo Baseline

This snapshot records the first verified usable four-camera 3D pose baseline.

## Frozen configuration

- Intrinsics:
  - cam1: `camtest/line/calib_out_cam1_hvflip/cam1_intrinsics.npz`
  - cam2: `camtest/line/calib_out_cam2_hvflip_recalc/cam2_intrinsics.npz`
  - cam3: `camtest/line/calib_out_cam3_hvflip_fast/cam3_intrinsics.npz`
  - cam4: `camtest/line/calib_out_cam4_hvflip_newintr_20260727/cam4_intrinsics.npz`
- Extrinsics: `extrinsics/cam1..4_line_extrinsics_refined.npz`
- Synchronization: explicit eight arm-motion peaks with affine frame mapping.
- 2D pose: HRNet filtered tracks.
- Left/right mapping: no camera-level COCO left/right swap.
- 3D: pose-level subset selection, strict joint RANSAC, temporal gate, smoothing and skeleton optimization.

## Verified results

- Global bilateral epipolar median: `5.7515 px`
- LR-invariant torso-center epipolar median: `4.94 px`
- Frames: `533`
- Empty camera-subset frames: `0`
- Temporal joint rejections: `0`
- Bad bones before/after projection: `0 / 0`
- Root Z range: `0.06196 m`
- Camera layout: four sides of the field, all optical axes pointing inward.

## Canonical outputs

- 3D JSON: `camtest/line/demo_hrnet_enhanced_20260728/robust_3d_global_axes_motion_peak_noswap_20260729/triangulated_enhanced_robust.json`
- 3D video: `camtest/line/demo_hrnet_enhanced_20260728/robust_3d_global_axes_motion_peak_noswap_20260729/triangulated_hrnet_global_axes_motion_peak_noswap_fixed_axis_95.mp4`
- Before/after video: `camtest/line/demo_hrnet_enhanced_20260728/robust_3d_global_axes_motion_peak_noswap_20260729/comparison_wrong_xy_vs_global_xy_3d.mp4`
- Camera layout: `camtest/line/line_refine_global_axes_20260729/camera_layout_before_after.png`

## Important findings

The earlier low per-camera floor reprojection RMS was misleading because each
camera used image-relative floor XY directions. Cam1/cam2/cam3/cam4 required
global rotations of 0/90/180/270 degrees. HRNet also must not reuse the fixed
cam3/cam4 left-right swap previously found for YOLO Pose.

## Known remaining work

- Cam2–cam4 epipolar median remains higher (`9.24 px`; torso center `11.82 px`).
- Fix the latent Viterbi recovery bug for future frames where every camera subset fails.
- Add bounded-gap interpolation and reject implausible learned bone priors.
- Optionally refine small residual extrinsic errors using reliable torso trajectories.

## 論文規範

目前的固定門檻主要用於 demo 防止數值爆炸，不能直接視為經過驗證的
論文參數。投稿版本必須將每個 threshold 與 outlier 判斷改為可重現、
可解釋且在 test set 之前凍結的定義。

### 資料與實驗切分

- 將資料分成 calibration、validation、test sets；不得使用 test sequence
  調整 threshold、filter window 或 optimizer weight。
- 優先依受測者與錄影 session 分割，避免同一人物、動作或相機噪聲同時
  出現在參數選擇與最終評估中。
- 保存每次實驗的完整設定、random seed、軟體版本、相機參數與資料版本。

### Threshold 與 outlier 定義

- 2D confidence：使用人工標註 validation frames 建立 precision–recall 與
  error-versus-confidence curve，以預先指定的 precision/recall operating point
  選擇 threshold；必要時依 camera 與 joint 分開校準。
- 2D temporal outlier：以 bbox 對角線與 FPS 正規化 joint velocity，使用
  median 與 scaled MAD 定義異常值，不再使用固定 `170 px/frame`。
- Reprojection outlier：由每台相機、每個 joint 的 2D localization covariance
  建立 Mahalanobis residual，使用預先指定的 chi-square confidence region；
  不再以所有相機共用固定 `35 px` 作為論文設定。
- RANSAC：依目標成功機率、validation inlier ratio 與最小 sample size 推導
  iteration count；明確記錄 inlier views、失敗原因及 two-view fallback 條件。
- Triangulation quality：由 projection Jacobian 傳播 2D covariance 至 3D
  covariance，依允許的 3D uncertainty 判斷是否保留；ray angle 僅作輔助指標。
- 3D temporal outlier：使用 constant-velocity prediction innovation 與 covariance
  的 normalized innovation squared gate，不再使用固定 `0.95 m/frame`。
- Bone-length outlier：只用 calibration sequence 中低 reprojection error、
  三視角以上且低 uncertainty 的骨段估計 subject-specific median/MAD；
  prior 在 test set 固定，並分別報告 optimizer 前後結果。
- Scene bounds：依實際量測的 5.4 m 場地、人體高度與 calibration uncertainty
  設定，不再使用僅防數值爆炸的 `±10 m` 寬範圍。
- Ground contact：使用已知地板平面、foot height uncertainty 與垂直速度共同
  判定，不只依 ankle height percentile。

### 插值與平滑

- 設定以秒表示的最大 interpolation gap；超過上限必須保持 missing。
- 在輸出與評估中區分 observed、triangulated、interpolated、optimized joints，
  不得把長時間插值當成成功偵測。
- 以 validation set 比較 smoothing window，並同時量測 static jitter、MPJPE、
  motion amplitude attenuation 與 event timing error。

### Ground truth 與評估

- 若宣稱 absolute 3D accuracy，必須使用 marker-based motion capture、已知 3D
  rigid object、可靠 reference system 或其他獨立 3D ground truth。
- 若沒有獨立 3D ground truth，只能宣稱 multiview consistency、geometric
  consistency 與 temporal stability，不得將 reprojection error 當作 absolute
  pose accuracy。
- 最終至少報告 MPJPE、PA-MPJPE、joint coverage、reprojection median/P95、
  uncertainty、RANSAC inlier views、static jitter、bone-length variation、
  foot-to-floor error 與 action-event timing error。
- 每個 filter 必須輸出被拒絕的 camera、joint、frame、原始數值、threshold、
  rejection reason，以及被刪除、降權或插值後的處理結果。

### Ablation 與敏感度分析

- 分別移除 2D temporal filter、uncertainty/RANSAC、pose-level view selection、
  3D temporal gate 與 bone prior，報告各模組對 accuracy 和 coverage 的影響。
- 對主要 threshold 做多點 sweep 或至少 `±20%` 敏感度分析；論文不得只報
  validation set 上的最佳單一數值。
- 同時報告 accuracy–coverage trade-off。若小幅改變 threshold 就造成結果大幅
  改變，必須視為方法不穩定並進一步修正。

在上述程序完成前，現有 `0.25/0.30` confidence、`170 px/frame`、`35 px`
reprojection、`0.95 m/frame` temporal gate 與寬鬆骨長範圍均應標記為
`demo heuristic`，不可包裝成具有統計或生物力學保證的最終論文設定。

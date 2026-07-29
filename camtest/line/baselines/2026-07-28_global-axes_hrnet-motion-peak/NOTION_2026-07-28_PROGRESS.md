# 7/28 進度｜四相機 3D Pose Demo 正確 Baseline

<callout icon="✅" color="green_bg">
	已得到第一個經過幾何驗證、可正確呈現動作的四相機 3D pose baseline。
</callout>

## 本日完成

- 完成四相機 8-event motion peak affine synchronization。
- 確認地板標線外參先前使用了各相機畫面相對方向，而非同一個實體全域 XY。
- 使用實體 Center、+X、+Y 三個錨點重新統一世界座標。
- 全域方向修正：cam1 `0°`、cam2 `90°`、cam3 `180°`、cam4 `270°`。
- 確認 HRNet 不應套用 YOLO Pose 的 cam3/cam4 固定 left-right swap。
- 使用新全域外參、8-peak sync、HRNet no-swap 重跑 robust triangulation。

## 關鍵成果

<table fit-page-width="true" header-row="true">
	<tr>
		<td>指標</td>
		<td>修正前</td>
		<td>修正後</td>
	</tr>
	<tr>
		<td>Epipolar median</td>
		<td>31.13 px</td>
		<td>5.75 px</td>
	</tr>
	<tr>
		<td>Torso-center epipolar</td>
		<td>17.82 px</td>
		<td>4.94 px</td>
	</tr>
	<tr>
		<td>空 camera subset frames</td>
		<td>48</td>
		<td>0</td>
	</tr>
	<tr>
		<td>Temporal rejected joints</td>
		<td>2371</td>
		<td>0</td>
	</tr>
	<tr>
		<td>Optimizer 前異常骨長</td>
		<td>3728</td>
		<td>0</td>
	</tr>
	<tr>
		<td>Root Z range</td>
		<td>0.600 m</td>
		<td>0.062 m</td>
	</tr>
</table>

## 正確版本設定

- 2D pose：HRNet filtered tracks。
- Sync：開頭與結尾共八個張開／收回 velocity peaks，估計每台相機 affine time mapping。
- Extrinsics：地線 PnP + line refinement，再依實體三錨點統一 global XY。
- LR mapping：四台 HRNet 全部 no-swap。
- 3D：pose-level subset selection、joint RANSAC、temporal gate、skeleton optimization。
- 共 533 幀，每幀所有 11 種 camera subset 均能產生候選。

## 根因

先前每台相機的 floor reprojection RMS 雖低，但 P00–P08 的方向是依各自畫面推測，導致四台相機宣稱共用 `+X/+Y`，實際卻指向不同物理方向。這會讓單相機 RMS 看起來正常，但跨相機射線無法在同一世界空間相交。

## Baseline 路徑

`camtest/line/baselines/2026-07-28_global-axes_hrnet-motion-peak`

主要輸出：

- `triangulated_enhanced_robust.json`
- `triangulated_hrnet_global_axes_motion_peak_noswap_fixed_axis_95.mp4`
- `comparison_wrong_xy_vs_global_xy_3d.mp4`
- `camera_layout_before_after.png`

## 後續工作

- [ ] 修正 Viterbi 遇到全空候選後無法恢復的 latent bug。
- [ ] 對 interpolation 設定最大可補幀數，避免長遮擋被線性補滿。
- [ ] 限制 skeleton bone prior 在合理人體比例範圍。
- [ ] 針對 cam2–cam4 較高的 residual 做 torso-based extrinsic refinement。
- [ ] 將此 baseline 接入正式 demo 執行入口。

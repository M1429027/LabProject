# Karate Self-Calibration Pipeline

## 目標
本方向的核心目標是：

`在不依賴人工外參校正的前提下，從遠距、多視角空手道影片中重建兩位選手的 3D 骨架，並進一步延伸到 SMPL 與物理合理化。`

相較於目前以傳統多視角校正為主的流程，新方向的重點不再只是「校正後做 triangulation」，而是把研究焦點轉向：

- 無人工外參的跨視角身份匹配
- 多幀自校正
- 遮擋下的 3D 骨架恢復
- 後端人體模型與物理約束整合

---

## 問題設定

### 輸入條件
- 多台固定相機拍攝空手道比賽或訓練影片
- 相機固定不動
- 影片時間同步或至少接近同步
- 內參最好可先取得，或至少知道焦距、解析度、畸變近似值
- 場地範圍固定，例如榻榻米區域
- 畫面中通常是兩位選手，偶爾有裁判或旁人

### 關鍵限制
- 不希望手動量測外參
- 空手道服裝高度相似，不能只靠外觀做跨視角 ReID
- 場景存在快速踢擊、轉身、交錯與遮擋
- 遠距拍攝下，2D keypoint 容易抖動或遺失

### 核心研究命題
`嚴重遮擋的空手道場景下，如何在不依賴人工外參校正的情況下，完成跨視角身份匹配、自校正與 3D 骨架重建。`

---

## 完整 Pipeline

```text
Multi-view Karate Videos
-> Person Detection
-> 2D Pose Estimation + Heatmaps
-> Single-view Tracking
-> Cross-view Matching without Extrinsics
-> Multi-frame Self-Calibration
-> Weighted 3D Triangulation
-> 3D Pose Refinement
-> SMPL Fitting
-> Physics-based Refinement
-> Final 3D Karate Motion
```

---

## Transformer / Refinement 共識

### 核心定位
Transformer 不作為從零解出整個系統的主模型，而是後端的 refinement 模組。

也就是說，前面仍然要先透過可解釋的 matching、自校正與 triangulation 產生：

- cross-view matched tracklets
- rough camera extrinsics
- rough 3D pose
- 2D keypoints / heatmaps / confidence / visibility

Transformer 的任務是根據這些中間結果進行修正，而不是直接從原始多視角影片中憑空推論所有外參與 3D pose。

### 為什麼這樣設計
目前實驗顯示，單視角 2D-to-3D lifting 在空手道 close-contact 場景下不穩：

- 單人時可得到基本人體形狀，但仍有前後方向 ambiguity。
- 多人遮擋與近身互動時，VideoPose3D 的 3D pose 明顯退化。
- 直接使用 lifted skeleton 的 mean forward vector 做跨視角 matching 會誤導分數。

因此，lifting 目前只能當作弱 cue / 輔助訊號，不應該成為整個 pipeline 的主判斷來源。

### 建議驗證順序
後續應該採用逐步驗證，而不是一開始就把所有問題交給 Transformer：

1. 先完成可解釋的 cross-view matching baseline。
2. 再做 multi-frame self-calibration，得到 rough extrinsics。
3. 再用 rough extrinsics 做 weighted triangulation，得到 rough 3D pose。
4. 先做 optimization-based refinement baseline：
   - reprojection loss
   - bone length consistency
   - temporal smoothness
   - camera regularization
5. 最後再加入 small Transformer 做 refinement。

### Transformer 第一版目標
第一版 Transformer 建議只做 3D pose refinement，不先同時修 camera。

**輸入**
- rough 3D joints
- multi-view 2D keypoints
- heatmaps / confidence
- visibility pattern
- track / view metadata

**輸出**
- refined 3D joints

等 3D pose refinement 有明確改善後，再考慮第二版輸出 camera residual：

```text
refined_camera = rough_camera + predicted_residual
```

### 成功標準
Transformer 是否值得加入，至少要能在 optimization baseline 之上帶來可觀察改善：

- 3D pose 更平滑且形狀更合理
- reprojection error 不惡化
- 遮擋 frame 的 joint recovery 變好
- 不需要人工外參即可穩定 refine rough result

### 目前結論
Transformer 方向可行，但前提是它只做 refinement。

本研究的主線應是：

`先用可解釋的 tracking / matching / self-calibration 產生 rough camera + rough 3D，再讓 Transformer 修正。`

不能把前端 matching 與 self-calibration 的錯誤全部留給 Transformer 處理。

---

## Review 使用方式

為了讓這份文件能持續作為實作中的 review 規格，後面每個 stage 都統一用下面幾個欄位描述：

- `目標`：這一步要解的核心問題
- `目前設計`：現在決定採用的主做法
- `第一版 baseline`：現階段先做得動、可驗證的版本
- `輸入 / 輸出`：之後模組介面要固定的資料格式
- `升級方向`：未來如果效果不夠，優先加什麼
- `Review / Debug 重點`：之後回頭檢查時要先看哪裡

這樣你之後每次改設計時，只要更新對應 stage 的這幾個欄位即可。

---

## Stage 0：多視角影片輸入

### 目標
建立後續多視角人體分析的原始資料來源，並把資料取得條件固定下來。

### 目前設計
- 多台固定相機拍攝空手道訓練或比賽影片
- 以固定場地、固定機位為前提
- 允許沒有人工外參，但內參盡量先有

### 第一版 baseline
- 相機固定
- 影片時間大致同步
- 內參可先用近似值或簡化校正取得
- 場地範圍固定，例如榻榻米區

### 輸入 / 輸出
**輸入**
- 多視角原始影片

**輸出**
- 後續各 stage 共用的影片清單與 view_id

### 升級方向
- 若之後真的做遠距大場地版本，可再加入：
  - 場地線先驗
  - camera metadata
  - 更精準的時間同步

### Review / Debug 重點
- 相機是否固定不動
- 影片是否同一段時間
- 場地範圍是否一致
- 內參是否至少有可用近似值

---

## Stage 1：單視角人物偵測與 2D 骨架估計

### 目標
在每個視角上獨立得到可供 tracking 與後續 matching 使用的 2D 人體觀測。

### 目前設計
目前保留兩條前端：

1. `YOLO pose`
   - 快速 baseline
   - 直接輸出 bbox + keypoints + confidence

2. `YOLO detector -> HRNet top-down pose`
   - 主研究線的 2D frontend
   - 保留 top-down 架構
   - 可以輸出 keypoints、joint confidence、heatmap

### 第一版 baseline
- 單人 demo：先用 `YOLO -> HRNet top-down + heatmap`
- 保留 `YOLO pose` 當快速 baseline 與 fallback

### 輸入 / 輸出
**輸入**
- 原始多視角影片

**輸出**
- `keypoints_<view>.json`
- `run_summary.json`
- 若為 HRNet 路線，額外輸出 `heatmaps/<view>/...npz`

每個人的資訊至少包含：
- `bbox`
- `keypoints`
- `keypoint confidence`
- `heatmap_path`（若有）

### 升級方向
- `YOLO + ViTPose`
- 更穩的多人 detector
- 遠距版本的高解析輸入策略

### Review / Debug 重點
- 先看 bbox 是否穩
- 再看 keypoints 是否合理
- 再看 heatmap 是否真的寫出來
- 若多人時 keypoints 很亂，優先懷疑 detector 與 bbox crop，而不是先怪後端 3D

### 備註
這一步是整個系統的地基。  
如果 2D pose 在這裡就錯很多，後面的 tracking、matching、self-calibration、3D 都只是在補救錯的輸入。

---

## Stage 2：單視角 Tracking

### 目標
在每個 camera view 內穩定追蹤同一位選手，產生單視角 tracklets。

### 目前設計
目前已實作一個 `pose-aware single-view tracker`，不是完整 ReID tracker，而是混合：
- bbox IoU
- pose similarity
- center continuity

再用 Hungarian matching 去做每幀配對。

### 第一版 baseline
- 單人：保證能穩定維持單一 `track_id`
- 簡單雙人：先看是否能在不嚴重交錯下維持穩定 ID

### 輸入 / 輸出
**輸入**
- `keypoints_<view>.json`

**輸出**
- `tracks_<view>.json`
- `tracked_<view>.mp4`
- `run_summary.json`

每個人的資訊會保留：
- `bbox`
- `keypoints`
- `confidence`
- `heatmap_path`
- `track_id`
- `tracking_score`

### 升級方向
- track reactivation
- 更長遮擋的恢復邏輯
- appearance / ReID feature
- segmentation 或 optical flow 輔助

### Review / Debug 重點
- 先看 detection 本身是否穩
- 再看 `tracks_<view>.json` 的 `track_id` 有沒有跳
- 再看 `tracked_<view>.mp4` 肉眼是否一致
- 若雙人時開始亂，先調：
  - `match_threshold`
  - `lost_track_buffer`
  - `iou_weight / pose_weight / center_weight`

### 備註
目前這版足夠先把單人與簡單雙人流程打通，但未來多人對打時，大概率還要再加技術，ReID 可能是其中之一，但不應該是唯一訊號。

---

## Stage 3：不依賴外參的跨視角 Matching

### 目標
建立跨相機的身份對應：

`view A 的選手 1 <-> view B 的同一位選手`

### 目前設計
這一步不打算只靠外觀 ReID，而是採兩段式：

1. `coarse matching`
   - temporal continuity
   - bbox / position prior
   - 單視角 tracklets
   - 少量 appearance

2. `skeleton-direction refinement`
   - 先做簡單各視角 2D-to-3D lifting
   - 估骨架方向向量
   - 比較跨視角 rotation consistency
   - 用來修正第一階段的粗配對

### 第一版 baseline
- 先限制場景為兩位選手
- coarse matching 先做得動
- 再把 `skeleton direction consistency` 補成 refinement signal

### 輸入 / 輸出
**輸入**
- 各視角 `tracks_<view>.json`
- 每個人的 keypoints / bbox / confidence / heatmap

**輸出**
- 跨視角 candidate matches
- 最終 cross-view match table

### 升級方向
- appearance embedding
- clip-level temporal embedding
- segmentation-aware matching
- 更完整的 coarse 3D consistency score

### Review / Debug 重點
- 單視角 `track_id` 若不穩，這一步先不要往下怪
- 先看 coarse matching 錯在哪裡
- 再看 skeleton-direction refinement 是否真的有把錯誤配對降分
- 如果空手道服裝太像，優先相信 temporal / skeleton，不要先過度依賴外觀

### 角色
這是第一個核心研究點。

---

## Stage 4：自校正相機外參

### 目標
不用人工標定板，從跨視角 matched poses 中估計相機相對姿態。

### 目前設計
保留兩條方法，但第一版優先以幾何法為主：

1. 幾何法
   - `fundamental matrix`
   - `essential matrix`
   - 多幀高可信 correspondences

2. 人體模型輔助法
   - 先有粗 lifting / 粗 3D
   - 再反推相機相對關係

### 第一版 baseline
- 以 `matched multi-view 2D keypoints` 為主
- 多幀累積
- 高可信 joints 過濾
- robust optimization 求初始 `R, t`

### 輸入 / 輸出
**輸入**
- cross-view matched tracklets
- 高可信 2D keypoints
- 多幀時間窗

**輸出**
- initial relative camera poses
- 可供 Stage 5 使用的自校正結果

### 升級方向
- 融合 coarse 3D 人體先驗
- 引入場地平面與場地線
- bundle refinement

### Review / Debug 重點
- 不要用單幀直接下結論
- 先看高可信 joints 篩掉之後還剩多少資訊
- 若相機關係很飄，先回頭檢查 Stage 3 的 matching 是否有錯
- 遠距拍攝時，先看 2D keypoints noise 是否已經太高

### 角色
這是第二個核心研究點。

---

## Stage 5：粗略 3D 骨架重建

### 目標
在估出相機相對姿態後，先做一版 rough 3D skeleton。

### 目前設計
- weighted triangulation
- confidence-based fusion
- 後處理式 temporal smoothing
- 簡單骨長約束

### 第一版 baseline
- 不追求最終品質
- 只要能從 self-calibration 的相機結果得到「可用的粗 3D 初值」

### 輸入 / 輸出
**輸入**
- multi-view matched 2D keypoints
- estimated camera poses

**輸出**
- rough 3D skeleton sequence

### 升級方向
- 更強的 outlier filtering
- 每關節 confidence fusion
- 更穩的 temporal prior

### Review / Debug 重點
- 如果粗 3D 整體很飄，先看相機關係
- 若只有局部關節亂跳，先看 2D keypoint consistency
- 不要把這一步的輸出當最終結果，它是 refinement 的初始值

---

## Stage 6：3D Skeleton Refinement

### 目標
把粗 3D skeleton 變穩，減少抖動與幾何不合理。

### 目前設計
第一版先走 optimization-based refinement，不直接上 Transformer。

### 第一版 baseline
最小版本先做：

`reprojection loss + bone length loss + smoothness loss + pose prior`

### 輸入 / 輸出
**輸入**
- rough 3D skeleton
- multi-view 2D keypoints
- 若有，heatmap / confidence

**輸出**
- refined 3D skeleton

### 升級方向
- heatmap-aware loss
- uncertainty weighting
- transformer-based refinement

### Review / Debug 重點
- 如果 refine 後變差，先看 loss 權重是否失衡
- 若 temporal 變平但姿勢失真，可能 smoothness 太強
- 若 head / wrist 常亂，優先檢查 2D pose 與 heatmap

---

## Stage 7：SMPL Fitting

### 目標
把 3D skeleton 對齊到完整人體模型。

### 目前設計
這一步先作為後續延伸，不是第一版主線。

### 第一版 baseline
- 先保留介面
- 先不阻塞前面 `2D -> tracking -> matching -> self-calibration -> rough 3D`

### 輸入 / 輸出
**輸入**
- refined 3D skeleton
- multi-view 2D keypoints

**輸出**
- SMPL pose `theta`
- SMPL shape `beta`
- global translation

### 可用工具參考
- `SMPLify-X`
- `VPoser`
- `HybrIK`
- `CLIFF`
- `WHAM`
- `PIXIE`

### 升級方向
- foot contact loss
- better pose prior
- shape prior tuning

### Review / Debug 重點
- 若 3D skeleton 本身不穩，這一步不要急著做
- 先確認前面 refined skeleton 夠穩，再接 SMPL

---

## Stage 8：Physics-based Refinement

### 目標
修正 SMPL 與骨架中的物理不合理現象，例如：

- 腳滑
- 腳穿地板
- 人與人穿模
- 關節不合理
- 重心不穩
- 動作抖動

### 目前設計
這一步完全放後期，不列入前段 MVP。

### 第一版 baseline
- 不做完整 physics engine
- 若真的要補，只先加簡單物理約束

### 建議分層實作
1. 簡單物理約束
   - foot-ground constraint
   - joint limit
   - velocity smoothness
2. 人體碰撞修正
   - 避免兩位選手 SMPL mesh 穿透
3. 完整 physics simulation
   - `MuJoCo`
   - `Isaac Gym`
   - `Nimble`
   - `Brax`

### 升級方向
- SMPL body collision
- humanoid dynamics optimization

### Review / Debug 重點
- 這一步若過早做，會拖慢主線
- 除非前面 `rough 3D / refined 3D / SMPL` 已穩，否則先不要投入

---

## 研究重點建議

不建議把重點放成：
- 我做了一個超強 SMPL 模型
- 我做了完整的 physics engine

更聚焦、也更有研究價值的主題應該是：

`嚴重遮擋空手道場景下，不依賴人工外參校正的跨視角身份匹配與自校正 3D 骨架重建`

### 賣點
- 不需要手動外參校正
- 適用空手道這類相似服裝、高遮擋、快速移動場景
- 用 skeleton consistency 與 temporal consistency 做跨視角 matching
- 用多幀自校正估計相機關係
- 後端以 SMPL / physics prior 提升合理性

---

## 模組可行性總評

### 核心評估表
| 模組                         | 難度   | 可行性   | 實作建議                                             |
|----------------------------|--------|----------|------------------------------------------------------|
| 多視角影片擷取              | 中 | 高        | 先確保同步與固定機位                                  |
| 2D pose estimation        | 低到中  | 高        | 優先使用現成模型快速建立 baseline                      |
| 單視角 tracking            | 中   | 中高       | 先用 `ByteTrack` 或 `BoT-SORT`               |
| 無外參 cross-view matching | 高    | 中        | 第一個核心研究點，先限制為兩位選手                  |
| 自校正外參                  | 高  | 中低到中    | 建議用多幀累積，不要用單幀估計                     |
| 粗 3D triangulation        | 中     | 中        | 強烈依賴前面 matching 與自校正品質              |
| 3D refinement             | 中      | 中        | 先做 optimization-based refinement，再考慮 Transformer |
| SMPL fitting              | 中高    | 中高       | 可直接借助現成工具鏈整合                                 |
| Physics refinement        | 很高    | 低到中      | 放在後期，不要作為第一版主線                              |

### 優先順序建議

#### 第一優先：高可行、可快速落地
- 多視角影片擷取
- 2D pose estimation
- 單視角 tracking

#### 第二優先：研究核心、決定系統價值
- 無外參 cross-view matching
- 自校正外參
- 粗 3D triangulation

#### 第三優先：提升結果穩定性與可用性
- 3D refinement
- SMPL fitting

#### 第四優先：高風險進階模組
- Physics refinement

### 總結判斷
- 第一版最值得投入的主線，不是 physics，也不是完整 SMPL avatar，而是先把 `2D -> tracking -> matching -> self-calibration -> rough 3D` 打通。
- 若前半段沒有穩定，後面的 refinement、SMPL 與 physics 只會建立在不穩定的 3D 上，整體效果不會漂亮。
- 因此實作上應優先追求：`可運作的多視角匹配與自校正骨架重建 baseline`，再逐步加上人體模型與物理約束。

---

## 實作優先順序

### 第一階段：先把基礎流程打通
1. Multi-view video ingestion
2. Single-view person detection
3. Single-view 2D pose estimation
4. Single-view tracking

### 第二階段：建立無外參多視角核心
5. Cross-view matching without extrinsics
6. Multi-frame self-calibration
7. Weighted triangulation

### 第三階段：讓 3D 更穩
8. 3D pose refinement
9. SMPL fitting

### 第四階段：進階研究延伸
10. Physics-based refinement

---

## 第一版建議成功標準

第一版不追完整 physics，也不追 end-to-end SOTA。  
較合理的成功標準是：

- 能在兩位選手場景下穩定做單視角 tracking
- 能完成基本跨視角 matching
- 能從多幀觀測估出可用的相機相對關係
- 能重建粗 3D skeleton
- 在遮擋場景下，相較雙視角 baseline，能恢復更多有效關節

---

## 目前實作狀態：3B 到 4C 前

本節記錄目前從 Stage 3B 到 Stage 4C 前的實作狀態。此段落用於後續 review 與報告撰寫，重點放在已完成項目、驗證結論，以及下一步不應再停留在單純 filter 的原因。

### Stage 3B：跨視角身份匹配

目前 Stage 3B 已完成從 pairwise matching 到 geometry-based hypothesis selection 的銜接。系統不再直接相信兩兩 track 的局部分數，而是先產生全域身份假設，再以多視角幾何一致性決定最終身份分組。

#### Pairwise scoring

Pairwise scoring 的任務是產生候選，而不是決定最終答案。它會對不同 view 的 track pair 計算多個局部相似度：

- `pose_shape`：比較 root-centered、scale-normalized 2D skeleton shape。
- `motion_prior`：比較 track root 的移動節奏與 motion energy sequence。
- `visibility`：比較 joint confidence 轉換出的 visible / not visible pattern。
- `temporal`：檢查兩條 track 是否有足夠時間重疊。
- `skeleton_consistency`：使用 VideoPose3D lifting 產生的 forward / spine / frontality descriptor 作為弱 cue。
- `appearance`：保留外觀特徵介面，但因空手道白色道服相似，目前不作為主決策訊號。

目前結論是：`pose_shape`、`motion_prior`、`visibility` 在空手道對打中容易同時對錯誤配對給出合理分數，因為兩位選手本來就會有相近走位、同步進退與互相遮擋。因此 pairwise scoring 的定位是候選生成，不是 identity 定義。

#### Global identity hypotheses

在四視角、兩位主要選手的設定下，系統會固定 anchor view 的 local identity order，並枚舉其他 view 是否交換 local track order。若共有四個 view，則會形成 `2^3 = 8` 種 global identity hypotheses。

此步驟將問題從「單一 pair 是否相似」提升為「整組跨視角身份分配是否一致」。

#### Geometry validation

Geometry validation 會對每個 hypothesis 收集同一 identity、同一 frame、同一 joint id 的跨視角 2D correspondence，接著估計 fundamental matrix 並用 RANSAC 排除 outliers。

評分標準包含：

- `inlier_ratio`：符合該 hypothesis 幾何關係的 correspondence 比例。
- `median_sampson_error`：對應點相對於 epipolar geometry 的中位誤差。
- `geometry_score`：綜合 inlier ratio 與 Sampson error 的排序分數。

目前 `selected_hypothesis.json` 已成為 Stage 3B 到 Stage 4A 的正式輸出介面。這代表跨視角 identity selection 的主判斷已從 2D similarity 轉移到 geometry consistency。

### Stage 4A：Rough extrinsics 與 rough triangulation

Stage 4A 使用 `selected_hypothesis.json` 收集跨視角對應點，並以 essential matrix baseline 估計 pairwise relative pose。此階段可估計相機間的相對旋轉與 translation direction，但無法單靠 essential matrix 決定 metric scale。

#### Cheirality sign check

由 essential matrix recover pose 時，translation direction 存在 `t` 與 `-t` 的歧義。系統加入 cheirality check，比較兩個方向下 triangulated points 位於相機前方的比例。

目前結果顯示原始 `t` 的 positive-depth ratio 明顯優於 `-t`，因此目前主要問題不是 translation sign 反轉。

#### Translation scale refinement

translation scale refinement 用於把 pairwise translation directions 對齊成 anchor-based camera graph。此步驟可以檢查與改善相機方向一致性，但不能單獨恢復真實尺度。

目前結論是：pairwise direction consistency 可用，但 absolute scale 需要額外 prior 或後續 optimization。

#### Human scale prior

human scale prior 使用 median human height 作為全域尺度約束。系統會估計 3D skeleton 中 nose 到 ankle midpoint 的高度，並將其對齊到預設人體高度。

此步驟能把人物與相機尺度拉回合理範圍，但它只修正 global scale，無法保證每個 joint 都合理。

### Stage 4B：Oracle geometry validation branch

Stage 4B 是驗證分支，不是 self-calibration 主方法。它使用 dataset / COLMAP 提供的相機 intrinsics 與 extrinsics 作為 oracle geometry，用來回答：

`如果相機幾何是正確的，目前 matching 與 triangulation backend 是否能產生可用 3D？`

目前 4B 已完成其主要驗證任務：

- oracle geometry 下 reprojection error 明顯降低。
- 3D 結果能看出人體與動作。
- 仍存在 joint 缺失、跳動與局部不完整。

因此 4B 給出的結論是：目前問題不只來自 self-calibration 外參，還包含 2D observation quality、joint-level triangulation stability，以及缺乏 pose-level prior。4B 已足夠作為 refinement 前的上限參考，不需要在此階段繼續擴成主方法。

### Refinement 前的 filter 與 constraint

目前在正式 pose refinement 前，已加入以下 baseline filter / constraint：

- inlier-aware triangulation
- per-joint view subset selection
- minimum triangulation angle check
- human scale prior
- bone length prior
- bone outlier filtering
- temporal smoothing / completion
- oracle geometry baseline

#### Joint-level triangulation quality

joint-level triangulation quality 會針對每個 frame、identity、joint 枚舉可用 view subset。每個 subset 會根據 reprojection error、triangulation angle、joint confidence 與 dropped-view penalty 評分，最後選出最穩定的 joint 3D estimate。

此步驟能降低 reprojection error，但仍然是 per-joint decision。它無法保證整副 skeleton 具備人體結構。

#### Bone prior

bone prior 會估計每個 identity 的 median bone length，並將偏離過大的 bone length 以 soft correction 拉回穩定範圍。此步驟能降低骨長波動，但不是完整人體模型。

#### Bone outlier filter

bone outlier filter 專門處理偶發超長骨頭。當 bone length 超過 median length 的倍率門檻時，系統會 clamp 該骨頭並做簡單 temporal smoothing。

此步驟能降低蜘蛛狀爆炸，但不能從結構錯誤的 3D 點中重新推論出完整人體。

### 目前結論：應進入 Stage 4C

目前可以確認，4B 能做的驗證工作已完成到合理段落，refinement 前的 filter 與 constraint 也已達到 baseline 上限。繼續堆疊更多 filter 可能讓序列更平滑，但不一定讓結果更像人，甚至可能抹掉空手道快速動作。

下一步應進入：

`Stage 4C：optimization-based pose refinement`

Stage 4C 的目標不是修單一 joint 或單一 bone，而是以一個 identity 的整副 skeleton 和短時間 window 為單位進行最佳化。建議 loss 包含：

- `reprojection loss`
- `bone length consistency`
- `left-right symmetry`
- `temporal smoothness`
- `joint confidence weighting`
- `robust outlier loss`

此階段應作為 Transformer 前的可解釋 baseline。若 optimization-based refinement 無法建立合理人體，直接進入 Transformer 會使錯誤來源變得過度黑盒，難以判斷問題來自 matching、camera、triangulation 或 pose prior。

---

## 總結
這個新方向的重心不是傳統校正後 triangulation，而是：

- 無人工外參的跨視角 matching
- 多幀自校正
- 遮擋下的 3D skeleton recovery
- SMPL 與物理先驗的後端整合

如果要以專題或研究實作方式推進，最合理的策略是：

`先把 2D -> tracking -> matching -> self-calibration -> rough 3D 打通，再逐步加上 refinement、SMPL 與 physics。`

這樣不只題目聚焦，也比較能控制風險與進度。

from ultralytics import YOLO
import cv2
import os
import json

# 1. 載入模型
print("正在載入模型，請稍候...")
model = YOLO('yolo11l-pose.pt')

# 2. 設定影片路徑
video_path = "rendervideo/output_backR.mp4" 

# 檢查檔案是否存在
if not os.path.exists(video_path):
    print(f"❌ 錯誤：找不到檔案 '{video_path}'")
    print("請確認資料夾名稱 'rendervideo' 和檔名是否正確。")
    exit()

cap = cv2.VideoCapture(video_path)

# 檢查影片有沒有成功打開
if not cap.isOpened():
    print("❌ 無法讀取影片，請檢查路徑或檔案格式。")
    exit()

# 取得原始影片的 寬、高、FPS
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 設定輸出檔名
output_video_path = "rendervideo/yolo_output_backR.mp4"
output_json_path = "rendervideo/output_backR.json"  # 🔥 新增：JSON 輸出路徑

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

print(f"🚀 開始辨識！")
print(f"   影片將儲存為: {output_video_path}")
print(f"   骨架資料將儲存為: {output_json_path}")

# 🔥 新增：準備儲存所有幀的骨架資料
all_frames_data = []

# 3. 逐格讀取影片並辨識
frame_idx = 0
while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("\n✅ 影片處理完成！")
        break

    # 4. 讓 YOLO 辨識
    results = model(frame, show=False, conf=0.5)

    # 取出畫好骨架的圖
    annotated_frame = results[0].plot()
    
    # 寫入新的影片檔
    out.write(annotated_frame)

    # 🔥 新增：提取骨架關鍵點座標
    frame_data = {
        "frame": frame_idx,
        "people": []
    }
    
    # results[0].keypoints 包含所有偵測到的人的關鍵點
    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.data.cpu().numpy()  # 轉成 numpy array
        
        for person_idx, person_kpts in enumerate(keypoints):
            person_data = {
                "person_id": person_idx,
                "keypoints": []
            }
            
            # 每個人有 17 個關鍵點 (COCO格式)
            # 格式: [x, y, confidence]
            for kpt_idx, kpt in enumerate(person_kpts):
                x, y, conf = float(kpt[0]), float(kpt[1]), float(kpt[2])
                person_data["keypoints"].append({
                    "id": kpt_idx,
                    "x": x,
                    "y": y,
                    "confidence": conf
                })
            
            frame_data["people"].append(person_data)
    
    all_frames_data.append(frame_data)

    # 印個點點當進度條
    print(".", end="", flush=True)
    frame_idx += 1

# 5. 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()

# 🔥 新增：將骨架資料存成 JSON
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump({
        "video_info": {
            "width": w,
            "height": h,
            "fps": fps,
            "total_frames": frame_idx
        },
        "keypoint_names": [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ],
        "frames": all_frames_data
    }, f, indent=2)

print(f"\n🎉 大功告成！")
print(f"   影片檔案：{output_video_path}")
print(f"   骨架資料：{output_json_path}")
print(f"   總共處理了 {frame_idx} 幀")
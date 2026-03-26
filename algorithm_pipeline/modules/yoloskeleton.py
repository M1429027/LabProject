from ultralytics import YOLO
import cv2
import os
import json
import glob

def process_video(video_path, model, output_folder, swap_left_right=False):
    """
    處理單一影片並生成骨架 JSON
    Args:
        swap_left_right: 是否交換左右關鍵點 ID (針對背面視角)
    """
    if not os.path.exists(video_path):
        print(f"Error: File not found {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # 取得影片資訊
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 設定輸出檔名
    filename = os.path.basename(video_path)
    filename_no_ext = os.path.splitext(filename)[0]
    
    output_video_path = os.path.join(output_folder, f"yolo_output_{filename}")
    output_json_path = os.path.join(output_folder, f"keypoints_{filename_no_ext}.json")
    
    # 準備影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    print(f"Processing: {filename}")
    print(f" -> Video: {output_video_path}")
    print(f" -> JSON:  {output_json_path}")
    if swap_left_right:
        print(" -> Mode: Back-view Left/Right Swap Enabled")

    all_frames_data = []
    frame_idx = 0
    
    # 定義左右交換的對應表 (COCO Format)
    # Left: [1, 3, 5, 7, 9, 11, 13, 15] -> Right: [2, 4, 6, 8, 10, 12, 14, 16]
    # Note: Index in list is ID (0-16)
    swap_indices = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # YOLO 推論
        results = model(frame, show=False, conf=0.5, verbose=False)
        
        # 繪製骨架圖
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        # 提取數據
        frame_data = {
            "frame": frame_idx,
            "people": []
        }
        
        if results[0].keypoints is not None:
            # 轉成 numpy array (People, 17, 3)
            keypoints = results[0].keypoints.data.cpu().numpy()
            
            for person_idx, person_kpts in enumerate(keypoints):
                person_data = {
                    "person_id": person_idx,
                    "keypoints": []
                }
                
                # 如果開啟交換模式，針對這個人的所有點進行重排序
                if swap_left_right:
                    # 使用 numpy 直接根據索引交換整列
                    person_kpts = person_kpts[swap_indices]

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

        # 簡單進度顯示
        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{total_frames}", end='\r')
        frame_idx += 1

    cap.release()
    out.release()
    
    # 存 JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "source_video": filename,
                "width": w, 
                "height": h, 
                "fps": fps, 
                "total_frames": frame_idx,
                "swapped_left_right": swap_left_right
            },
            "keypoint_names": [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ],
            "frames": all_frames_data
        }, f, indent=2)
    
    print(f"\nDone: {filename}")

def main():
    # 1. 設定路徑
    input_folder = "rendervideo"
    model_name = 'yolo11l-pose.pt'
    
    # 檢查資料夾
    if not os.path.exists(input_folder):
        print(f"Error: Folder '{input_folder}' not found.")
        return

    # 2. 載入模型
    print(f"Loading model {model_name}...")
    model = YOLO(model_name)

    # 3. 搜尋所有 MP4
    video_files = glob.glob(os.path.join(input_folder, "*.mp4"))
    
    # 過濾掉已經是輸出結果的檔案 (包含 'yolo_output_')
    target_videos = [f for f in video_files if "yolo_output_" not in os.path.basename(f)]
    
    if not target_videos:
        print("No videos found to process.")
        return

    print(f"Found {len(target_videos)} videos to process.")

    # 4. 逐一處理
    for video_path in target_videos:
        # 判斷是否需要開啟「背面左右交換」
        # 簡單判斷：如果檔名含有 'back' 且含有 'left' 或 'right'，通常需要交換
        # 你可以根據你的檔名規則自定義這裡的邏輯
        filename = os.path.basename(video_path).lower()
        
        # 預設不交換
        need_swap = False
        
        # 如果你想自動對背面視角開啟交換，可以取消下面註解並修改條件：
        # if "back" in filename:
        #     need_swap = True
        
        process_video(video_path, model, input_folder, swap_left_right=need_swap)

    print("\nAll videos processed successfully.")

if __name__ == "__main__":
    main()
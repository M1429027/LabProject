import cv2
import os
import natsort

def pngs_to_video(frames_dir, output_path, fps=120):
    # 取得所有 PNG 檔案
    images = [img for img in os.listdir(frames_dir) if img.endswith(".png")]
    
    if not images:
        print(f"Skip: No PNG files found in {frames_dir}")
        return

    # 依名稱排序 (支援數字順序)
    images = natsort.natsorted(images)

    # 讀取第一張圖以取得影像尺寸
    first_frame_path = os.path.join(frames_dir, images[0])
    first_frame = cv2.imread(first_frame_path)
    
    if first_frame is None:
        print(f"Error: Failed to read the first image {first_frame_path}")
        return

    height, width, _ = first_frame.shape

    # 建立影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 寫入每一張影格
    frame_count = len(images)
    print(f"Processing {frames_dir} -> {output_path} ({frame_count} frames)")
    
    for img_name in images:
        frame_path = os.path.join(frames_dir, img_name)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}, skipping.")
            continue
            
        out.write(frame)

    out.release()
    print(f"Finished: {output_path}")

def batch_process_folders(input_root, output_root, fps=120):
    # 檢查輸入目錄是否存在
    if not os.path.exists(input_root):
        print(f"Error: Input directory '{input_root}' does not exist.")
        return

    # 建立輸出目錄
    os.makedirs(output_root, exist_ok=True)

    # 遍歷輸入目錄下的所有項目
    items = os.listdir(input_root)
    # 排序以確保處理順序固定
    items = natsort.natsorted(items)

    for item in items:
        item_path = os.path.join(input_root, item)

        # 只處理資料夾
        if os.path.isdir(item_path):
            # 設定輸出影片檔名 (與資料夾同名)
            video_name = f"{item}.mp4"
            output_path = os.path.join(output_root, video_name)

            # 執行轉換
            pngs_to_video(item_path, output_path, fps)

if __name__ == "__main__":
    # 設定輸入與輸出路徑
    input_folder = "visualization_output_4cams"
    output_folder = "rendervideo"
    
    print("Start batch processing...")
    batch_process_folders(input_folder, output_folder, fps=120)
    print("All tasks completed.")
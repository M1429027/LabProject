import cv2
import os
import natsort

def pngs_to_video(frames_dir, output_path="output.mp4", fps=120):
    # 取得所有 PNG 檔案
    images = [img for img in os.listdir(frames_dir) if img.endswith(".png")]
    images = natsort.natsorted(images)  # 依名稱排序 (支援數字順序)

    if not images:
        raise ValueError("找不到任何 PNG 檔案")

    # 讀取第一張圖以取得影像尺寸
    first_frame = cv2.imread(os.path.join(frames_dir, images[0]))
    height, width, _ = first_frame.shape

    # 建立影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 儲存為 MP4 格式
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 寫入每一張影格
    for img_name in images:
        frame = cv2.imread(os.path.join(frames_dir, img_name))
        out.write(frame)

    out.release()
    print(f"✅ 成功輸出影片：{output_path}")

if __name__ == "__main__":
    # 例：把 ./frames 裡的 PNG 轉成 output.mp4
    pngs_to_video(r"visualization_output_4cams/camera4_back_right", "output.mp4", fps=120)

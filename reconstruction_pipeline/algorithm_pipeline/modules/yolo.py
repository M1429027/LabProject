from ultralytics import YOLO
import cv2
import os

# 1. 載入模型
print("正在載入模型，請稍候...")
model = YOLO('yolo11l-pose.pt')

# 2. 設定影片路徑
video_path = "rendervideo/pyrenderoutput.mp4" 

# 檢查檔案是否存在 (多加這行保險一點)
if not os.path.exists(video_path):
    print(f"❌ 錯誤：找不到檔案 '{video_path}'")
    print("請確認資料夾名稱 'rendervideo' 和檔名是否正確。")
    exit()

cap = cv2.VideoCapture(video_path)

# 檢查影片有沒有成功打開
if not cap.isOpened():
    print("❌ 無法讀取影片，請檢查路徑或檔案格式。")
    exit()

# --- 🔥 新增：準備錄影機 (VideoWriter) ---
# 取得原始影片的 寬、高、FPS，這樣輸出的影片才會跟原本一樣
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 設定輸出檔名 (存在跟原影片一樣的資料夾)
output_path = "rendervideo/output_with_pose.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 設定編碼
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

print(f"🚀 開始辨識！結果將儲存為: {output_path}")

# 3. 逐格讀取影片並辨識
while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("\n✅ 影片處理完成！")
        break

    # 4. 讓 YOLO 辨識
    # show=False: 關閉視窗避免報錯
    # save=False: 我們下面自己存，所以這裡不用自動存
    results = model(frame, show=False, conf=0.5)

    # --- 🔥 新增：把骨架畫上去並存入影片 ---
    # 取出畫好骨架的圖
    annotated_frame = results[0].plot()
    
    # 寫入新的影片檔
    out.write(annotated_frame)

    # 印個點點當進度條
    print(".", end="", flush=True)

# 5. 釋放資源 (重要！out.release() 沒做影片會打不開)
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n🎉 大功告成！請去查看檔案：{output_path}")
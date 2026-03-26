import cv2
import mediapipe as mp
import csv
import os
import glob

# ---------------------------
# 設定資料夾路徑
# ---------------------------
input_folder = "mediavideos"       # 放影片的資料夾
output_folder = "mediaoutputs"     # 輸出影片和 CSV
os.makedirs(output_folder, exist_ok=True)

# ---------------------------
# 初始化 MediaPipe
# ---------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 找到資料夾內所有影片 (avi / mp4)
video_files = glob.glob(os.path.join(input_folder, "*.avi")) + glob.glob(os.path.join(input_folder, "*.mp4"))

if not video_files:
    print("⚠️ 沒有找到任何影片，請確認資料夾路徑！")
    exit()

for input_file in video_files:
    print(f"▶️ 處理影片: {input_file}")

    # 產生對應的輸出檔名
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_folder, f"{base_name}_pose.avi")
    csv_file = os.path.join(output_folder, f"{base_name}_pose.csv")

    cap = cv2.VideoCapture(input_file)

    # 抓影片資訊
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps == 0:
        print("⚠️ 無法抓到影片 fps，設定為 30")
        fps = 30

    print(f"影片資訊: width={width}, height={height}, fps={fps}")

    # 建立影片輸出 (AVI + XVID 編碼)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    if not out.isOpened():
        print("⚠️ VideoWriter 建立失敗！")
        continue

    # ---------------------------
    # 建立 CSV 標頭
    # ---------------------------
    landmark_names = [f"{i}_{coord}" for i in range(33) for coord in ["x", "y", "z", "v"]]
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame"] + landmark_names)

    # ---------------------------
    # 偵測骨架並輸出影片 & CSV
    # ---------------------------
    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=1,
                      smooth_landmarks=True,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            row = [frame_idx]

            # 畫骨架 & 存 CSV
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                for lm in results.pose_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                # 沒偵測到骨架，用 None 補齊
                row.extend([None] * 33 * 4)

            # 寫入 CSV
            with open(csv_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            # 寫入影片
            out.write(frame)

            # 即時顯示
            #cv2.imshow("Pose Detection", frame)
            #if cv2.waitKey(1) & 0xFF == ord("q"):
            #    break

            frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"✅ 完成！影片輸出：{output_file}，CSV 輸出：{csv_file}")

print("🎉 全部影片處理完成！")

import sys
import vlc
import os
import subprocess
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QHBoxLayout,
    QFrame,
    QLabel,
    QInputDialog,
    QMessageBox,
    QCheckBox,
)
from PyQt5.QtCore import Qt, QTimer

# ==========================================
# RTSP URLs
# stream0 = main stream for recording
# stream1 = sub stream for preview
# ==========================================

camera1_view_url = 'rtsp://admin:Pass1234@192.168.0.200:554/stream1'
camera2_view_url = 'rtsp://admin:Pass1234@192.168.0.201:554/stream1'

camera1_record_url = 'rtsp://admin:Pass1234@192.168.0.200:554/stream0'
camera2_record_url = 'rtsp://admin:Pass1234@192.168.0.201:554/stream0'

# ==========================================
# ROI mask settings (visual only)
# ==========================================
CROP_LEFT_RATIO = 0.10
CROP_RIGHT_RATIO = 0.10
CROP_TOP_RATIO = 0.05
CROP_BOTTOM_RATIO = 0.05

FRAME_W = 640
FRAME_H = 360


class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Viewer")
        self.setGeometry(100, 100, FRAME_W * 2 + 80, FRAME_H + 220)

        layout = QVBoxLayout()
        video_layout = QHBoxLayout()

        self.instance = vlc.Instance(
            '--no-audio',
            '--network-caching=200',
            '--clock-jitter=0',
            '--clock-synchro=0',
            '--ffmpeg-hw'
        )

        self.player1 = self.instance.media_player_new()
        self.player2 = self.instance.media_player_new()

        self.frame1 = QFrame()
        self.frame2 = QFrame()
        self.frame1.setStyleSheet("background-color: black;")
        self.frame2.setStyleSheet("background-color: black;")
        self.frame1.setFixedSize(FRAME_W, FRAME_H)
        self.frame2.setFixedSize(FRAME_W, FRAME_H)

        # Dedicated video surfaces for VLC rendering
        self.video_surface1 = QFrame(self.frame1)
        self.video_surface2 = QFrame(self.frame2)
        self.video_surface1.setGeometry(0, 0, FRAME_W, FRAME_H)
        self.video_surface2.setGeometry(0, 0, FRAME_W, FRAME_H)
        self.video_surface1.setStyleSheet("background-color: black;")
        self.video_surface2.setStyleSheet("background-color: black;")

        # ROI masks (visual overlays only)
        self.mask1 = self._create_roi_mask_bars(self.frame1)
        self.mask2 = self._create_roi_mask_bars(self.frame2)
        self._set_roi_mask_visible(False)

        video_layout.addWidget(self.frame1)
        video_layout.addWidget(self.frame2)
        layout.addLayout(video_layout)

        self.time_label = QLabel(self)
        self.time_label.setStyleSheet(
            "color: white; font-size: 18px; "
            "background-color: rgba(0, 0, 0, 0.5); padding: 4px;"
        )
        self.time_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.time_label.setGeometry(10, 10, 260, 30)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)

        roi_layout = QHBoxLayout()
        self.crop_checkbox = QCheckBox("Enable ROI Mask")
        self.crop_checkbox.setChecked(False)
        self.crop_checkbox.stateChanged.connect(self.on_crop_toggle)
        self.crop_info = QLabel("Mask preset: L/R 10%, T/B 5%")
        roi_layout.addWidget(self.crop_checkbox)
        roi_layout.addWidget(self.crop_info)
        layout.addLayout(roi_layout)

        control_layout = QHBoxLayout()
        self.record_button = QPushButton("Start Recording")
        self.stop_button = QPushButton("Stop Recording")
        self.stop_button.setEnabled(False)

        self.record_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)

        self.record_button.setStyleSheet(
            "background-color: #4CAF50; color: white; "
            "font-size: 16px; padding: 10px; font-weight: bold;"
        )
        self.stop_button.setStyleSheet(
            "background-color: #f44336; color: white; "
            "font-size: 16px; padding: 10px; font-weight: bold;"
        )

        control_layout.addWidget(self.record_button)
        control_layout.addWidget(self.stop_button)
        layout.addLayout(control_layout)

        self.setLayout(layout)
        self.rec_proc = None

        self.start_streams()

    def _create_roi_mask_bars(self, parent):
        bars = {
            "left": QFrame(parent),
            "right": QFrame(parent),
            "top": QFrame(parent),
            "bottom": QFrame(parent),
        }
        for b in bars.values():
            b.setStyleSheet("background-color: black;")
            b.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            b.raise_()
        return bars

    def _update_roi_mask_geometry(self):
        def apply(frame, bars):
            w = frame.width()
            h = frame.height()
            left = int(round(w * CROP_LEFT_RATIO))
            right = int(round(w * CROP_RIGHT_RATIO))
            top = int(round(h * CROP_TOP_RATIO))
            bottom = int(round(h * CROP_BOTTOM_RATIO))

            left = max(0, min(left, w))
            right = max(0, min(right, w))
            top = max(0, min(top, h))
            bottom = max(0, min(bottom, h))

            bars["left"].setGeometry(0, 0, left, h)
            bars["right"].setGeometry(max(0, w - right), 0, right, h)
            bars["top"].setGeometry(0, 0, w, top)
            bars["bottom"].setGeometry(0, max(0, h - bottom), w, bottom)

        apply(self.frame1, self.mask1)
        apply(self.frame2, self.mask2)

    def _set_roi_mask_visible(self, visible):
        for b in self.mask1.values():
            b.setVisible(visible)
        for b in self.mask2.values():
            b.setVisible(visible)

    def on_crop_toggle(self, _state):
        if self.crop_checkbox.isChecked():
            self._update_roi_mask_geometry()
            self._set_roi_mask_visible(True)
        else:
            self._set_roi_mask_visible(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.video_surface1.setGeometry(0, 0, self.frame1.width(), self.frame1.height())
        self.video_surface2.setGeometry(0, 0, self.frame2.width(), self.frame2.height())
        if self.crop_checkbox.isChecked():
            self._update_roi_mask_geometry()

    def update_time(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.setText(f"Time: {now}")

    def start_streams(self):
        self.player1.set_hwnd(int(self.video_surface1.winId()))
        self.player2.set_hwnd(int(self.video_surface2.winId()))

        media1 = self.instance.media_new(camera1_view_url)
        media2 = self.instance.media_new(camera2_view_url)

        self.player1.set_media(media1)
        self.player2.set_media(media2)

        # Keep native preview behavior (no forced crop/scaling changes).
        self.player1.video_set_aspect_ratio(b"16:9")
        self.player2.video_set_aspect_ratio(b"16:9")
        self.player1.video_set_scale(0)
        self.player2.video_set_scale(0)

        self.player1.play()
        self.player2.play()

    def start_recording(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename1 = f"camera1_{timestamp}.mp4"
        self.filename2 = f"camera2_{timestamp}.mp4"

        cmd = [
            "ffmpeg", "-y",
            "-rtsp_transport", "tcp",
            "-i", camera1_record_url,
            "-rtsp_transport", "tcp",
            "-i", camera2_record_url,
            "-map", "0:v",
            "-c:v", "copy",
            "-an",
            self.filename1,
            "-map", "1:v",
            "-c:v", "copy",
            "-an",
            self.filename2
        ]

        self.rec_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        self.record_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.setWindowTitle("Recording...")

    def stop_recording(self):
        if self.rec_proc:
            try:
                self.rec_proc.stdin.write(b'q')
                self.rec_proc.stdin.flush()
                self.rec_proc.wait(timeout=5)
            except Exception:
                self.rec_proc.kill()

            self.rec_proc = None

            custom_name, ok = QInputDialog.getText(self, "Rename", "Recording name (optional):")
            if ok and custom_name.strip():
                custom_name = custom_name.strip()
                try:
                    if os.path.exists(self.filename1):
                        os.rename(self.filename1, f"{custom_name}_camera1.mp4")
                    if os.path.exists(self.filename2):
                        os.rename(self.filename2, f"{custom_name}_camera2.mp4")
                    QMessageBox.information(self, "Saved", "Files renamed.")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Rename failed: {e}")
            else:
                QMessageBox.information(self, "Saved", "Files saved with default names.")

        self.record_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.setWindowTitle("Camera Viewer")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())

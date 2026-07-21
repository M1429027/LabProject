import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config_camera_webcam_quad.yaml"


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    ffmpeg_cfg = cfg.get("ffmpeg", {})
    capture_cfg = cfg.get("capture", {})
    output_cfg = cfg.get("output", {})
    sync_cfg = cfg.get("sync_check", {})

    defaults = {
        "ffmpeg_bin": ffmpeg_cfg.get("ffmpeg_bin", "ffmpeg"),
        "ffprobe_bin": ffmpeg_cfg.get("ffprobe_bin", "ffprobe"),
        "rtbufsize": ffmpeg_cfg.get("rtbufsize", "512M"),
        "width": int(capture_cfg.get("width", 1280)),
        "height": int(capture_cfg.get("height", 720)),
        "fps": int(capture_cfg.get("fps", 30)),
        "codec": capture_cfg.get("codec", "libx264"),
        "preset": capture_cfg.get("preset", "veryfast"),
        "pix_fmt": capture_cfg.get("pix_fmt", "yuv420p"),
        "output_root": output_cfg.get("root", "outputs/calibration"),
        "session_tag": output_cfg.get("session_tag", "webcam_quad"),
        "frame_tol": int(sync_cfg.get("frame_diff_tolerance", 1)),
        "duration_tol": float(sync_cfg.get("duration_diff_tolerance_sec", 0.05)),
        "devices": cfg.get("devices", []),
    }
    return defaults


def list_dshow_video_devices(ffmpeg_bin: str):
    cmd = [ffmpeg_bin, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = (proc.stderr or "") + "\n" + (proc.stdout or "")

    devices = []
    in_video_section = False
    for raw in text.splitlines():
        line = raw.strip()
        if "DirectShow video devices" in line:
            in_video_section = True
            continue
        if "DirectShow audio devices" in line:
            in_video_section = False
        if in_video_section:
            m = re.search(r'"([^"]+)"', line)
            if m:
                name = m.group(1)
                if name not in devices:
                    devices.append(name)
    return devices, text


def ffprobe_video(ffprobe_bin: str, path: Path):
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,avg_frame_rate,width,height",
        "-show_entries",
        "format=duration,start_time",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffprobe failed")

    payload = json.loads(proc.stdout)
    stream = (payload.get("streams") or [{}])[0]
    fmt = payload.get("format", {})

    avg_rate = stream.get("avg_frame_rate", "0/1")
    fps = 0.0
    if "/" in avg_rate:
        num, den = avg_rate.split("/", 1)
        den_val = float(den) if float(den) != 0 else 1.0
        fps = float(num) / den_val

    duration = float(fmt.get("duration", 0.0))
    nb_frames_raw = stream.get("nb_frames", "0")
    if nb_frames_raw in ("N/A", None):
        nb_frames = int(round(duration * fps)) if fps > 0 else 0
    else:
        nb_frames = int(float(nb_frames_raw))

    return {
        "path": str(path),
        "width": int(stream.get("width", 0) or 0),
        "height": int(stream.get("height", 0) or 0),
        "fps": fps,
        "nb_frames": nb_frames,
        "duration": duration,
        "start_time": float(fmt.get("start_time", 0.0) or 0.0),
    }


def build_report(metrics_by_cam, frame_tol: int, duration_tol: float):
    frames = [m["nb_frames"] for m in metrics_by_cam.values()]
    durations = [m["duration"] for m in metrics_by_cam.values()]
    starts = [m["start_time"] for m in metrics_by_cam.values()]

    frame_diff = max(frames) - min(frames)
    duration_diff = max(durations) - min(durations)
    start_diff = max(starts) - min(starts)

    if frame_diff <= frame_tol and duration_diff <= duration_tol:
        status = "pass"
    else:
        status = "warn"

    return {
        "status": status,
        "thresholds": {
            "frame_diff_tolerance": frame_tol,
            "duration_diff_tolerance_sec": duration_tol,
        },
        "summary": {
            "frame_diff": frame_diff,
            "duration_diff_sec": duration_diff,
            "start_time_diff_sec": start_diff,
        },
        "cameras": metrics_by_cam,
    }


class WebcamQuadApp(QWidget):
    def __init__(self, config_path: Path):
        super().__init__()
        self.config_path = config_path
        self.cfg = load_config(config_path)
        self.ffmpeg_proc = None
        self.current_session_dir = None

        self.setWindowTitle("Webcam Quad Recorder (720p/30fps)")
        self.setGeometry(100, 80, 980, 620)

        layout = QVBoxLayout()

        top = QLabel(
            "4-camera synchronized recording via single ffmpeg process (DirectShow)."
        )
        top.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(top)

        self.env_label = QLabel()
        if os.name != "nt":
            self.env_label.setText(
                "Warning: DirectShow capture works on Windows runtime. Current OS is not Windows."
            )
            self.env_label.setStyleSheet("color: #b00020;")
        else:
            self.env_label.setText("Runtime: Windows DirectShow")
        layout.addWidget(self.env_label)

        form = QGridLayout()
        self.session_tag_input = QLineEdit(self.cfg["session_tag"])
        form.addWidget(QLabel("Session tag"), 0, 0)
        form.addWidget(self.session_tag_input, 0, 1)

        self.device_boxes = []
        for i in range(4):
            form.addWidget(QLabel(f"Camera {i + 1}"), i + 1, 0)
            box = QComboBox()
            box.setEditable(False)
            self.device_boxes.append(box)
            form.addWidget(box, i + 1, 1)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh devices")
        self.start_btn = QPushButton("Start recording")
        self.stop_btn = QPushButton("Stop recording")
        self.stop_btn.setEnabled(False)

        self.refresh_btn.clicked.connect(self.refresh_devices)
        self.start_btn.clicked.connect(self.start_recording)
        self.stop_btn.clicked.connect(self.stop_recording)

        btn_row.addWidget(self.refresh_btn)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        self.status_label = QLabel("Idle")
        layout.addWidget(self.status_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        self.setLayout(layout)
        self.refresh_devices()

    def log(self, message: str):
        self.log_text.append(message)

    def refresh_devices(self):
        for box in self.device_boxes:
            box.clear()
        self.status_label.setText("Scanning devices...")

        try:
            devices, raw = list_dshow_video_devices(self.cfg["ffmpeg_bin"])
        except FileNotFoundError:
            QMessageBox.critical(self, "ffmpeg missing", "ffmpeg not found in PATH.")
            self.status_label.setText("ffmpeg not found")
            return

        if not devices:
            self.log("No DirectShow video devices detected.")
            self.log(raw[-1200:])
            self.status_label.setText("No devices")
            return

        self.log(f"Detected {len(devices)} video device(s).")
        for box in self.device_boxes:
            box.addItems(devices)

        preferred = [d.get("name", "") for d in self.cfg["devices"]][:4]
        for idx, name in enumerate(preferred):
            if idx >= len(self.device_boxes):
                break
            if name in devices:
                self.device_boxes[idx].setCurrentText(name)

        self.status_label.setText(f"Ready ({len(devices)} devices found)")

    def _selected_devices(self):
        names = [box.currentText().strip() for box in self.device_boxes]
        if any(not n for n in names):
            raise ValueError("All 4 camera selections must be set.")
        if len(set(names)) != len(names):
            raise ValueError("Camera selections must be unique (no duplicates).")
        return names

    def _build_session_dir(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = self.session_tag_input.text().strip() or "webcam_quad"
        session_name = f"{ts}_{tag}"
        output_root = REPO_ROOT / self.cfg["output_root"]
        output_root.mkdir(parents=True, exist_ok=True)
        session_dir = output_root / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _build_ffmpeg_cmd(self, devices, session_dir: Path):
        size = f"{self.cfg['width']}x{self.cfg['height']}"
        fps = str(self.cfg["fps"])

        cmd = [self.cfg["ffmpeg_bin"], "-hide_banner", "-y"]
        for name in devices:
            cmd.extend(
                [
                    "-f",
                    "dshow",
                    "-rtbufsize",
                    self.cfg["rtbufsize"],
                    "-framerate",
                    fps,
                    "-video_size",
                    size,
                    "-i",
                    f"video={name}",
                ]
            )

        for idx in range(4):
            out = session_dir / f"cam{idx + 1}.mp4"
            cmd.extend(
                [
                    "-map",
                    f"{idx}:v:0",
                    "-c:v",
                    self.cfg["codec"],
                    "-preset",
                    self.cfg["preset"],
                    "-pix_fmt",
                    self.cfg["pix_fmt"],
                    "-r",
                    fps,
                    "-vsync",
                    "cfr",
                    "-an",
                    str(out),
                ]
            )
        return cmd

    def start_recording(self):
        if self.ffmpeg_proc is not None:
            return

        try:
            devices = self._selected_devices()
            session_dir = self._build_session_dir()
            cmd = self._build_ffmpeg_cmd(devices, session_dir)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid setup", str(exc))
            return

        self.log("Starting synchronized recording...")
        self.log("Command: " + " ".join(cmd))

        self.current_session_dir = session_dir
        self.ffmpeg_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText(f"Recording -> {session_dir}")

    def stop_recording(self):
        if self.ffmpeg_proc is None:
            return

        self.log("Stopping recording...")
        try:
            if self.ffmpeg_proc.stdin:
                self.ffmpeg_proc.stdin.write("q\n")
                self.ffmpeg_proc.stdin.flush()
            self.ffmpeg_proc.wait(timeout=10)
        except Exception:
            self.ffmpeg_proc.kill()

        stderr_log = ""
        if self.ffmpeg_proc.stderr:
            stderr_log = self.ffmpeg_proc.stderr.read() or ""
        self.ffmpeg_proc = None

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        if self.current_session_dir is None:
            self.status_label.setText("Stopped")
            return

        report_path = self.current_session_dir / "sync_report.json"
        metrics = {}
        try:
            for idx in range(4):
                video_path = self.current_session_dir / f"cam{idx + 1}.mp4"
                if not video_path.exists():
                    raise FileNotFoundError(f"Missing output: {video_path.name}")
                metrics[f"cam{idx + 1}"] = ffprobe_video(self.cfg["ffprobe_bin"], video_path)

            report = build_report(metrics, self.cfg["frame_tol"], self.cfg["duration_tol"])
            with report_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            self.log(f"sync_report.json saved: {report_path}")
            self.log(json.dumps(report["summary"], indent=2))
            self.status_label.setText(f"Stopped ({report['status'].upper()})")

            QMessageBox.information(
                self,
                "Recording complete",
                f"Session: {self.current_session_dir}\nStatus: {report['status'].upper()}\nReport: {report_path.name}",
            )
        except Exception as exc:
            self.status_label.setText("Stopped (FAIL)")
            self.log("Stop error: " + str(exc))
            if stderr_log:
                self.log("ffmpeg stderr (tail):\n" + "\n".join(stderr_log.splitlines()[-40:]))
            QMessageBox.critical(self, "Post-check failed", str(exc))


def main():
    config_path = DEFAULT_CONFIG
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1]).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    app = QApplication(sys.argv)
    window = WebcamQuadApp(config_path)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

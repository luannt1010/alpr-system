import os
import sys
import textwrap
import cv2
import numpy as np
from alpr.detector.helper_utils import crop_image
from alpr.pipeline import PIPELINE
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel,
                             QMessageBox, QPushButton, QSizePolicy, QTextEdit, QVBoxLayout, QWidget)

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

class ModelCache:
    _model = None
    @classmethod
    def get_model(cls):
        if cls._model is None:
            print("Loading model...")
            cls._model = PIPELINE(False)
        return cls._model


class FrameWorker(QThread):
    finished = pyqtSignal(str, object)
    error = pyqtSignal(str)

    def __init__(self, frame):
        super().__init__()
        self.frame = frame.copy()

    def run(self):
        try:
            result_text, crop_frame = self.run_inference(self.frame)
            self.finished.emit(result_text, crop_frame)
        except Exception as exc:
            self.error.emit(str(exc))

    def run_inference(self, frame):
        model = ModelCache.get_model()
        result = model.predict(frame)

        if result is None:
            return "No license plate detected in the current frame.", None

        text = str(result.get("Text") or "").strip()
        if text == "":
            text = "No OCR text"
        crop_frame = np.asarray(crop_image(frame, result["BBox"]))
        result_text = textwrap.dedent(
            f"""
            Text: {text}
            """
        ).strip()

        return result_text, crop_frame


class DropVideoBox(QFrame):
    video_selected = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.video_path = None
        self.preview_pixmap = None

        self.setAcceptDrops(True)
        self.setMinimumSize(520, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            """
            QFrame {
                background-color: white;
                border: 2px solid black;
            }
            """
        )

        self.label = QLabel("Drag & Drop Video Here\nor click Browse", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(
            """
            QLabel {
                border: none;
                font-size: 18px;
                color: #333333;
                background: transparent;
            }
            """
        )

    def resizeEvent(self, event):
        self.label.setGeometry(0, 0, self.width(), self.height())
        self.render_preview()
        super().resizeEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.preview_pixmap is not None:
            return

        painter = QPainter(self)
        painter.setPen(QPen(QColor(0, 0, 0), 1))

        margin = 25
        painter.drawLine(margin, margin, self.width() - margin, self.height() - margin)
        painter.drawLine(self.width() - margin, margin, margin, self.height() - margin)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            path = event.mimeData().urls()[0].toLocalFile()
            if self.is_video_file(path):
                event.acceptProposedAction()
            else:
                event.ignore()

    def dropEvent(self, event):
        path = event.mimeData().urls()[0].toLocalFile()
        self.set_video(path)

    def mousePressEvent(self, event):
        self.browse_video()

    def browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm)",
        )

        if file_path:
            self.set_video(file_path)

    def set_video(self, path):
        if not self.is_video_file(path):
            QMessageBox.warning(self, "Invalid File", "Please select a video file.")
            return

        self.video_path = path
        self.set_message(f"Selected Video:\n{os.path.basename(path)}")
        self.video_selected.emit(path)

    def is_video_file(self, path):
        return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS

    def set_message(self, message):
        self.preview_pixmap = None
        self.label.clear()
        self.label.setText(message)
        self.update()

    def set_preview_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width
        image = QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        ).copy()

        self.preview_pixmap = QPixmap.fromImage(image)
        self.render_preview()
        self.update()

    def render_preview(self):
        if self.preview_pixmap is None:
            return

        scaled_pixmap = self.preview_pixmap.scaled(
            self.label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.label.setText("")
        self.label.setPixmap(scaled_pixmap)


class VideoApp(QWidget):
    def __init__(self):
        super().__init__()

        self.video_path = None
        self.worker = None
        self.preview_capture = None
        self.preview_interval = 33
        self.current_frame = None
        self.crop_pixmap = None

        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self.play_preview_frame)

        self.setWindowTitle("Video Recognition App")
        self.resize(1000, 520)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        content_layout = QHBoxLayout()

        left_layout = QVBoxLayout()

        self.drop_box = DropVideoBox()
        self.drop_box.video_selected.connect(self.on_video_selected)

        self.browse_btn = QPushButton("Browse Video")
        self.browse_btn.setFixedHeight(40)
        self.browse_btn.clicked.connect(self.drop_box.browse_video)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setFixedHeight(40)
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.toggle_pause)

        video_controls = QHBoxLayout()
        video_controls.addWidget(self.browse_btn)
        video_controls.addWidget(self.pause_btn)

        left_layout.addWidget(self.drop_box)
        left_layout.addLayout(video_controls)

        right_layout = QVBoxLayout()

        result_title = QLabel("RESULT")
        result_title.setAlignment(Qt.AlignCenter)
        result_title.setStyleSheet(
            """
            QLabel {
                font-size: 18px;
                font-weight: bold;
            }
            """
        )

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setPlaceholderText("Pause video, then press GET to predict current frame...")
        self.result_box.setStyleSheet(
            """
            QTextEdit {
                background-color: white;
                border: 2px solid black;
                font-size: 15px;
                padding: 10px;
            }
            """
        )

        self.crop_label = QLabel("Cropped plate will appear here.")
        self.crop_label.setAlignment(Qt.AlignCenter)
        self.crop_label.setMinimumHeight(140)
        self.crop_label.setStyleSheet(
            """
            QLabel {
                background-color: white;
                border: 2px solid black;
                color: #333333;
                font-size: 14px;
            }
            """
        )

        self.get_btn = QPushButton("GET")
        self.get_btn.setFixedHeight(45)
        self.get_btn.setStyleSheet(
            """
            QPushButton {
                font-size: 16px;
                font-weight: bold;
            }
            """
        )
        self.get_btn.clicked.connect(self.process_current_frame)

        right_layout.addWidget(result_title)
        right_layout.addWidget(self.result_box)
        right_layout.addWidget(self.crop_label)
        right_layout.addWidget(self.get_btn)

        content_layout.addLayout(left_layout, stretch=3)
        content_layout.addLayout(right_layout, stretch=2)

        main_layout.addLayout(content_layout)
        self.setLayout(main_layout)

    def on_video_selected(self, path):
        self.video_path = path
        self.current_frame = None
        self.result_box.clear()
        self.clear_crop_preview()
        self.start_video_preview(path)

    def start_video_preview(self, path):
        self.stop_video_preview(reset_frame=False)

        self.preview_capture = cv2.VideoCapture(path)
        if not self.preview_capture.isOpened():
            self.preview_capture = None
            self.drop_box.set_message("Cannot preview selected video.")
            self.pause_btn.setEnabled(False)
            return

        fps = self.preview_capture.get(cv2.CAP_PROP_FPS)
        self.preview_interval = int(1000 / fps) if fps and fps > 0 else 33
        self.preview_interval = max(15, min(self.preview_interval, 100))

        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("Pause")
        self.play_preview_frame()
        self.preview_timer.start(self.preview_interval)

    def play_preview_frame(self):
        if self.preview_capture is None:
            return

        ret, frame = self.preview_capture.read()
        if not ret:
            self.preview_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.preview_capture.read()
            if not ret:
                self.stop_video_preview()
                self.drop_box.set_message("Cannot preview selected video.")
                return

        self.current_frame = frame.copy()
        self.drop_box.set_preview_frame(frame)

    def toggle_pause(self):
        if self.preview_capture is None:
            return

        if self.preview_timer.isActive():
            self.pause_preview()
        else:
            self.preview_timer.start(self.preview_interval)
            self.pause_btn.setText("Pause")

    def pause_preview(self):
        if self.preview_timer.isActive():
            self.preview_timer.stop()
        self.pause_btn.setText("Resume")

    def stop_video_preview(self, reset_frame=True):
        if self.preview_timer.isActive():
            self.preview_timer.stop()

        if self.preview_capture is not None:
            self.preview_capture.release()
            self.preview_capture = None

        if reset_frame:
            self.current_frame = None

        if hasattr(self, "pause_btn"):
            self.pause_btn.setText("Pause")
            self.pause_btn.setEnabled(False)

    def process_current_frame(self):
        if self.video_path is None:
            QMessageBox.warning(self, "No Video", "Please select a video first.")
            return

        if self.current_frame is None:
            QMessageBox.warning(self, "No Frame", "No current frame is available.")
            return

        self.pause_preview()
        frame = self.current_frame.copy()

        self.result_box.setText("Predicting current frame...")
        self.clear_crop_preview("Waiting for cropped plate...")
        self.get_btn.setEnabled(False)

        self.worker = FrameWorker(frame)
        self.worker.finished.connect(self.on_process_finished)
        self.worker.error.connect(self.on_process_error)
        self.worker.start()

    def on_process_finished(self, result, crop_frame):
        self.result_box.setText(result)
        self.set_crop_preview(crop_frame)
        self.get_btn.setEnabled(True)

    def on_process_error(self, error):
        self.result_box.setText("Error:\n" + error)
        self.clear_crop_preview("No cropped plate available.")
        self.get_btn.setEnabled(True)

    def clear_crop_preview(self, message="Cropped plate will appear here."):
        self.crop_pixmap = None
        self.crop_label.clear()
        self.crop_label.setText(message)

    def set_crop_preview(self, crop_frame):
        if crop_frame is None:
            self.clear_crop_preview("No cropped plate available.")
            return
        height, width, channels = crop_frame.shape
        bytes_per_line = channels * width
        image = QImage(
            crop_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        ).copy()

        self.crop_pixmap = QPixmap.fromImage(image)
        self.render_crop_preview()

    def render_crop_preview(self):
        if self.crop_pixmap is None or not hasattr(self, "crop_label"):
            return

        scaled_pixmap = self.crop_pixmap.scaled(
            self.crop_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.crop_label.setText("")
        self.crop_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        self.render_crop_preview()
        super().resizeEvent(event)

    def closeEvent(self, event):
        self.stop_video_preview()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoApp()
    window.show()
    sys.exit(app.exec_())

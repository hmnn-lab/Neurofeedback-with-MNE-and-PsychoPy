import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog,
    QMessageBox, QLineEdit, QFormLayout, QProgressDialog
)
from PyQt6.QtCore import pyqtSignal, QThread
from recordnsave_eeg import record_eeg_stream
from fixation_display import run_fixation_display
from final_preproc_window import PreprocessingGUI  # Adjust import as needed


class RecordingThread(QThread):
    recording_finished = pyqtSignal(str)
    recording_error = pyqtSignal(str)

    def __init__(self, stream_name, duration):
        super().__init__()
        self.stream_name = stream_name
        self.duration = duration

    def run(self):
        try:
            filepath = record_eeg_stream(self.stream_name, self.duration)
            self.recording_finished.emit(filepath)
        except Exception as e:
            self.recording_error.emit(str(e))


class BaselineWindow(QWidget):
    baseline_ready = pyqtSignal(str)  # Signal emitted with baseline file path when ready

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Baseline Recording")
        self.setGeometry(300, 300, 500, 300)

        self.baseline_file = None
        self.progress = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        title = QLabel("Choose a method to provide baseline EEG:")
        layout.addWidget(title)

        # File selection
        self.browse_button = QPushButton("Browse Existing .fif File")
        self.browse_button.clicked.connect(self.browse_file)
        layout.addWidget(self.browse_button)

        # Recording input
        self.stream_name_input = QLineEdit()
        self.stream_name_input.setPlaceholderText("e.g., EEG_Stream")
        self.duration_input = QLineEdit()
        self.duration_input.setPlaceholderText("e.g., 10 (seconds)")

        form = QFormLayout()
        form.addRow("LSL Stream Name:", self.stream_name_input)
        form.addRow("Recording Duration (sec):", self.duration_input)
        layout.addLayout(form)

        self.record_button = QPushButton("Record Baseline EEG")
        self.record_button.clicked.connect(self.record_baseline)
        layout.addWidget(self.record_button)

        # Next button
        self.next_button = QPushButton("Next")
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self.proceed_to_preprocessing)
        layout.addWidget(self.next_button)

        # Back button

        # self.back_button = QPushButton("Back")
        # self.back_button.clicked.connect(self.back_requested.emit)
        # layout.addWidget(self.back_button)

        self.setLayout(layout)

    def browse_file(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("FIF files (*.fif)")
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                print(f"[DEBUG] Baseline browse selected: {file_path}")

                self.baseline_file = file_path
                self.next_button.setEnabled(True)
                QMessageBox.information(self, "File Selected", f"Baseline file selected:\n{file_path}")

                self.baseline_ready.emit(file_path)  # <-- Emit signal here

    def record_baseline(self):
        stream_name = self.stream_name_input.text().strip()
        duration = self.duration_input.text().strip()

        if not stream_name or not duration:
            QMessageBox.warning(self, "Input Error", "Please provide both stream name and duration.")
            return

        try:
            duration = float(duration)
            if duration <= 0:
                raise ValueError

            self.record_button.setEnabled(False)
            self.progress = QProgressDialog("Recording EEG...", None, 0, 0, self)
            self.progress.setWindowTitle("Recording")
            self.progress.setMinimumDuration(0)
            self.progress.setCancelButton(None)
            self.progress.show()

            self.thread = RecordingThread(stream_name, duration)
            self.thread.recording_finished.connect(self.on_recording_finished)
            self.thread.recording_error.connect(self.on_recording_error)
            self.thread.start()

            run_fixation_display(duration)

        except ValueError:
            QMessageBox.warning(self, "Input Error", "Duration must be a positive number.")

    def on_recording_finished(self, filepath):
        print(f"[DEBUG] Recording finished: {filepath}")
        self.baseline_file = filepath
        self.next_button.setEnabled(True)
        self.record_button.setEnabled(True)
        if self.progress:
            self.progress.close()
        QMessageBox.information(self, "Recording Complete", f"Saved baseline EEG to:\n{filepath}")

        self.baseline_ready.emit(filepath)  # <-- Emit signal here

    def on_recording_error(self, error_msg):
        if self.progress:
            self.progress.close()
        self.record_button.setEnabled(True)
        self.next_button.setEnabled(False)
        QMessageBox.critical(self, "Recording Error", error_msg)

    def proceed_to_preprocessing(self):
        if not self.baseline_file:
            QMessageBox.warning(self, "File Error", "No baseline file provided.")
            return
        if not os.path.exists(self.baseline_file):
            QMessageBox.warning(self, "File Error", f"Baseline file not found:\n{self.baseline_file}")
            return

        print(f"[DEBUG] Proceeding with baseline file: {self.baseline_file}")

        self.preproc_window = PreprocessingGUI(baseline_file_path=self.baseline_file)
        self.preproc_window.show()
        self.close()


                                                          
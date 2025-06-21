import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QDialog, QMainWindow, QVBoxLayout, QHBoxLayout, QLineEdit, 
    QPushButton, QLabel, QMessageBox, QWidget, QFileDialog, QFormLayout, QProgressDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from recordnsave_eeg import record_eeg_stream
from fixation_display import run_fixation_display

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

class ChannelDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Channel Input Dialog")
        self.setFixedSize(300, 150)
        
        self.channel_data = {"num_channels": 0, "channel_names": []}
        
        layout = QVBoxLayout()
        
        self.num_channels_label = QLabel("Number of channels:")
        self.num_channels_input = QLineEdit()
        self.num_channels_input.setPlaceholderText("Enter a number")
        
        self.channel_names_label = QLabel("Channel names (comma-separated):")
        self.channel_names_input = QLineEdit()
        self.channel_names_input.setPlaceholderText("e.g., Channel1,Channel2")
        
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addWidget(self.num_channels_label)
        layout.addWidget(self.num_channels_input)
        layout.addWidget(self.channel_names_label)
        layout.addWidget(self.channel_names_input)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        self.ok_button.clicked.connect(self.validate_and_accept)
        self.cancel_button.clicked.connect(self.reject)
    
    def validate_and_accept(self):
        try:
            num_channels = int(self.num_channels_input.text().strip())
            if num_channels <= 0:
                raise ValueError("Number of channels must be positive")
                
            channel_names = [name.strip() for name in self.channel_names_input.text().split(",")]
            channel_names = [name for name in channel_names if name]
            
            if len(channel_names) != num_channels:
                raise ValueError(f"Expected {num_channels} channel names, got {len(channel_names)}")
                
            self.channel_data["num_channels"] = num_channels
            self.channel_data["channel_names"] = channel_names
            self.accept()
            
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", str(e))
    
    def get_data(self):
        return self.channel_data

class BaselineWindow(QWidget):
    # Updated signal to emit baseline_file, channel_names, and num_channels
    proceed_to_paradigm = pyqtSignal(str, list, int)

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

        file_layout = QHBoxLayout()
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Enter or select .fif file path")
        self.file_input.textChanged.connect(self.validate_file_path)
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_input)
        file_layout.addWidget(self.browse_button)
        layout.addLayout(file_layout)

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

        self.next_button = QPushButton("Next")
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self.open_channel_dialog)
        layout.addWidget(self.next_button)

    def browse_file(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("FIF files (*.fif)")
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                print(f"[DEBUG] Baseline browse selected: {file_path}")
                self.file_input.setText(file_path)
                self.validate_file_path(file_path)

    def validate_file_path(self, file_path=None):
        file_path = file_path or self.file_input.text().strip()
        if file_path and os.path.exists(file_path) and file_path.lower().endswith('.fif'):
            self.baseline_file = file_path
            self.next_button.setEnabled(True)
            print(f"[DEBUG] Valid baseline file path: {file_path}")
        else:
            self.baseline_file = None
            self.next_button.setEnabled(False)
            print(f"[DEBUG] Invalid baseline file path: {file_path}")

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
        if self.progress:
            self.progress.close()
        self.record_button.setEnabled(True)
        self.file_input.setText(filepath)
        self.validate_file_path(filepath)

    def on_recording_error(self, error_msg):
        if self.progress:
            self.progress.close()
        self.record_button.setEnabled(True)
        self.next_button.setEnabled(False)
        QMessageBox.critical(self, "Recording Error", error_msg)

    def open_channel_dialog(self):
        if not self.baseline_file:
            QMessageBox.warning(self, "File Error", "No baseline file provided.")
            return
        if not os.path.exists(self.baseline_file):
            QMessageBox.warning(self, "File Error", f"Baseline file not found:\n{self.baseline_file}")
            return

        print(f"[DEBUG] Opening channel dialog with baseline file: {self.baseline_file}")

        self.channel_dialog = ChannelDialog()
        if self.channel_dialog.exec():
            channel_data = self.channel_dialog.get_data()
            print(f"[DEBUG] Channel data collected: {channel_data}")
            # Emit signal with baseline_file, channel_names, and num_channels
            self.proceed_to_paradigm.emit(
                self.baseline_file,
                channel_data["channel_names"],
                channel_data["num_channels"]
            )

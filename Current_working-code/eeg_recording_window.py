import sys
import os
import re
import logging
import multiprocessing
from multiprocessing import Manager
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel,
                             QLineEdit, QPushButton, QMessageBox, QRadioButton, QButtonGroup, QFileDialog)
from PyQt6.QtCore import Qt
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from recordnsave_eeg import record_eeg_stream
from fixation_display import run_fixation_display
from eeg_processing_window import PreprocessingGUI
from cal_baseline_psd import compute_psd_auc

class EEGRecordingWindow(QMainWindow):
    def __init__(self, input_stream_name="Signal_generator"):
        super().__init__()
        self.setWindowTitle("EEG Baseline Recording")
        self.setGeometry(100, 100, 500, 500)

        self.input_stream_name = input_stream_name
        self.initialize_state()
        self.setup_ui()

    def initialize_state(self):
        self.recording_process = None
        self.manager = None
        self.result_dict = None
        self.start_event = None
        self.baseline_file = None

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        layout.addWidget(QLabel("Do you need to record a new baseline?"))
        self.radio_group = QButtonGroup()
        self.radio_yes = QRadioButton("Yes, record new baseline")
        self.radio_no = QRadioButton("No, use existing baseline")
        self.radio_group.addButton(self.radio_yes)
        self.radio_group.addButton(self.radio_no)
        self.radio_yes.setChecked(True)
        layout.addWidget(self.radio_yes)
        layout.addWidget(self.radio_no)

        self.stream_name_input = QLineEdit()
        self.stream_name_input.setPlaceholderText("e.g., Signal_generator")
        self.stream_name_input.setText(self.input_stream_name)
        layout.addWidget(QLabel("Input stream name:"))
        layout.addWidget(self.stream_name_input)

        self.duration_input = QLineEdit()
        self.duration_input.setPlaceholderText("e.g., 60")
        layout.addWidget(QLabel("Recording duration (seconds):"))
        layout.addWidget(self.duration_input)

        self.start_button = QPushButton("Start Recording")
        self.start_button.clicked.connect(self.start_recording)
        layout.addWidget(self.start_button)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.open_preprocessing_window)
        self.next_button.setEnabled(False)
        layout.addWidget(self.next_button)

        self.status_label = QLabel("Select whether to record a new baseline.")
        layout.addWidget(self.status_label)
        layout.addStretch()

        self.radio_yes.toggled.connect(self.toggle_inputs)
        self.radio_no.toggled.connect(self.select_existing_baseline)
        self.toggle_inputs()

    def toggle_inputs(self):
        recording_needed = self.radio_yes.isChecked()
        self.stream_name_input.setEnabled(recording_needed)
        self.duration_input.setEnabled(recording_needed)
        self.start_button.setEnabled(recording_needed)
        self.next_button.setEnabled(self.baseline_file is not None or not recording_needed)

    def select_existing_baseline(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Baseline File", "", "FIF Files (*.fif)")
        if file_path:
            self.baseline_file = file_path
            self.status_label.setText(f"Selected baseline: {os.path.basename(file_path)}")
            try:
                df, excel_path, plot_path = compute_psd_auc(file_path)
                self.status_label.setText("PSD processing complete.")
                QMessageBox.information(
                    self,
                    "Success",
                    f"Processed existing baseline: {file_path}\n"
                    f"PSD results saved at: {excel_path}\n"
                    f"PSD plot saved at: {plot_path}"
                )
                self.next_button.setEnabled(True)
            except Exception as e:
                self.status_label.setText(f"PSD processing failed: {e}")
                QMessageBox.critical(self, "Error", f"Failed to process {file_path}: {e}")
                self.next_button.setEnabled(False)
        else:
            self.baseline_file = None
            self.status_label.setText("No baseline file selected.")
            self.next_button.setEnabled(False)

    def generate_filename(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        baseline_folder = os.path.join(script_dir, "baseline_recordings")
        os.makedirs(baseline_folder, exist_ok=True)
        existing_files = os.listdir(baseline_folder)
        pattern = r'baseline_(\d{3})_eeg\.fif'
        max_num = 0
        for fname in existing_files:
            match = re.match(pattern, fname)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)
        file_num = max_num + 1
        filename = os.path.join(baseline_folder, f"baseline_{file_num:03d}_eeg.fif")
        return filename

    def validate_inputs(self, stream_name, duration):
        try:
            if not stream_name.strip():
                raise ValueError("Stream name cannot be empty.")
            duration = float(duration)
            if duration <= 0:
                raise ValueError("Duration must be positive.")
            return stream_name, duration
        except ValueError as e:
            return str(e)

    def fixation_task(self, duration, start_event):
        logging.debug("Starting fixation task")
        try:
            run_fixation_display(duration, start_event)
            logging.debug("Fixation task completed")
        except Exception as e:
            logging.error(f"Fixation task failed: {e}")
            raise

    def start_recording(self):
        if not self.radio_yes.isChecked():
            self.status_label.setText("No recording needed. Select a baseline file.")
            self.next_button.setEnabled(self.baseline_file is not None)
            return

        stream_name = self.stream_name_input.text()
        duration = self.duration_input.text()

        result = self.validate_inputs(stream_name, duration)
        if isinstance(result, str):
            QMessageBox.critical(self, "Input Error", f"Invalid input: {result}")
            return

        stream_name, duration = result
        filename = self.generate_filename()

        self.set_inputs_enabled(False)
        self.status_label.setText(f"Recording to {filename}...")
        QApplication.processEvents()

        try:
            self.manager = Manager()
            self.result_dict = self.manager.dict()
            self.start_event = multiprocessing.Event()

            self.recording_process = multiprocessing.Process(
                target=self.wrapper_recording_task,
                args=(stream_name, duration, filename, self.result_dict, self.start_event)
            )

            logging.debug("Starting recording process")
            self.recording_process.start()

            self.fixation_task(duration, self.start_event)
            QApplication.processEvents()

            self.recording_process.join()
            QApplication.processEvents()

            if self.recording_process.exitcode != 0:
                raise RuntimeError(f"Recording process failed with exit code: {self.recording_process.exitcode}")

            fif_path = self.result_dict.get("fif_path")
            if fif_path and os.path.exists(fif_path):
                self.baseline_file = fif_path
                self.status_label.setText("Recording complete. Processing PSD...")
                QApplication.processEvents()

                try:
                    df, excel_path, plot_path = compute_psd_auc(fif_path)
                    self.status_label.setText("PSD processing complete.")
                    QMessageBox.information(
                        self,
                        "Success",
                        f"Baseline recorded at: {fif_path}\n"
                        f"PSD results saved at: {excel_path}\n"
                        f"PSD plot saved at: {plot_path}\n"
                        "Proceed to preprocessing."
                    )
                    self.next_button.setEnabled(True)
                except Exception as e:
                    self.status_label.setText(f"PSD processing failed: {e}")
                    QMessageBox.critical(self, "Processing Error", f"PSD processing failed: {e}")
                    logging.error(f"PSD processing error: {e}")
                    self.next_button.setEnabled(False)
            else:
                raise FileNotFoundError(f"EEG file not found at {fif_path}.")

        except Exception as e:
            self.status_label.setText(f"Recording failed: {e}")
            QMessageBox.critical(self, "Recording Error", f"Recording failed: {e}")
            logging.error(f"Recording process error: {e}")
            self.next_button.setEnabled(False)

        finally:
            if self.recording_process and self.recording_process.is_alive():
                self.recording_process.terminate()
                self.recording_process.join()
            self.initialize_state()
            self.set_inputs_enabled(True)
            QApplication.processEvents()

    def set_inputs_enabled(self, enabled: bool):
        self.radio_yes.setEnabled(enabled)
        self.radio_no.setEnabled(enabled)
        self.stream_name_input.setEnabled(enabled and self.radio_yes.isChecked())
        self.duration_input.setEnabled(enabled and self.radio_yes.isChecked())
        self.start_button.setEnabled(enabled and self.radio_yes.isChecked())
        self.next_button.setEnabled(enabled and (self.baseline_file is not None or not self.radio_yes.isChecked()))

    @staticmethod
    def wrapper_recording_task(stream_name, duration, filename, result_dict, start_event):
        try:
            start_event.set()
            filepath = record_eeg_stream(stream_name, float(duration), filename)
            result_dict["fif_path"] = filepath
        except Exception as e:
            logging.error(f"Error in recording task: {e}")
            result_dict["fif_path"] = ""

    def open_preprocessing_window(self):
        if self.baseline_file is None:
            QMessageBox.critical(self, "Error", "No baseline file selected or recorded. Please record or select a baseline file.")
            return
        try:
            self.preprocessing_window = PreprocessingGUI(
                baseline_file=self.baseline_file,
                input_stream_name=self.stream_name_input.text().strip() or self.input_stream_name
            )
            self.preprocessing_window.show()
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open preprocessing window: {str(e)}")
            logging.error(f"Error opening PreprocessingGUI: {e}")

    def closeEvent(self, event):
        if self.recording_process and self.recording_process.is_alive():
            self.recording_process.terminate()
            self.recording_process.join()
        if event is not None:
            event.accept()
import sys
import os
import mne
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QFileDialog, QLineEdit,
    QVBoxLayout, QHBoxLayout, QProgressBar, QMessageBox
)
from preproc_initialize import preproc_flow
from channel_utils import rename_eeg_channels
from power_stream_window import RealTimePsdGui

class PreprocessingGUI(QWidget):
    def __init__(self, baseline_file=None, input_stream_name="Signal_generator"):
        super().__init__()
        self.setWindowTitle("EEG Preprocessing")
        self.setGeometry(200, 200, 500, 400)
        self.preprocessed_params = None
        self.input_stream_name = input_stream_name
        self.init_ui()

        if baseline_file:
            self.file_input.setText(baseline_file)
        self.lsl_stream_input.setText(self.input_stream_name)

    def init_ui(self):
        layout = QVBoxLayout()

        # Baseline file selection
        file_layout = QHBoxLayout()
        self.file_input = QLineEdit(self)
        self.file_input.setPlaceholderText("Select a .fif file")
        self.file_button = QPushButton("Browse Baseline File", self)
        self.file_button.clicked.connect(self.browse_file)
        file_layout.addWidget(QLabel("Baseline File:"))
        file_layout.addWidget(self.file_input)
        file_layout.addWidget(self.file_button)

        # Number of channels input
        ch_num_layout = QHBoxLayout()
        self.ch_num_label = QLabel("Number of Channels:")
        self.ch_num_input = QLineEdit(self)
        self.ch_num_input.setPlaceholderText("e.g., 16")
        ch_num_layout.addWidget(self.ch_num_label)
        ch_num_layout.addWidget(self.ch_num_input)

        # Channel names input
        ch_names_layout = QHBoxLayout()
        self.ch_names_label = QLabel("Channel Names (e.g., Pz,Fz,Cz):")
        self.ch_names_input = QLineEdit(self)
        self.ch_names_input.setPlaceholderText("Enter comma-separated channel names")
        ch_names_layout.addWidget(self.ch_names_label)
        ch_names_layout.addWidget(self.ch_names_input)

        # LSL stream name input
        lsl_stream_layout = QHBoxLayout()
        self.lsl_stream_label = QLabel("LSL Stream Name:")
        self.lsl_stream_input = QLineEdit(self)
        self.lsl_stream_input.setPlaceholderText("Enter LSL stream name")
        lsl_stream_layout.addWidget(self.lsl_stream_label)
        lsl_stream_layout.addWidget(self.lsl_stream_input)

        # Preprocess button
        self.preprocess_button = QPushButton("Start Preprocessing", self)
        self.preprocess_button.clicked.connect(self.run_preprocessing)

        # Next button
        self.next_button = QPushButton("Next", self)
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self.open_power_stream_window)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)

        layout.addLayout(file_layout)
        layout.addLayout(ch_num_layout)
        layout.addLayout(ch_names_layout)
        layout.addLayout(lsl_stream_layout)
        layout.addWidget(self.preprocess_button)
        layout.addWidget(self.next_button)
        layout.addWidget(self.progress_bar)
        layout.addStretch()

        self.setLayout(layout)

    def browse_file(self):
        file_dialog = QFileDialog()
        filepath, _ = file_dialog.getOpenFileName(self, "Select Baseline File", "", "FIF files (*.fif)")
        if filepath:
            self.file_input.setText(filepath)

    def parse_channel_input(self, ch_input, raw):
        if not ch_input.strip():
            return None

        channels = [ch.strip() for ch in ch_input.split(',')]
        montage = mne.channels.make_standard_montage('standard_1020')
        valid_montage_channels = set(montage.ch_names)

        for ch in channels:
            if ch not in valid_montage_channels:
                raise ValueError(f"Channel {ch} not in standard_1020 montage.")

        if len(channels) != len(raw.ch_names):
            raise ValueError(f"Number of channel names ({len(channels)}) does not match number of channels in raw data ({len(raw.ch_names)}).")

        rename_dict = {raw_ch: new_ch for raw_ch, new_ch in zip(raw.ch_names, channels)}
        return rename_dict

    def run_preprocessing(self):
        baseline_path = self.file_input.text()
        ch_input = self.ch_names_input.text()
        n_channels_input = self.ch_num_input.text()
        lsl_stream_name = self.lsl_stream_input.text()

        if not os.path.exists(baseline_path):
            QMessageBox.warning(self, "File Error", "Baseline file not found.")
            return

        if not lsl_stream_name.strip():
            QMessageBox.warning(self, "Input Error", "Please enter a valid LSL stream name.")
            return

        try:
            n_channels = int(n_channels_input)
            if n_channels <= 0:
                raise ValueError("Number of channels must be positive.")
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter a valid number of channels.")
            return

        self.progress_bar.setValue(10)
        self.repaint()

        try:
            raw = mne.io.read_raw_fif(baseline_path, preload=True)

            if n_channels != len(raw.ch_names):
                raise ValueError(f"Entered number of channels ({n_channels}) does not match raw data ({len(raw.ch_names)}).")

            rename_dict = self.parse_channel_input(ch_input, raw)
            raw, _ = rename_eeg_channels(raw, rename_dict)

            self.progress_bar.setValue(20)
            self.repaint()

            raw.set_eeg_reference('average')
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage)
            tmp_path = os.path.join(os.path.dirname(baseline_path), "_tmp_referenced.fif")
            raw.save(tmp_path, overwrite=True)

            self.progress_bar.setValue(40)
            self.repaint()

            raw_cleaned, bads, asr, ica, artifact_comps = preproc_flow(tmp_path, n_channels)

            self.progress_bar.setValue(80)
            self.repaint()

            # Store parameters locally
            self.preprocessed_params = {
                'bad_channels': bads,
                'asr': asr,
                'ica': ica,
                'artifact_components': artifact_comps,
                'info': raw_cleaned.info,
                'rename_dict': rename_dict,
                'lsl_stream_name': lsl_stream_name
            }

            self.progress_bar.setValue(100)
            QMessageBox.information(self, "Success", "Preprocessing complete. Proceed to power stream.")
            self.next_button.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Processing Error", f"Error: {str(e)}")
            self.progress_bar.setValue(0)
            self.next_button.setEnabled(False)
            self.preprocessed_params = None

    def open_power_stream_window(self):
        if not self.preprocessed_params:
            QMessageBox.critical(self, "Error", "No preprocessing parameters available. Please preprocess data first.")
            return

        try:
            self.power_stream_window = RealTimePsdGui(
                asr=self.preprocessed_params['asr'],
                ica=self.preprocessed_params['ica'],
                artifact_components=self.preprocessed_params['artifact_components'],
                rename_dict=self.preprocessed_params['rename_dict'],
                bad_channels=self.preprocessed_params['bad_channels'],
                info=self.preprocessed_params['info'],
                lsl_stream_name=self.preprocessed_params['lsl_stream_name']
            )
            self.power_stream_window.show()
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open power stream window: {str(e)}")
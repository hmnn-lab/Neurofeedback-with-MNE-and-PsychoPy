import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog, QCheckBox
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
from mne_realtime import LSLClient
from asrpy import ASR
from pyprep.find_noisy_channels import NoisyChannels
import subprocess

class EEGConfigGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout()
        
        self.sfreq_label = QLabel("Sampling Frequency:")
        self.sfreq_input = QLineEdit()
        layout.addWidget(self.sfreq_label)
        layout.addWidget(self.sfreq_input)
        
        self.n_channels_label = QLabel("Number of Channels:")
        self.n_channels_input = QLineEdit()
        layout.addWidget(self.n_channels_label)
        layout.addWidget(self.n_channels_input)
        
        self.host_label = QLabel("LSL Host Name:")
        self.host_input = QLineEdit()
        layout.addWidget(self.host_label)
        layout.addWidget(self.host_input)
        
        self.ch_names_label = QLabel("Channel Names (comma-separated):")
        self.ch_names_input = QTextEdit()
        layout.addWidget(self.ch_names_label)
        layout.addWidget(self.ch_names_input)
        
        self.baseline_label = QLabel("Baseline File Path:")
        self.baseline_input = QLineEdit()
        self.browse_button = QPushButton("Browse")
        self.record_button = QPushButton("Record New Baseline")
        self.task_baseline_checkbox = QCheckBox("Use recorded baseline as task baseline")
        
        self.browse_button.clicked.connect(self.browse_baseline_file)
        self.record_button.clicked.connect(self.record_baseline)
        
        layout.addWidget(self.baseline_label)
        layout.addWidget(self.baseline_input)
        layout.addWidget(self.browse_button)
        layout.addWidget(self.record_button)
        layout.addWidget(self.task_baseline_checkbox)
        
        self.start_button = QPushButton("Start EEG Processing")
        self.start_button.clicked.connect(self.start_processing)
        layout.addWidget(self.start_button)
        
        self.setLayout(layout)
        self.setWindowTitle("EEG Processing Configuration")
    
    def browse_baseline_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Baseline File", "", "FIF Files (*.fif)")
        if file_path:
            self.baseline_input.setText(file_path)
    
    def record_baseline(self):
        subprocess.run(["python", r"C:\Users\varsh\NFB_Spyder\Neurofeedback-with-MNE-and-PsychoPy\Final codes\record_baseline.py"])  # Replace with actual baseline recording script name
        
    def start_processing(self):
        sfreq = int(self.sfreq_input.text())
        n_channels = int(self.n_channels_input.text())
        host = self.host_input.text()
        ch_names = [ch.strip() for ch in self.ch_names_input.toPlainText().split(',')]
        baseline_file = self.baseline_input.text()
        
        self.close()
        run_eeg_processing(sfreq, n_channels, host, ch_names, baseline_file)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = EEGConfigGUI()
    gui.show()
    sys.exit(app.exec())

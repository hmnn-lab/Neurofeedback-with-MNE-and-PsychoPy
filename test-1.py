import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
from mne_realtime import LSLClient
from asrpy import ASR
from pyprep.find_noisy_channels import NoisyChannels

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
        self.browse_button.clicked.connect(self.browse_baseline_file)
        layout.addWidget(self.baseline_label)
        layout.addWidget(self.baseline_input)
        layout.addWidget(self.browse_button)
        
        self.start_button = QPushButton("Start EEG Processing")
        self.start_button.clicked.connect(self.start_processing)
        layout.addWidget(self.start_button)
        
        self.setLayout(layout)
        self.setWindowTitle("EEG Processing Configuration")
    
    def browse_baseline_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Baseline File", "", "FIF Files (*.fif)")
        if file_path:
            self.baseline_input.setText(file_path)
    
    def start_processing(self):
        sfreq = int(self.sfreq_input.text())
        n_channels = int(self.n_channels_input.text())
        host = self.host_input.text()
        ch_names = [ch.strip() for ch in self.ch_names_input.toPlainText().split(',')]
        baseline_file = self.baseline_input.text()
        
        self.close()
        run_eeg_processing(sfreq, n_channels, host, ch_names, baseline_file)

    def run_eeg_processing(sfreq, n_channels, host, ch_names, baseline_file):
        raw = mne.io.read_raw_fif(baseline_file, preload=True)
        raw.notch_filter(50, picks='eeg').filter(l_freq=0.1, h_freq=40)
        raw.set_eeg_reference('average')
    
        print("Available channel names in raw:", raw.info['ch_names'])
        print("User provided channel names:", ch_names)
    
        available_channels = set(raw.info['ch_names'])
        valid_channels = [ch for ch in ch_names if ch in available_channels]
    
        if len(valid_channels) < len(ch_names):
            missing_channels = set(ch_names) - set(valid_channels)
            print(f"Warning: The following user-input channels were not found in raw data: {missing_channels}")
    
        rename_dict = {ch: ch.capitalize() for ch in valid_channels if ch.upper() != ch.capitalize()}
        if rename_dict:
            raw.rename_channels(rename_dict)
    
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
    
        nd = NoisyChannels(raw, random_state=1337)
        nd.find_bad_by_ransac(channel_wise=True, max_chunk_size=1)
        bad_channels = nd.bad_by_ransac
        raw.info['bads'].extend(bad_channels)
    
        asr = ASR(sfreq=raw.info['sfreq'])
        asr.fit(raw)
        raw = asr.transform(raw)
    
        ica = ICA(n_components=n_channels-len(bad_channels), method='infomax', max_iter=500, random_state=42)
        ica.fit(raw)

    labels = label_components(raw, ica, 'iclabel')
    component_labels = labels['labels']
    component_probs = labels['y_pred_proba']
    artifact_components = []
    for i, prob in enumerate(component_probs):
        if prob >= 0.9 and prob <= 1and component_labels[i] in ['muscle', 'eye']:
            print(f"Component (i) is likely an artifact")
            artifact_components.append(i)
    
    ica_weights = ica.unmixing_matrix_
    ica_inverse_weights = ica.mixing_matrix_
    print("flagged artifact components: ", artifact_components)

    raw_cleaned = ica.apply(raw.copy(), exclude=artifact_components)
    
    fig, axs = plt.subplots(len(valid_channels), 1, figsize=(10, 12), sharex=True)
    plt.subplots_adjust(hspace=0.5)
    
    with LSLClient(info=None, host=host, wait_max=5) as client:
        client_info = client.get_measurement_info()
        sfreq = int(client_info['sfreq'])
        
    while True:
        epoch = client.get_data_as_epoch(n_samples=sfreq)
        epoch.apply_baseline(baseline=(0, None))
        raw_realtime = mne.io.RawArray(epoch.get_data().squeeze(), client_info)
        raw_realtime.info['bads'].extend(bad_channels)
        raw_realtime_asr = asr.transform(raw_realtime)
        raw_realtime_asr_ica = ica.apply(raw_realtime_asr, exclude=artifact_components)
            
        for i, ax in enumerate(axs):
            ax.clear()
            ax.plot(raw_realtime_asr_ica.times, raw_realtime_asr_ica.get_data(picks=[i])[0].T)
            ax.set_title(f'Channel: {valid_channels[i]}')
            ax.set_ylabel('Amplitude (μV)')
        axs[-1].set_xlabel('Time (s)')
        plt.pause(0.1)
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = EEGConfigGUI()
    gui.show()
    sys.exit(app.exec())

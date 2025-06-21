import sys
import os
import numpy as np
import mne
import logging
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QComboBox, QDoubleSpinBox, QGroupBox, QMessageBox
)
from PyQt6.QtCore import QTimer, pyqtSignal
from mne_lsl.lsl import StreamInlet, resolve_streams, StreamOutlet
import pyqtgraph as pg
from preproc_apply import preprocess_realtime_stream
from power_auc_cal import compute_band_auc
from single_freq_vis_utils import run_simple_visualization, launch_game_mode

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class VisualGui(QMainWindow):
    def __init__(self, asr, ica, artifact_components, bad_channels, info, lsl_stream_name, plotter=None):
        super().__init__()

        self.setWindowTitle("Visual Choice & PSD Monitor")
        self.setGeometry(100, 100, 800, 700)

        self.asr = asr
        self.ica = ica
        self.artifact_components = artifact_components
        self.bad_channels = bad_channels
        self.info = info
        self.lsl_stream_name = lsl_stream_name
        self.plotter = plotter

        if not all(key in self.info for key in ['ch_names', 'sfreq']):
            raise ValueError("info missing required keys: 'ch_names' or 'sfreq'")

        self.epoch_count = 0
        self.auc_values = []

        self.init_ui()
        self.connect_to_lsl_outlet()
        self.init_lsl_inlet_with_retry()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000)  # Update every second
        logging.info("VisualGuiWindow initialized")

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # Frequency settings widget
        self.freq_settings = self.create_freq_widget()

        # Plot widget for real-time AUC plotting
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setTitle("Real-Time Power AUC")
        self.plot_widget.setLabel('left', 'AUC')
        self.plot_widget.setLabel('bottom', 'Epochs (1 sec each)')
        self.plot_data = self.plot_widget.plot(pen='y')

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.launch_selected_visual)

        main_layout.addWidget(self.freq_settings)
        main_layout.addWidget(QLabel("Real-Time PSD AUC Plot:"))
        main_layout.addWidget(self.plot_widget)
        main_layout.addWidget(self.next_button)
        main_layout.addStretch()

        self.setCentralWidget(main_widget)

        self.setLayout(main_layout)

    def create_freq_widget(self):
        box = QGroupBox("Frequency Settings")
        layout = QVBoxLayout()

        self.df_channel = QComboBox()
        self.df_channel.addItems(self.info['ch_names'])

        self.df_band1 = QComboBox()
        self.df_band1.addItems(["Delta", "Theta", "Alpha", "Beta", "Gamma"])

        self.df_low1 = QDoubleSpinBox()
        self.df_low1.setRange(0, 100)
        self.df_low1.setValue(4.0)

        self.df_high1 = QDoubleSpinBox()
        self.df_high1.setRange(0, 100)
        self.df_high1.setValue(8.0)

        self.df_band2 = QComboBox()
        self.df_band2.addItems(["Delta", "Theta", "Alpha", "Beta", "Gamma"])

        self.df_low2 = QDoubleSpinBox()
        self.df_low2.setRange(0, 100)
        self.df_low2.setValue(30.0)

        self.df_high2 = QDoubleSpinBox()
        self.df_high2.setRange(0, 100)
        self.df_high2.setValue(45.0)

        self.df_analysis = QComboBox()
        self.df_analysis.addItems(["Coherence", "Phase Amplitude Coupling", "Both"])

        self.df_mode = QComboBox()
        self.df_mode.addItems(["Simple (Coherence)", "Simple (PAC)", "Simple (Both)"])

        layout.addWidget(QLabel("Select Channel:"))
        layout.addWidget(self.df_channel)
        layout.addWidget(QLabel("Band 1:"))
        layout.addWidget(self.df_band1)
        layout.addWidget(QLabel("Low Frequency 1:"))
        layout.addWidget(self.df_low1)
        layout.addWidget(QLabel("High Frequency 1:"))
        layout.addWidget(self.df_high1)
        layout.addWidget(QLabel("Band 2:"))
        layout.addWidget(self.df_band2)
        layout.addWidget(QLabel("Low Frequency 2:"))
        layout.addWidget(self.df_low2)
        layout.addWidget(QLabel("High Frequency 2:"))
        layout.addWidget(self.df_high2)
        layout.addWidget(QLabel("Analysis:"))
        layout.addWidget(self.df_analysis)
        layout.addWidget(QLabel("Mode:"))
        layout.addWidget(self.df_mode)

        box.setLayout(layout)
        return box

    def init_lsl_inlet_with_retry(self, retries=5, delay=1.0):
        import time
        for attempt in range(retries):
            try:
                streams = resolve_streams(timeout=5.0)
                target_stream = next((s for s in streams if s.name == self.lsl_stream_name), None)
                if target_stream is None:
                    raise RuntimeError(f"LSL stream '{self.lsl_stream_name}' not found")
                self.inlet = StreamInlet(target_stream)
                logging.info("LSL inlet initialized")
                return
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} to init LSL inlet failed: {e}")
                time.sleep(delay)
        # After retries failed
        err_msg = f"Failed to initialize LSL inlet after {retries} attempts"
        logging.error(err_msg)
        QMessageBox.critical(self, "Error", err_msg)
        self.close()

    def connect_to_lsl_outlet(self):
        try:
            streams = resolve_streams(timeout=5.0)
            target_stream = next((s for s in streams if s.name == 'PSD_AUC_Stream'), None)
            if target_stream is None:
                raise RuntimeError("LSL stream 'PSD_AUC_Stream' not found")
            self.outlet = StreamOutlet(target_stream)
            logging.info("Connected to existing LSL outlet 'PSD_AUC_Stream'")
        except Exception as e:
            logging.error(f"Failed to connect to LSL outlet: {e}")
            QMessageBox.critical(self, "Error", str(e))
            self.close()

    def update_plot(self):
        try:
            channel = self.df_channel.currentText()
            band = self.df_band1.currentText()
            low = self.df_low1.value()
            high = self.df_high1.value()

            n_channels = len(self.info['ch_names'])
            n_samples = int(self.info['sfreq'])

            samples, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=n_samples)
            if not samples:
                logging.warning("No samples received")
                return

            data = np.array(samples).T
            # Check shape is valid
            if data.shape[0] != n_channels or data.shape[1] < 10:
                logging.warning(f"Invalid data shape received: {data.shape}")
                return

            # Use last n_samples samples to build data matrix
            data = data[:, -n_samples:]

            raw = preprocess_realtime_stream(
                data, self.info, self.rename_dict, self.bad_channels, self.asr, self.ica, self.artifact_components
            )
            epochs = mne.make_fixed_length_epochs(raw, duration=1.0, overlap=0)
            if len(epochs) == 0:
                logging.warning("No epochs created")
                return

            power_change, auc_value = compute_band_auc(
                epochs[0], [channel], self.epoch_count, self.auc_values, band, low, high
            )

            self.auc_values.append(auc_value)
            # Keep buffer size manageable
            if len(self.auc_values) > 300:
                self.auc_values.pop(0)

            self.outlet.push_sample([np.float32(auc_value)])

            self.epoch_count += 1
            self.plot_data.setData(np.array(self.auc_values))
            logging.debug("Plot updated")

        except Exception as e:
            logging.error(f"Plot update error: {e}")
            QMessageBox.warning(self, "Plot Error", str(e))

    def launch_selected_visual(self):
        try:
            self.open_psd_window()

        except Exception as e:
            logging.error(f"Launch error: {e}")
            QMessageBox.critical(self, "Error", str(e))

    def open_psd_window(self):
        try:
            from feedback_window import RealTimePsdGui
            self.psd_window = RealTimePsdGui(
                asr=self.asr,
                ica=self.ica,
                artifact_components=self.artifact_components,
                bad_channels=self.bad_channels,
                info=self.info
            )
            self.psd_window.show()
            self.close()
        except Exception as e:
            logging.error(f"PSD window error: {e}")
            QMessageBox.critical(self, "Error", str(e))
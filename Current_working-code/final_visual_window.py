import sys
import os
import numpy as np
import mne
import logging
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QComboBox, QDoubleSpinBox, QGroupBox, QRadioButton, QStackedWidget, QButtonGroup, QMessageBox
)
from PyQt6.QtCore import QTimer
from mne_lsl.lsl import StreamInfo, StreamOutlet, StreamInlet, resolve_streams
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
        self.create_lsl_outlet()
        self.init_lsl_inlet_with_retry()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000)  # Update every second
        logging.info("VisualGuiWindow initialized")

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # Radio buttons to select mode (single or dual frequency)
        self.mode_selector = QButtonGroup()
        radio_single = QRadioButton("Single Frequency")
        radio_dual = QRadioButton("Two Frequencies")
        radio_single.setChecked(True)
        self.mode_selector.addButton(radio_single, 0)
        self.mode_selector.addButton(radio_dual, 1)

        radio_layout = QHBoxLayout()
        radio_layout.addWidget(radio_single)
        radio_layout.addWidget(radio_dual)

        # Stack widget to switch between single and dual frequency settings
        self.stack = QStackedWidget()
        self.stack.addWidget(self.create_single_freq_widget())
        self.stack.addWidget(self.create_dual_freq_widget())

        self.mode_selector.buttonClicked[int].connect(lambda idx: self.stack.setCurrentIndex(idx))

        # Plot widget for real-time AUC plotting
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setTitle("Real-Time Power AUC")
        self.plot_widget.setLabel('left', 'AUC')
        self.plot_widget.setLabel('bottom', 'Epochs (1 sec each)')
        self.plot_data = self.plot_widget.plot(pen='y')

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.launch_selected_visual)

        main_layout.addLayout(radio_layout)
        main_layout.addWidget(self.stack)
        main_layout.addWidget(QLabel("Real-Time PSD AUC Plot:"))
        main_layout.addWidget(self.plot_widget)
        main_layout.addWidget(self.next_button)
        main_layout.addStretch()

        self.setCentralWidget(main_widget)

    def create_single_freq_widget(self):
        box = QGroupBox("Single Frequency Settings")
        layout = QVBoxLayout()

        self.sf_channel = QComboBox()
        self.sf_channel.addItems(self.info['ch_names'])

        self.sf_band = QComboBox()
        self.sf_band.addItems(["delta", "theta", "alpha", "beta", "gamma"])

        self.sf_low = QDoubleSpinBox()
        self.sf_low.setRange(0, 100)
        self.sf_low.setValue(8.0)

        self.sf_high = QDoubleSpinBox()
        self.sf_high.setRange(0, 100)
        self.sf_high.setValue(12.0)

        self.sf_mode = QComboBox()
        self.sf_mode.addItems(["Simple", "Game"])

        layout.addWidget(QLabel("Select Channel:"))
        layout.addWidget(self.sf_channel)
        layout.addWidget(QLabel("Frequency Band:"))
        layout.addWidget(self.sf_band)
        layout.addWidget(QLabel("Low Frequency:"))
        layout.addWidget(self.sf_low)
        layout.addWidget(QLabel("High Frequency:"))
        layout.addWidget(self.sf_high)
        layout.addWidget(QLabel("Mode:"))
        layout.addWidget(self.sf_mode)

        box.setLayout(layout)
        return box

    def create_dual_freq_widget(self):
        box = QGroupBox("Two Frequencies Settings")
        layout = QVBoxLayout()

        self.df_channel = QComboBox()
        self.df_channel.addItems(self.info['ch_names'])

        self.df_band1 = QComboBox()
        self.df_band1.addItems(["delta", "theta", "alpha", "beta", "gamma"])

        self.df_low1 = QDoubleSpinBox()
        self.df_low1.setRange(0, 100)
        self.df_low1.setValue(4.0)

        self.df_high1 = QDoubleSpinBox()
        self.df_high1.setRange(0, 100)
        self.df_high1.setValue(8.0)

        self.df_band2 = QComboBox()
        self.df_band2.addItems(["delta", "theta", "alpha", "beta", "gamma"])

        self.df_low2 = QDoubleSpinBox()
        self.df_low2.setRange(0, 100)
        self.df_low2.setValue(30.0)

        self.df_high2 = QDoubleSpinBox()
        self.df_high2.setRange(0, 100)
        self.df_high2.setValue(45.0)

        self.df_analysis = QComboBox()
        self.df_analysis.addItems(["coherence", "phase amplitude coupling", "both"])

        self.df_mode = QComboBox()
        self.df_mode.addItems(["Simple (coherence)", "Simple (PAC)", "Simple (both)"])

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

    def create_lsl_outlet(self):
        try:
            info = StreamInfo('PSD_AUC_Stream', 'Markers', 1, 1, 'float32', 'psdauc1234')
            if not info.is_valid():
                raise RuntimeError("Invalid StreamInfo")
            self.outlet = StreamOutlet(info)
            logging.info("LSL outlet created")
        except Exception as e:
            logging.error(f"Failed to create LSL outlet: {e}")
            QMessageBox.critical(self, "Error", str(e))
            self.close()

    def update_plot(self):
        try:
            is_single = self.mode_selector.checkedId() == 0
            channel = self.sf_channel.currentText() if is_single else self.df_channel.currentText()
            band = self.sf_band.currentText() if is_single else self.df_band1.currentText()
            low = self.sf_low.value() if is_single else self.df_low1.value()
            high = self.sf_high.value() if is_single else self.df_high1.value()

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

            power_change, auc_value = compute_band_auc_epochs(
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
            if self.mode_selector.checkedId() == 0:
                mode = self.sf_mode.currentText()
                channel = self.sf_channel.currentText()
                band = self.sf_band.currentText()
                low = self.sf_low.value()
                high = self.sf_high.value()

                if mode == "Simple":
                    run_simple_visualization(self.auc_values, freq_band=f"{band} ({low}-{high} Hz)")
                elif mode == "Game":
                    exe_path = "C:/Users/varsh/BuildGames/EEGCubeGame.exe"
                    if not os.path.exists(exe_path):
                        raise FileNotFoundError("Game executable not found at: " + exe_path)
                    launch_game_mode(exe_path)
            else:
                self.open_psd_window()

        except Exception as e:
            logging.error(f"Launch error: {e}")
            QMessageBox.critical(self, "Error", str(e))

    def open_psd_window(self):
        try:
            from final_power_window import RealTimePsdGui  # Import here if it's in separate file
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

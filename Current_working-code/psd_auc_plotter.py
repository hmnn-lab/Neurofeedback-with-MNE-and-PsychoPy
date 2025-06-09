import numpy as np
import mne
import logging
from mne_lsl.lsl import StreamInfo, StreamOutlet, StreamInlet, resolve_streams
import pyqtgraph as pg
from PyQt6.QtCore import QTimer
from preproc_apply import preprocess_realtime_stream
from power_auc_cal import compute_band_auc_epochs
from PyQt6.QtWidgets import QMessageBox

class PsdAucPlotter:
    def __init__(self, parent_widget, asr, ica, artifact_components, rename_dict, bad_channels, info, lsl_stream_name):
        self.asr = asr
        self.ica = ica
        self.artifact_components = artifact_components
        self.rename_dict = rename_dict
        self.bad_channels = bad_channels
        self.info = info
        self.lsl_stream_name = lsl_stream_name

        self.epoch_count = 0
        self.auc_values = []

        self.n_channels = len(info['ch_names'])
        self.sfreq = int(info['sfreq'])

        self.parent_widget = parent_widget

        # Setup plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setTitle("Real-Time Power AUC")
        self.plot_widget.setLabel('left', 'AUC')
        self.plot_widget.setLabel('bottom', 'Epochs (1 sec each)')
        self.plot_data = self.plot_widget.plot(pen='y')

        # Initialize LSL inlet/outlet
        self.init_lsl_inlet()
        self.create_lsl_outlet()

        # Setup timer for periodic update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000)  # every 1 second

    def init_lsl_inlet(self):
        try:
            streams = resolve_streams(timeout=5.0)
            target_stream = next((s for s in streams if s.name == self.lsl_stream_name), None)
            if target_stream is None:
                raise RuntimeError(f"LSL stream '{self.lsl_stream_name}' not found")
            self.inlet = StreamInlet(target_stream)
            logging.info("LSL inlet initialized")
        except Exception as e:
            logging.error(f"Failed to initialize LSL inlet: {e}")
            QMessageBox.critical(self.parent_widget, "Error", f"Failed to initialize LSL inlet:\n{e}")

    def create_lsl_outlet(self):
        try:
            info = StreamInfo('PSD_AUC_Stream', 'Markers', 1, 1, 'float32', 'psdauc1234')
            if not info.is_valid():
                raise RuntimeError("Invalid StreamInfo")
            self.outlet = StreamOutlet(info)
            logging.info("LSL outlet created")
        except Exception as e:
            logging.error(f"Failed to create LSL outlet: {e}")
            QMessageBox.critical(self.parent_widget, "Error", f"Failed to create LSL outlet:\n{e}")

    def update_plot(self):
        try:
            channel = self.info['ch_names'][0]  # You can make this dynamic
            band = "alpha"
            low = 8.0
            high = 12.0

            samples, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=self.sfreq)
            if not samples:
                logging.warning("No samples received")
                return

            data = np.array(samples).T
            if data.shape[0] != self.n_channels or data.shape[1] < 10:
                logging.warning(f"Invalid shape: {data.shape}")
                return
            data = data[:, -self.sfreq:]

            raw = preprocess_realtime_stream(
                data, self.info, self.rename_dict, self.bad_channels, self.asr, self.ica, self.artifact_components)
            epochs = mne.make_fixed_length_epochs(raw, duration=1.0, overlap=0)
            if len(epochs) == 0:
                logging.warning("No epochs created")
                return

            power_change, auc_value = compute_band_auc_epochs(
                epochs[0], [channel], self.epoch_count, self.auc_values, band, low, high)
            self.auc_values.append(auc_value)
            if len(self.auc_values) > 300:
                self.auc_values.pop(0)

            self.outlet.push_sample([np.float32(auc_value)])

            self.epoch_count += 1
            self.plot_data.setData(np.array(self.auc_values))
            logging.debug("Plot updated")

        except Exception as e:
            logging.error(f"Plot update error: {e}")
            QMessageBox.warning(self.parent_widget, "Plot Error", str(e))

    def get_plot_widget(self):
        return self.plot_widget

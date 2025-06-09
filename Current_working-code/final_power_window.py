import sys
import numpy as np
import mne
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QDoubleSpinBox, QVBoxLayout, QHBoxLayout, QMessageBox
)
from PyQt6.QtCore import QTimer
import pyqtgraph as pg

from preproc_apply import preprocess_realtime_stream
from power_auc_cal import compute_band_auc
from mne_lsl.lsl import resolve_streams, StreamOutlet, StreamInfo
from final_visual_window import VisualGui
from threading_stream import StreamThread  

mne.set_log_level('ERROR')


class PsdAucPlotter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.auc_values = []
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setTitle("Real-Time Power AUC")
        self.plot_widget.setLabel('left', 'AUC')
        self.plot_widget.setLabel('bottom', 'Epoch')
        self.plot_data = self.plot_widget.plot(pen='y')

        layout = QVBoxLayout(self)
        layout.addWidget(self.plot_widget)

    def update_plot_data(self, new_value):
        self.auc_values.append(new_value)
        self.plot_data.setData(np.array(self.auc_values))

    def get_auc_values(self):
        return self.auc_values


class RealTimePsdGui(QMainWindow):
    def __init__(self, asr, ica, artifact_components, bad_channels, info, plotter=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Real-Time PSD AUC")
        self.setFixedSize(800, 550)

        self.asr = asr
        self.ica = ica
        self.artifact_components = artifact_components
        self.bad_channels = bad_channels
        self.info = info

        self.lsl_stream_name = None
        self.stream_thread = None
        self.epoch_count = 0

        self.plotter = plotter or PsdAucPlotter()

        self.init_ui()
        self.populate_streams()
        # Create LSL output stream for power change values (1 channel)
        output_stream_name = "PowerChangeStream"
        epoch_duration = 1.0  # seconds, should match your epoch duration
        power_info = StreamInfo(
            name=output_stream_name,
            stype='PowerChange',
            n_channels=1,
            sfreq=1.0 / epoch_duration,
            dtype='float32',
            source_id='power_change_stream'
        )
        self.outlet = StreamOutlet(power_info)
        print(f"Streaming power change values in 'PowerChangeStream'...")

    def init_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)

        form_layout = QVBoxLayout()
        form_layout.addWidget(QLabel("Real-Time PSD AUC"))

        self.stream_selector = QComboBox()
        form_layout.addWidget(QLabel("Select LSL Stream:"))
        form_layout.addWidget(self.stream_selector)

        self.channel_selector = QComboBox()
        self.channel_selector.addItems(self.info['ch_names'])
        form_layout.addWidget(QLabel("Select Channel:"))
        form_layout.addWidget(self.channel_selector)

        self.band_selector = QComboBox()
        self.band_selector.addItems(["Delta", "Theta", "Alpha", "Beta", "Gamma"])
        form_layout.addWidget(QLabel("Band Name:"))
        form_layout.addWidget(self.band_selector)

        self.low_freq_input = QDoubleSpinBox()
        self.low_freq_input.setRange(0, 100)
        self.low_freq_input.setValue(8.0)
        form_layout.addWidget(QLabel("Low Freq:"))
        form_layout.addWidget(self.low_freq_input)

        self.high_freq_input = QDoubleSpinBox()
        self.high_freq_input.setRange(0, 100)
        self.high_freq_input.setValue(12.0)
        form_layout.addWidget(QLabel("High Freq:"))
        form_layout.addWidget(self.high_freq_input)

        self.start_button = QPushButton("Start Streaming")
        self.start_button.clicked.connect(self.start_streaming)
        form_layout.addWidget(self.start_button)

        form_layout.addStretch()

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.open_next_window)
        form_layout.addWidget(self.next_button)

        main_layout.addLayout(form_layout, 1)
        main_layout.addWidget(self.plotter, 2)

        self.setCentralWidget(central_widget)

    def populate_streams(self):
        try:
            streams = resolve_streams(timeout=5.0)
            self.stream_selector.clear()
            for stream in streams:
                self.stream_selector.addItem(stream.name)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to resolve LSL streams: {e}")

    def start_streaming(self):
        if self.stream_thread:
            self.stream_thread.stop()
            self.stream_thread = None

        self.lsl_stream_name = self.stream_selector.currentText()
        self.epoch_count = 0
        self.plotter.auc_values.clear()
        self.plotter.plot_data.clear()

        self.stream_thread = StreamThread(
            lsl_stream_name=self.lsl_stream_name,
            info=self.info,
            epoch_duration=1.0
        )
        self.stream_thread.new_epoch.connect(self.process_epoch)
        self.stream_thread.start()

    def process_epoch(self, epoch_data):
        """
        Slot to receive each epoch as (channels, samples) ndarray from StreamThread.
        Applies preprocessing, computes power AUC, updates plot and streams output.
        """
    
        print("Received epoch_data type:", type(epoch_data))
        print("epoch_data content/shape:", epoch_data if isinstance(epoch_data, np.ndarray) else str(epoch_data))
        # then your existing code

        try:
            raw_processed = preprocess_realtime_stream(
                data=epoch_data,
                client_info=self.info,
                bad_channels=self.bad_channels,
                asr=self.asr,
                ica=self.ica,
                artifact_components=self.artifact_components
            )
            print(f"Preprocessing complete. Processed raw data shape: {raw_processed.get_data().shape}")

            channel = self.channel_selector.currentText()
            events = np.array([[0, 0, 1]])

            epoch = mne.Epochs(
                raw_processed,
                events,
                event_id=1,
                tmin=0,
                tmax=1.0 - 1.0 / self.info['sfreq'],
                baseline=None,
                preload=True
            )
            print(f"Created epoch with shape: {epoch.get_data().shape}")

            if len(epoch) == 0:
                return

            single_epoch = epoch[0]
            print(f"Processing single epoch with shape: {single_epoch.get_data().shape}")
            
            result = compute_band_auc(
                        epoch=single_epoch,
                        ch_names=[channel],
                        epoch_count=self.epoch_count,
                        band_name=self.band_selector.currentText(),
                        low_freq=self.low_freq_input.value(),
                        high_freq=self.high_freq_input.value()
                    )
            
            print("compute_band_auc returned:", result)
            power_change, band_auc = result
            print(f"Power Change: {power_change}, Band AUC: {band_auc}")
            

            self.epoch_count += 1

            # Update plot with new power change value
            self.plotter.update_plot_data(band_auc)

            # If you want to push this power_change 
            self.outlet.push_sample([float(power_change)])

        except Exception as e:
            print(f"[Error processing epoch] {e}")

    def open_next_window(self):
        if self.stream_thread:
            self.stream_thread.stop()
            self.stream_thread = None
        self.close()
        self.next_window = VisualGui(
            asr=self.asr,
            ica=self.ica,
            artifact_components=self.artifact_components,
            bad_channels=self.bad_channels,
            info=self.info,
            lsl_stream_name=self.lsl_stream_name,
            plotter=self.plotter
        )
        self.next_window.show()

    def closeEvent(self, event):
        if self.stream_thread:
            self.stream_thread.stop()
        event.accept()

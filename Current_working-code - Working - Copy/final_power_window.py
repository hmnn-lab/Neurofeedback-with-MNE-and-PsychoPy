import sys
import numpy as np
import mne
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QDoubleSpinBox, QVBoxLayout, QHBoxLayout, QMessageBox, QListWidget, QListWidgetItem
)
from queue import Queue
import time
from PyQt6.QtCore import QMutex, QThread, QObject, pyqtSignal, pyqtSlot
import pyqtgraph as pg
from mne_lsl.lsl import resolve_streams, StreamOutlet, StreamInfo
from preproc_apply import preprocess_realtime_stream
from power_auc_cal import compute_band_auc
from final_visual_window import VisualGui
from threading_stream import StreamThread 
from single_freq_vis_utils import run_simple_visualization, launch_game_mode, DEFAULT_GAME_PATH


class EpochProcessor(QObject):
    result = pyqtSignal(list, list, int)
    error = pyqtSignal(str)

    def __init__(self, info, bad_channels, asr, ica, artifact_components):
        super().__init__()
        self.info = info
        self.bad_channels = bad_channels
        self.asr = asr
        self.ica = ica
        self.artifact_components = artifact_components
        self.selected_channels = []
        self.selected_ch_names = []
        self.band = ""
        self.low_freq = 0.0
        self.high_freq = 0.0
        self.epoch_queue = Queue()
        self.mutex = QMutex()
        self.running = True
        self.max_queue_size = 4  # Limit queue to 4 epochs (1 second of data at 4 Hz)

    @pyqtSlot(np.ndarray)
    def process_epoch(self, epoch_data):
        self.mutex.lock()
        if self.epoch_queue.qsize() < self.max_queue_size:
            self.epoch_queue.put(epoch_data)
            queue_size = self.epoch_queue.qsize()
            self.mutex.unlock()
            print(f"Epoch queued, current queue size: {queue_size}")
            self.process_next_epoch()
        else:
            self.mutex.unlock()
            print("Warning: Queue full, dropping epoch to prevent backlog")

    def process_next_epoch(self):
        while self.running:
            self.mutex.lock()
            if self.epoch_queue.empty():
                self.mutex.unlock()
                break
            epoch_data = self.epoch_queue.get()
            self.mutex.unlock()

            try:
                start_time = time.time()
                raw_processed = preprocess_realtime_stream(
                    data=epoch_data,
                    client_info=self.info,
                    bad_channels=self.bad_channels,
                    asr=self.asr,
                    ica=self.ica,
                    artifact_components=self.artifact_components
                )

                if not all(ch in raw_processed.ch_names for ch in self.selected_ch_names):
                    print(f"Some selected channels not found in raw data. Available: {raw_processed.ch_names}")
                    continue

                epochs = mne.make_fixed_length_epochs(raw_processed, duration=1.0, overlap=0, preload=True)
                epochs.drop_bad()
                if len(epochs) == 0:
                    print("No epochs created")
                    continue

                relative_auc, auc_values = compute_band_auc(
                    epoch=epochs[0],
                    ch_names=self.selected_ch_names,
                    epoch_count=0,
                    band_name=self.band,
                    low_freq=self.low_freq,
                    high_freq=self.high_freq
                )

                self.result.emit(relative_auc, auc_values, 0)
                processing_time = time.time() - start_time
                print(f"Epoch processed in {processing_time:.3f} seconds")

                if processing_time > 0.25:
                    print("Warning: Processing time exceeds step duration (0.25s), consider reducing channels or optimizing preprocessing")

                del raw_processed
                del epochs

            except Exception as e:
                self.error.emit(str(e))

    def stop(self):
        self.running = False

# Modified RealTimePsdGui
class RealTimePsdGui(QMainWindow):
    def __init__(self, asr, ica, artifact_components, bad_channels, info, parent=None):
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
        self.auc_values = {}
        self.plot_data = {}

        # Plotting setup
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setTitle("Real-Time Power AUC")
        self.plot_widget.setLabel('left', 'AUC')
        self.plot_widget.setLabel('bottom', 'Epoch')

        # Worker thread setup
        self.worker_thread = QThread()
        self.worker = EpochProcessor(info, bad_channels, asr, ica, artifact_components)
        self.worker.moveToThread(self.worker_thread)
        self.worker.result.connect(self.update_plot)
        self.worker.error.connect(self.handle_error)
        self.worker_thread.start()

        self.init_ui()
        self.populate_streams()

        self.outlet = None

    def start_streaming(self):
        if self.stream_thread:
            self.stream_thread.stop()
            self.stream_thread = None

        self.lsl_stream_name = self.stream_selector.currentText()
        self.epoch_count = 0
        self.auc_values.clear()
        self.plot_data.clear()
        self.plot_widget.clear()

        selected_items = self.channel_selector.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select at least one channel.")
            return
        self.selected_channels = [int(item.text().split()[1]) for item in selected_items]
        self.selected_ch_names = [self.info['ch_names'][idx] for idx in self.selected_channels]

        self.worker.selected_channels = self.selected_channels
        self.worker.selected_ch_names = self.selected_ch_names
        self.worker.band = self.band_selector.currentText()
        self.worker.low_freq = self.low_freq_input.value()
        self.worker.high_freq = self.high_freq_input.value()

        for idx in self.selected_channels:
            self.auc_values[idx] = []
            self.plot_data[idx] = self.plot_widget.plot(pen=pg.mkPen(color=(np.random.randint(50, 200), np.random.randint(50, 200), np.random.randint(50, 200)), width=2))

        output_stream_name = "PowerChangeStream"
        power_info = StreamInfo(
            name=output_stream_name,
            stype='PowerChange',
            n_channels=len(self.selected_channels),
            sfreq=0,
            dtype='float32',
            source_id='power_change_stream'
        )
        power_info.set_channel_names([f"Ch{idx}" for idx in self.selected_channels])
        self.outlet = StreamOutlet(power_info)
        print(f"Streaming power change values for {len(self.selected_channels)} channels in 'PowerChangeStream'...")

        self.stream_thread = StreamThread(
            lsl_stream_name=self.lsl_stream_name,
            info=self.info,
            epoch_duration=1.0,
            step_duration=0.25  # Updated to 0.25 for 4 Hz emission
        )
        self.stream_thread.new_epoch.connect(self.worker.process_epoch)
        self.stream_thread.start()

    @pyqtSlot(list, list, int)
    def update_plot(self, relative_auc, auc_values, _):
        try:
            for idx, auc_value in enumerate(auc_values):
                ch_idx = self.selected_channels[idx]
                self.auc_values[ch_idx].append(auc_value)
                if len(self.auc_values[ch_idx]) > 300:
                    self.auc_values[ch_idx].pop(0)

                if np.all(np.isfinite(self.auc_values[ch_idx])):
                    xdata = np.arange(max(0, len(self.auc_values[ch_idx]) - 300), len(self.auc_values[ch_idx]))
                    ydata = np.array(self.auc_values[ch_idx][-300:])
                    self.plot_data[ch_idx].setData(x=xdata, y=ydata)
                    print(f"Updated plot for channel {ch_idx} with AUC: {auc_value}")

            if relative_auc:
                relative_auc_array = np.array(relative_auc, dtype=np.float32)
                print(f"Pushing sample with shape: {relative_auc_array.shape}, values: {relative_auc_array}")
                self.outlet.push_sample(relative_auc_array)

            self.epoch_count += 1

        except Exception as e:
            print(f"[Error updating plot] {e}")

    @pyqtSlot(str)
    def handle_error(self, error_msg):
        print(f"[Error from worker] {error_msg}")
        QMessageBox.critical(self, "Processing Error", f"Error processing epoch: {error_msg}")

    def stop_streaming(self):
        if self.stream_thread:
            self.stream_thread.stop()
            self.stream_thread = None
        self.outlet = None
        print("Streaming stopped.")

    def closeEvent(self, event):
        self.stop_streaming()
        self.worker_thread.quit()
        self.worker_thread.wait()
        event.accept()

    def init_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)

        form_layout = QVBoxLayout()
        form_layout.addWidget(QLabel("Real-Time PSD AUC"))

        form_layout.addWidget(QLabel("Select LSL Stream:"))
        stream_selector_layout = QHBoxLayout()

        self.stream_selector = QComboBox()
        stream_selector_layout.addWidget(self.stream_selector)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setFixedWidth(100)
        self.refresh_button.clicked.connect(self.populate_streams)
        stream_selector_layout.addWidget(self.refresh_button)

        form_layout.addLayout(stream_selector_layout)


        # Replace channel_selector with QListWidget for multi-selection
        self.channel_selector = QListWidget()
        self.channel_selector.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for idx in range(len(self.info['ch_names'])):
            item = QListWidgetItem(f"Channel {idx} ({self.info['ch_names'][idx]})")
            self.channel_selector.addItem(item)
        form_layout.addWidget(QLabel("Select Channels (Multiple):"))
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

        # Mode Selector
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Simple", "Game"])
        form_layout.addWidget(QLabel("Visualization Mode:"))
        form_layout.addWidget(self.mode_selector)

        # Start and Stop Buttons
        self.start_button = QPushButton("Start Streaming")
        self.start_button.clicked.connect(self.start_streaming)
        form_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Streaming")
        self.stop_button.clicked.connect(self.stop_streaming)
        form_layout.addWidget(self.stop_button)
        form_layout.addStretch()

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.open_next_window)
        form_layout.addWidget(self.next_button)

        main_layout.addLayout(form_layout, 1)
        main_layout.addWidget(self.plot_widget, 2)

        self.setCentralWidget(central_widget)

        self.setLayout(main_layout)

    def populate_streams(self):
        try:
            streams = resolve_streams(timeout=5.0)
            self.stream_selector.clear()
            for stream in streams:
                self.stream_selector.addItem(stream.name)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to resolve LSL streams: {e}")

    def open_next_window(self):
        selected_band = self.band_selector.currentText()
        mode = self.mode_selector.currentText()

        if mode == "Simple":
            print("Launching Simple Mode Visualization...")
            try:
                run_simple_visualization(
                    power_change_stream="PowerChangeStream",  # Replace with actual power stream if needed
                    band_name=selected_band,
                    monitor_name="TestMonitor"
                )
            except Exception as e:
                QMessageBox.critical(self, "Visualization Error", f"Failed to launch simple visualization: {e}")

        elif mode == "Game":
            print("Launching Game Mode...")
            try:
                launch_game_mode(DEFAULT_GAME_PATH)
            except Exception as e:
                QMessageBox.critical(self, "Game Launch Failed", f"Could not launch game: {e}")

        else:
            QMessageBox.warning(self, "Invalid Mode", "Please select a valid visualization mode.")

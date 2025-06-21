import sys
import numpy as np
import mne
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QLabel, QComboBox, QStatusBar
)
from PyQt6.QtCore import QMutex, QThread, QObject, pyqtSignal, pyqtSlot, Qt
from queue import Queue
import time
import pyqtgraph as pg
from mne_lsl.lsl import resolve_streams, StreamOutlet, StreamInfo
from preproc_apply import preprocess_realtime_stream
from power_auc_cal import compute_band_auc
from dual_freq_auc import dual_freq_auc
from pac_cal import pac_cal
from threading_stream import StreamThread
from single_freq_vis_utils import launch_game_mode, DEFAULT_GAME_PATH


class EpochProcessor(QObject):
    result = pyqtSignal(list, list, int)
    error = pyqtSignal(str)

    def __init__(self, info, bad_channels, asr, ica, artifact_components, selected_modality):
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
        self.band2 = ""  # Second band initialized like band
        self.low_freq2 = 0.0  # Second low frequency initialized like low_freq
        self.high_freq2 = 0.0  # Second high frequency initialized like high_freq
        self.epoch_queue = Queue()
        self.mutex = QMutex()
        self.running = True
        self.max_queue_size = 4
        self.selected_modality = selected_modality

    @pyqtSlot(np.ndarray)
    def process_epoch(self, epoch_data):
        self.mutex.lock()
        if self.epoch_queue.qsize() < self.max_queue_size:
            self.epoch_queue.put(epoch_data)
            queue_size = self.epoch_queue.qsize()
            self.mutex.unlock()
            print(f"Epoch queued, current queue size: {queue_size}")
            # Instead of calling process_next_epoch directly which could block,
            # consider using a QTimer.singleShot if you want to ensure it's
            # processed on the worker thread's event loop, or just let the
            # thread's event loop handle the loop.
            # For now, keeping the direct call as it works if the thread
            # is properly set up to handle the loop.
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

                if not isinstance(raw_processed, mne.io.BaseRaw):
                    raise TypeError("preprocess_realtime_stream did not return an MNE Raw object.")

                epochs = mne.make_fixed_length_epochs(raw_processed, duration=1.0, overlap=0, preload=True)
                epochs.drop_bad()
                if len(epochs) == 0:
                    print("No epochs created after dropping bad ones.")
                    continue

                primary_metric = None
                plot_value = None

                if self.selected_modality == "psd_auc":
                    result_data = compute_band_auc(
                        epoch=epochs[0],
                        ch_names=self.selected_ch_names,
                        epoch_count=0,
                        band_name=self.band,
                        low_freq=self.low_freq,
                        high_freq=self.high_freq
                    )
                    if isinstance(result_data, tuple) and len(result_data) == 2:
                        relative_auc, auc_values = result_data
                        print(f"psd_auc: selected_ch_names={self.selected_ch_names}, relative_auc={relative_auc}, type={type(relative_auc)}")
                        if isinstance(relative_auc, (list, np.ndarray)):
                            if len(relative_auc) != len(self.selected_ch_names):
                                raise ValueError(f"relative_auc length ({len(relative_auc)}) does not match selected channels ({len(self.selected_ch_names)})")
                            primary_metric = float(relative_auc[0])  # Single channel
                        else:
                            primary_metric = float(relative_auc)  # Already a scalar
                        plot_value = primary_metric
                    else:
                        print(f"[WARNING] compute_band_auc returned unexpected format: {result_data}")
                        primary_metric = float(result_data)
                        plot_value = primary_metric

                elif self.selected_modality == "psd_ratio":
                    result_data = dual_freq_auc(
                        epoch=epochs[0],
                        ch_name=self.selected_ch_names[0],
                        epoch_count=0,
                        band_name_1=self.band,
                        low_freq_1=self.low_freq,
                        high_freq_1=self.high_freq,
                        band_name_2=self.band2,
                        low_freq_2=self.low_freq2,
                        high_freq_2=self.high_freq2
                    )
                    primary_metric = float(result_data)
                    plot_value = primary_metric

                elif self.selected_modality == "pac":
                    result_data = pac_cal(
                        epoch=epochs[0],
                        ch_name=self.selected_ch_names[0],
                        epoch_count=0,
                        band_name_1=self.band,
                        low_freq_1=self.low_freq,
                        high_freq_1=self.high_freq,
                        band_name_2=self.band2,
                        low_freq_2=self.low_freq2,
                        high_freq_2=self.high_freq2
                    )
                    primary_metric = float(result_data)
                    plot_value = primary_metric

                elif self.selected_modality == "coh":
                    raise ValueError("Coherence modality is not implemented.")

                print(f"Emitting for {self.selected_modality}: metric={primary_metric}, plot_value={plot_value}")
                self.result.emit([primary_metric], [plot_value], 0)

                processing_time = time.time() - start_time
                print(f"Epoch processed in {processing_time:.3f} seconds")

                if processing_time > 0.25:
                    print("Warning: Processing time exceeds step duration (0.25s), consider reducing channels or optimizing preprocessing")

                del raw_processed
                del epochs

            except Exception as e:
                self.error.emit(f"Error during {self.selected_modality} processing for channels {self.selected_ch_names}: {e}")

    def stop(self):
        self.running = False


class RealTimePsdGui(QMainWindow):
    # New signal for going back to the paradigm selection window
    back_to_paradigm_signal = pyqtSignal()

    def __init__(self, asr, ica, artifact_components, bad_channels, info, parameters=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Real-Time Feedback")
        self.setFixedSize(800, 550) # Use setFixedSize for a non-resizable window

        self.asr = asr
        self.ica = ica
        self.artifact_components = artifact_components
        self.bad_channels = bad_channels
        self.info = info # MNE Info object
        self.lsl_stream_name = None # Will be set by populate_streams or configure_parameters
        self.stream_thread = None
        self.epoch_count = 0
        self.modality_values = {} # Stores historical values for plotting per channel
        self.plot_data = {}       # Stores PlotDataItem objects for each channel

        # --- Initialize attributes with safe defaults FIRST ---
        self.selected_modality = 'psd_auc' # Default, will be overwritten by configure_parameters
        self.selected_channels = [] # List of MNE channel indices (integers)
        self.selected_ch_names = [] # List of MNE channel names (strings)
        self.band = ""
        self.low_freq = 0.0
        self.high_freq = 0.0
        self.band2 = ""
        self.low_freq2 = 0.0
        self.high_freq2 = 0.0
        self.outlet = None # LSL outlet for pushing feedback values

        # Worker thread setup
        # Pass the initial (default) selected_modality to EpochProcessor.
        # Its other parameters will be set by configure_parameters.
        self.worker_thread = QThread()
        # Initialize worker with preprocessing objects and a default modality.
        self.worker = EpochProcessor(info, bad_channels, asr, ica, artifact_components, self.selected_modality)
        self.worker.moveToThread(self.worker_thread)
        self.worker.result.connect(self.update_plot)
        self.worker.error.connect(self.handle_error)
        self.worker_thread.start() # Start the worker thread's event loop

        # --- Call configure_parameters *after* worker is created and defaults are set ---
        # This method will correctly set all relevant self. attributes (selected_channels, etc.)
        # AND then apply them to self.worker.
        if parameters:
            self.configure_parameters(parameters)
        else:
            # Fallback for direct testing or if parameters are somehow missing.
            # Ensure worker gets some sensible defaults.
            print("[WARNING] RealTimePsdGui initialized without parameters. Using default values.")
            # Default values for selected_channels and selected_ch_names for testing
            if self.info and self.info['ch_names']:
                self.selected_channels = [0] # Default to first channel
                self.selected_ch_names = [self.info['ch_names'][0]]
            else:
                self.selected_channels = [0] # Placeholder if no info is available
                self.selected_ch_names = ["EEG 001"]

            self.band = 'Alpha'
            self.low_freq = 8.0
            self.high_freq = 12.0
            self.band2 = 'Gamma' # For dual band testing
            self.low_freq2 = 30.0
            self.high_freq2 = 50.0
            self.lsl_stream_name = 'DummyStream' # Default LSL stream name if not configured

            # Apply these default values to the worker
            self.worker.selected_modality = self.selected_modality
            self.worker.selected_channels = self.selected_channels
            self.worker.selected_ch_names = self.selected_ch_names
            self.worker.band = self.band
            self.worker.low_freq = self.low_freq
            self.worker.high_freq = self.high_freq
            self.worker.band2 = self.band2
            self.worker.low_freq2 = self.low_freq2
            self.worker.high_freq2 = self.high_freq2

        self.init_ui() # Initialize UI elements, including plot_widget

        # The `StatusBar` needs to be initialized. `QMainWindow` has one by default.
        self.setStatusBar(QStatusBar())


    def init_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)

        left_panel_layout = QVBoxLayout()
        left_panel_layout.setAlignment(Qt.AlignmentFlag.AlignTop) # Align elements to the top

        # Title
        title_label = QLabel("Real-time Feedback")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 20px;")
        left_panel_layout.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # LSL Stream selection
        left_panel_layout.addWidget(QLabel("Select LSL Stream:"))
        stream_selector_layout = QHBoxLayout()
        self.stream_selector = QComboBox()
        stream_selector_layout.addWidget(self.stream_selector)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setFixedWidth(100)
        self.refresh_button.clicked.connect(self.populate_streams)
        stream_selector_layout.addWidget(self.refresh_button)
        left_panel_layout.addLayout(stream_selector_layout)

        # Button styles
        default_button_style = """
            QPushButton {
                font-size: 14px;
                padding: 10px 20px;
                border-radius: 5px;
                background-color: #4CAF50;
                color: white;
                border: none;
                min-width: 150px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """
        game_button_style = """
            QPushButton {
                font-size: 14px;
                padding: 10px 20px;
                border-radius: 5px;
                background-color: #2196F3;
                color: white;
                border: none;
                min-width: 150px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #1e88e5;
            }
            QPushButton:pressed {
                background-color: #1976d2;
            }
        """
        back_button_style = """
            QPushButton {
                font-size: 14px;
                padding: 10px 20px;
                border-radius: 5px;
                background-color: #dc3545; /* Red for Stop/Back */
                color: white;
                border: none;
                min-width: 150px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:pressed {
                background-color: #bd2130;
            }
        """

        self.start_button = QPushButton("Start Streaming")
        self.start_button.setStyleSheet(default_button_style)
        self.start_button.clicked.connect(self.start_streaming)
        left_panel_layout.addWidget(self.start_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.stop_button = QPushButton("Stop Streaming")
        self.stop_button.setStyleSheet(back_button_style) # Using a slightly different style for stop
        self.stop_button.clicked.connect(self.stop_streaming)
        left_panel_layout.addWidget(self.stop_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.next_button = QPushButton("Launch Game")
        self.next_button.setStyleSheet(game_button_style)
        self.next_button.clicked.connect(self.open_next_window)
        left_panel_layout.addWidget(self.next_button, alignment=Qt.AlignmentFlag.AlignCenter)

        # Add a "Back to Paradigm" button
        self.back_button = QPushButton("Back to Paradigm")
        self.back_button.setStyleSheet(back_button_style)
        self.back_button.clicked.connect(self.emit_back_signal)
        left_panel_layout.addWidget(self.back_button, alignment=Qt.AlignmentFlag.AlignCenter)


        left_panel_layout.addStretch() # Pushes content to the top

        main_layout.addLayout(left_panel_layout, 1) # Left panel takes 1/3 of space
        
        # Plotting setup moved here to init_ui, ensuring plot_widget exists when used
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setTitle("Real-Time Feedback")
        self.plot_widget.setLabel('left', 'Modality Value')
        self.plot_widget.setLabel('bottom', 'Epoch')
        self.plot_widget.addLegend() # Ensure legend is added once

        main_layout.addWidget(self.plot_widget, 2) # Plot widget takes 2/3 of space

        self.setCentralWidget(central_widget)

        # Populate streams once UI is ready
        self.populate_streams()


    def start_streaming(self):
        try:
            if self.stream_thread:
                self.stop_streaming() # Stop any existing stream cleanly

            self.lsl_stream_name = self.stream_selector.currentText()
            if not self.lsl_stream_name:
                QMessageBox.warning(self, "No Stream Selected", "Please select an LSL stream to start streaming.")
                return

            self.epoch_count = 0
            # Clear historical data and plots for a new run
            self.modality_values.clear()
            self.plot_widget.clear() # Clears all plot items
            self.plot_data.clear() # Clear references to old plot items

            # Initialize plot data structures for selected channels
            # IMPORTANT: self.selected_channels and self.selected_ch_names must be populated
            # by configure_parameters *before* start_streaming is called.
            if not self.selected_channels:
                QMessageBox.warning(self, "No Channels Selected",
                                    "No channels are selected for plotting. Please ensure parameters were configured correctly.")
                return

            for idx, ch_idx_val in enumerate(self.selected_channels):
                # Use a more descriptive name for the plot legend
                plot_name = self.selected_ch_names[idx] if idx < len(self.selected_ch_names) else f"Ch{ch_idx_val}"
                self.modality_values[ch_idx_val] = [] # Initialize list for this channel's values
                self.plot_data[ch_idx_val] = self.plot_widget.plot(
                    name=plot_name,
                    pen=pg.mkPen(
                        color=(np.random.randint(50, 200), np.random.randint(50, 200), np.random.randint(50, 200)),
                        width=2
                    )
                )

            # Check for dual-frequency parameters if applicable
            if self.selected_modality in ["psd_ratio", "pac"]: # Using backend names
                if not self.worker.band2 or self.worker.low_freq2 <= 0 or self.worker.high_freq2 <= 0:
                    QMessageBox.critical(self, "Configuration Error",
                                         "Second band and frequency range must be set for dual-frequency ratio or PAC.")
                    return

            output_stream_name = "FeedbackStream"
            power_info = StreamInfo(
                name=output_stream_name,
                stype='FeedbackValue',
                n_channels=len(self.selected_channels), # Number of channels for output stream
                sfreq=0, # Or a more appropriate rate if known (e.g., 1.0 for 1 epoch/sec)
                dtype='float32',
                source_id='feedback_value_source'
            )
            power_info.set_channel_names(self.selected_ch_names) # Use selected_ch_names for outlet channels
            self.outlet = StreamOutlet(power_info)
            print(f"Streaming {self.selected_modality} values for {len(self.selected_channels)} channels in '{output_stream_name}'...")

            self.stream_thread = StreamThread(
                lsl_stream_name=self.lsl_stream_name,
                info=self.info, # Pass the MNE Info object
                epoch_duration=1.0, # Length of one epoch in seconds
                step_duration=0.25   # How often to get new data (e.g., 4 times per epoch)
            )
            # Connect the signal from StreamThread to the worker's slot
            self.stream_thread.new_epoch.connect(self.worker.process_epoch)
            self.stream_thread.start()
            self.statusBar().showMessage(f"Streaming from '{self.lsl_stream_name}' for {self.selected_modality}.")

        except Exception as e:
            QMessageBox.critical(self, "Streaming Error", f"Failed to start streaming: {e}")
            self.statusBar().showMessage(f"Streaming error: {e}")

    @pyqtSlot(list, list, int)
    def update_plot(self, metric_values, channel_values, _):
        try:
            if len(channel_values) != len(self.selected_channels):
                print(f"[WARNING] Mismatch in channel_values length ({len(channel_values)}) and selected_channels length ({len(self.selected_channels)}).")
                return

            for i, value in enumerate(channel_values):
                ch_idx = self.selected_channels[i]
                if ch_idx not in self.modality_values:
                    print(f"[WARNING] Modality values for channel {ch_idx} not initialized.")
                    continue
                self.modality_values[ch_idx].append(value)
                if len(self.modality_values[ch_idx]) > 300:
                    self.modality_values[ch_idx].pop(0)
                if ch_idx in self.plot_data:
                    xdata = np.arange(len(self.modality_values[ch_idx]))
                    ydata = np.array(self.modality_values[ch_idx])
                    self.plot_data[ch_idx].setData(x=xdata, y=ydata)
                else:
                    print(f"[WARNING] Plot item for channel {ch_idx} not found.")

            if self.outlet and metric_values:
                metric_array = np.array(metric_values, dtype=np.float32)
                if metric_array.shape[0] == self.outlet.n_channels:
                    self.outlet.push_sample(metric_array)
                else:
                    print(f"[WARNING] Metric array shape mismatch for LSL outlet. Expected {self.outlet.n_channels}, Got {metric_array.shape[0]}.")

            self.epoch_count += 1

        except Exception as e:
            print(f"[Error in update_plot] {e}")
            import traceback
            traceback.print_exc()

    @pyqtSlot(dict)
    def configure_parameters(self, parameters):
        try:
            self.selected_modality = parameters['modality']
            if self.selected_modality not in ['psd_auc', 'psd_ratio', 'pac', 'coh']:
                raise ValueError(f"Unsupported modality: {self.selected_modality}")

            if self.selected_modality in ['psd_auc', 'psd_ratio', 'pac']:
                ch_name = parameters['channel']
                if ch_name not in self.info['ch_names']:
                    raise ValueError(f"Selected channel {ch_name} not found in info['ch_names']")
                self.selected_channels = [self.info['ch_names'].index(ch_name)]
                self.selected_ch_names = [ch_name]
                print(f"Configured {self.selected_modality} with channel: {ch_name}")
            elif self.selected_modality == 'coh':
                ch_name1 = parameters['channel_1']
                ch_name2 = parameters['channel_2']
                self.selected_channels = [
                    self.info['ch_names'].index(ch_name1),
                    self.info['ch_names'].index(ch_name2)
                ]
                self.selected_ch_names = [ch_name1, ch_name2]
                print(f"Configured coh with channels: {self.selected_ch_names}")

            if self.selected_modality == 'psd_auc':
                self.band = parameters['frequency_band']['name']
                self.low_freq = parameters['frequency_band']['low']
                self.high_freq = parameters['frequency_band']['high']
                self.band2 = ""
                self.low_freq2 = 0.0
                self.high_freq2 = 0.0
            elif self.selected_modality == 'psd_ratio':
                self.band = parameters['band_1']['name']
                self.low_freq = parameters['band_1']['low']
                self.high_freq = parameters['band_1']['high']
                self.band2 = parameters['band_2']['name']
                self.low_freq2 = parameters['band_2']['low']
                self.high_freq2 = parameters['band_2']['high']
            elif self.selected_modality == 'pac':
                self.band = parameters['phase_frequency']['name']
                self.low_freq = parameters['phase_frequency']['low']
                self.high_freq = parameters['phase_frequency']['high']
                self.band2 = parameters['amplitude_frequency']['name']
                self.low_freq2 = parameters['amplitude_frequency']['low']
                self.high_freq2 = parameters['amplitude_frequency']['high']
            elif self.selected_modality == 'coh':
                self.band = parameters['frequency_band']['name']
                self.low_freq = parameters['frequency_band']['low']
                self.high_freq = parameters['frequency_band']['high']
                self.band2 = ""
                self.low_freq2 = 0.0
                self.high_freq2 = 0.0

            self.worker.selected_modality = self.selected_modality
            self.worker.selected_channels = self.selected_channels
            self.worker.selected_ch_names = self.selected_ch_names
            self.worker.band = self.band
            self.worker.low_freq = self.low_freq
            self.worker.high_freq = self.high_freq
            self.worker.band2 = self.band2
            self.worker.low_freq2 = self.low_freq2
            self.worker.high_freq2 = self.high_freq2

            self.lsl_stream_name = parameters.get('lsl_stream_name', 'default_stream')
            self.populate_streams()

            print(f"[DEBUG] Configured RealTimePsdGui with modality '{self.selected_modality}', channels: {self.selected_ch_names}")
        except Exception as e:
            print(f"[Error configuring parameters] {e}")
            self.statusBar().showMessage(f"Error configuring feedback: {str(e)}")


    @pyqtSlot(str)
    def handle_error(self, error_msg):
        print(f"[Error from worker] {error_msg}")
        self.statusBar().showMessage(f"Error: {error_msg}")
        # Optionally, stop streaming or disable buttons on critical error
        # self.stop_streaming()

    def closeEvent(self, event):
        print("RealTimePsdGui closing...")
        self.stop_streaming() # Ensure streaming is stopped
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker.stop() # Tell the worker to stop processing epochs
            self.worker_thread.quit() # Request thread to quit its event loop
            self.worker_thread.wait(5000) # Wait for worker thread to finish, with timeout
            if self.worker_thread.isRunning():
                print("[WARNING] WorkerThread did not terminate gracefully. Forcing termination.")
                self.worker_thread.terminate()
                self.worker_thread.wait()

        self.back_to_paradigm_signal.emit() # Emit signal to MainWindow to handle navigation
        event.accept()

    def populate_streams(self):
        self.stream_selector.clear() # Clear existing items
        try:
            streams = resolve_streams(timeout=3.0) # Reduce timeout slightly for faster UI response
            if streams:
                for stream in streams:
                    self.stream_selector.addItem(stream.name)
                # Select the stream if its name was passed via parameters or if it was previously selected
                if self.lsl_stream_name and self.lsl_stream_name in [s.name for s in streams]:
                    self.stream_selector.setCurrentText(self.lsl_stream_name)
                elif streams: # If lsl_stream_name isn't found, but streams exist, select the first one
                    self.stream_selector.setCurrentIndex(0)
                    self.lsl_stream_name = streams[0].name # Update internal state to match selection
                self.statusBar().showMessage(f"Found {len(streams)} LSL stream(s).")
            else:
                self.statusBar().showMessage("No LSL streams found. Ensure a stream is running.")
                self.lsl_stream_name = None # Important: Reset if no stream found
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to resolve LSL streams: {e}")
            self.statusBar().showMessage(f"Error resolving streams: {e}")
            self.lsl_stream_name = None

    def open_next_window(self):
        try:
            print("Launching Game Mode...")
            # It's good practice to stop the current stream before launching another application
            launch_game_mode(DEFAULT_GAME_PATH)
        except Exception as e:
            QMessageBox.critical(self, "Game Launch Failed", f"Could not launch game: {e}")

    def stop_streaming(self):
        if self.stream_thread:
            self.stream_thread.stop() # Tell the thread to stop data acquisition
            self.stream_thread.wait(5000) # Wait for the thread to finish, with a timeout
            if self.stream_thread.isRunning():
                print("[WARNING] StreamThread did not terminate gracefully.")
                self.stream_thread.terminate() # Force termination if it doesn't stop
                self.stream_thread.wait() # Wait for termination

            self.stream_thread = None

        if self.outlet:
            try:
                # Proper way to close an LSL outlet might involve a dispose or close method
                # MNE-LSL StreamOutlet does not have a explicit close, it relies on garbage collection.
                # Setting to None allows it to be garbage collected and recreated.
                self.outlet = None
                print("LSL Outlet released.")
            except Exception as e:
                print(f"[WARNING] Error releasing LSL outlet: {e}")

        # Clear plots and reset UI on stop
        self.plot_widget.clear()
        self.modality_values.clear()
        self.plot_data.clear()
        self.epoch_count = 0
        self.statusBar().showMessage("Streaming stopped.")
        print("Streaming stopped.")

    def emit_back_signal(self):
        """Slot for the 'Back to Paradigm' button."""
        print("RealTimePsdGui: 'Back to Paradigm' button clicked.")
        # `closeEvent` will handle stopping streams and emitting `back_to_paradigm_signal`
        self.close()


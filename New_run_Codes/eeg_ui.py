# eeg_ui.py
# -*- coding: utf-8 -*-
import sys
import numpy as np
import subprocess
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QLabel, QLineEdit, QSpinBox, QComboBox, QPushButton, QMessageBox, QFileDialog,
                             QScrollArea, QGroupBox)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from stream_thread import StreamThread

class FeedbackPlotWidget(FigureCanvas):
    def __init__(self, parent=None, band_name="Alpha"):
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()
        self.band_name = band_name
        self.power_changes = []
        self.epochs = []

    def set_band_name(self, band_name):
        self.band_name = band_name

    def plot_feedback(self, power_change, epoch_count):
        self.power_changes.append(power_change)
        self.epochs.append(epoch_count)
        if len(self.epochs) > 100:
            self.epochs = self.epochs[-100:]
            self.power_changes = self.power_changes[-100:]
        
        self.ax.clear()
        self.ax.plot(self.epochs, self.power_changes, 'b-')
        self.ax.set_title(f"Power change (AUC) of {self.band_name}")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Power Change (%)")
        self.ax.grid(True)
        self.draw()

class EEGParameterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Neurofeedback Parameter Input")
        
        # Set window size to 80% of screen size, capped at 1280x720
        screen = QApplication.primaryScreen().availableGeometry()
        max_width, max_height = min(screen.width() * 0.8, 1280), min(screen.height() * 0.8, 720)
        self.setGeometry(100, 100, int(max_width), int(max_height))

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # Scrollable input form
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        form_widget = QWidget()
        scroll_area.setWidget(form_widget)
        form_layout = QVBoxLayout()
        form_widget.setLayout(form_layout)

        # Styling for form
        form_widget.setStyleSheet("""
            QWidget { font-size: 14px; }
            QLineEdit, QComboBox { 
                padding: 2px; 
                margin: 2px; 
                min-width: 200px; 
            }
            QSpinBox { 
                padding: 2px; 
                margin: 2px; 
                min-width: 230px; /* Slightly wider to account for buttons */
            }
            QSpinBox::up-button, QSpinBox::down-button { 
                width: 25px; /* Ensure buttons are reasonably sized */
            }
            QPushButton { padding: 8px; margin: 5px; border-radius: 4px; }
            QGroupBox { margin-top: 10px; border: 1px solid #ccc; padding: 10px; }
            QLabel { font-weight: bold; }
        """)

        # Stream Settings Group
        stream_group = QGroupBox("Stream Settings")
        stream_form = QFormLayout()
        # Input stream name
        self.input_stream_input = QLineEdit()
        self.input_stream_input.setPlaceholderText("e.g., Signal_generator")
        self.input_stream_input.setToolTip("Name of the input EEG data stream")
        stream_form.addRow(QLabel("Input Stream Name:"), self.input_stream_input)
        # Baseline file path
        self.baseline_file_input = QLineEdit()
        self.baseline_file_input.setPlaceholderText("e.g., path_to_baseline.fif")
        self.baseline_file_button = QPushButton("Browse...")
        self.baseline_file_button.clicked.connect(self.browse_baseline_file)
        baseline_layout = QHBoxLayout()
        baseline_layout.addWidget(self.baseline_file_input)
        baseline_layout.addWidget(self.baseline_file_button)
        stream_form.addRow(QLabel("Baseline File (.fif):"), baseline_layout)
        stream_group.setLayout(stream_form)
        form_layout.addWidget(stream_group)

        # Signal Processing Group
        signal_group = QGroupBox("Signal Processing Parameters")
        signal_form = QFormLayout()
        # Time step
        self.step_input = QLineEdit()
        self.step_input.setPlaceholderText("e.g., 0.01")
        self.step_input.setToolTip("Time step for data processing (seconds)")
        signal_form.addRow(QLabel("Time Step (s):"), self.step_input)
        # Time window
        self.time_window_input = QLineEdit()
        self.time_window_input.setPlaceholderText("e.g., 1.0")
        self.time_window_input.setToolTip("Duration of each epoch (seconds)")
        signal_form.addRow(QLabel("Time Window (s):"), self.time_window_input)
        # Notch frequency
        self.notch_freq_input = QLineEdit()
        self.notch_freq_input.setPlaceholderText("e.g., 50")
        self.notch_freq_input.setToolTip("Frequency for notch filter to remove power line noise (Hz)")
        signal_form.addRow(QLabel("Notch Frequency (Hz):"), self.notch_freq_input)
        signal_group.setLayout(signal_form)
        form_layout.addWidget(signal_group)

        # Frequency Band Group
        freq_group = QGroupBox("Frequency Band Settings")
        freq_form = QFormLayout()
        # Frequency bounds
        self.low_freq_input = QLineEdit()
        self.low_freq_input.setPlaceholderText("e.g., 8")
        self.low_freq_input.setToolTip("Lower frequency bound for the band (Hz)")
        freq_form.addRow(QLabel("Lower Frequency (Hz):"), self.low_freq_input)
        self.high_freq_input = QLineEdit()
        self.high_freq_input.setPlaceholderText("e.g., 12")
        self.high_freq_input.setToolTip("Upper frequency bound for the band (Hz)")
        freq_form.addRow(QLabel("Upper Frequency (Hz):"), self.high_freq_input)
        # Band name
        self.band_name_input = QLineEdit()
        self.band_name_input.setPlaceholderText("e.g., Alpha")
        self.band_name_input.setToolTip("Name of the frequency band (e.g., Alpha, Beta)")
        freq_form.addRow(QLabel("Band Name:"), self.band_name_input)
        freq_group.setLayout(freq_form)
        form_layout.addWidget(freq_group)

        # Channel Configuration Group
        channel_group = QGroupBox("Channel Configuration")
        channel_form = QFormLayout()
        # Number of channels
        self.n_channels_input = QSpinBox()
        self.n_channels_input.setMinimum(1)
        self.n_channels_input.setValue(2)
        self.n_channels_input.setToolTip("Number of EEG channels")
        channel_form.addRow(QLabel("Number of Channels:"), self.n_channels_input)
        # Channel names
        self.feed_ch_names_input = QLineEdit()
        self.feed_ch_names_input.setPlaceholderText("e.g., F3,O1,Pz")
        self.feed_ch_names_input.setToolTip("Comma-separated list of channel names")
        channel_form.addRow(QLabel("Channel Names:"), self.feed_ch_names_input)
        # Channel selection
        self.channel_select = QComboBox()
        self.channel_select.setEnabled(False)
        self.channel_select.setToolTip("Select a channel for analysis")
        channel_form.addRow(QLabel("Select Channel:"), self.channel_select)
        channel_group.setLayout(channel_form)
        form_layout.addWidget(channel_group)

        # Control Buttons
        button_group = QGroupBox("Controls")
        button_layout = QVBoxLayout()
        # Submit button
        self.submit_button = QPushButton("Submit Parameters")
        self.submit_button.clicked.connect(self.submit_parameters)
        self.submit_button.setStyleSheet("""
            QPushButton { background-color: #2196F3; color: white; font-weight: bold; }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
        """)
        button_layout.addWidget(self.submit_button)
        # Start button
        self.start_button = QPushButton("Start Streaming")
        self.start_button.clicked.connect(self.start_streaming)
        self.start_button.setEnabled(False)
        self.start_button.setStyleSheet("""
            QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
        """)
        button_layout.addWidget(self.start_button)
        # Stop button
        self.stop_button = QPushButton("Stop Streaming")
        self.stop_button.clicked.connect(self.stop_streaming)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            QPushButton { background-color: #F44336; color: white; font-weight: bold; }
            QPushButton:hover { background-color: #d32f2f; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
        """)
        button_layout.addWidget(self.stop_button)
        # Unity game launch button
        self.launch_game_button = QPushButton("Launch Unity Game")
        self.launch_game_button.clicked.connect(self.launch_unity_game)
        self.launch_game_button.setStyleSheet("""
            QPushButton { background-color: #FF9800; color: white; font-weight: bold; }
            QPushButton:hover { background-color: #F57C00; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
        """)
        button_layout.addWidget(self.launch_game_button)
        button_group.setLayout(button_layout)
        form_layout.addWidget(button_group)

        # Feedback label
        self.feedback_label = QLabel("Power Change: Waiting for data...")
        self.feedback_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 10px;")
        form_layout.addWidget(self.feedback_label)
        form_layout.addStretch()

        # Plot widget
        self.plot_widget = FeedbackPlotWidget(self)
        main_layout.addWidget(scroll_area, stretch=1)
        main_layout.addWidget(self.plot_widget, stretch=1)  # Equal stretch factor

        # Store validated parameters
        self.validated_params = None
        self.stream_thread = None
        self.unity_process = None

        # Hardcoded Unity game path
        self.unity_game_path = r"C:\Users\varsh\BuildGames\EEGCubeGame.exe"

    def browse_baseline_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Baseline .fif File", "", "FIF Files (*.fif)")
        if file_path:
            self.baseline_file_input.setText(file_path)

    def launch_unity_game(self):
        unity_path = self.unity_game_path
        if not os.path.exists(unity_path):
            QMessageBox.critical(self, "Game Not Found", 
                               f"The Unity game executable was not found at:\n{unity_path}\n\nPlease ensure the game is installed in the correct location.")
            return
        
        if not os.access(unity_path, os.X_OK):
            QMessageBox.critical(self, "Permission Error", 
                               f"The Unity game file is not executable:\n{unity_path}")
            return
        
        if self.unity_process and self.unity_process.poll() is None:
            reply = QMessageBox.question(self, "Game Already Running", 
                                       "The Unity game is already running. Do you want to restart it?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    self.unity_process.terminate()
                    self.unity_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.unity_process.kill()
                    self.unity_process.wait()
                except Exception as e:
                    QMessageBox.warning(self, "Warning", f"Error closing previous game: {e}")
            else:
                return
        
        try:
            self.unity_process = subprocess.Popen([unity_path], 
                                                cwd=os.path.dirname(unity_path))
            QMessageBox.information(self, "Game Launched", 
                                  f"EEG Feedback Game launched successfully!\nPID: {self.unity_process.pid}\n\nNote: The game will close automatically when this application is closed.")
        except subprocess.SubprocessError as e:
            QMessageBox.critical(self, "Launch Error", 
                               f"Failed to launch Unity game:\n{str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Unexpected Error", 
                               f"An unexpected error occurred:\n{str(e)}")

    def validate_parameters(self, step, time_window, n_channels, feed_ch_names,
                           selected_channel, low_freq, high_freq, band_name, input_stream_name, baseline_file, notch_freq):
        try:
            step = float(step)
            if step <= 0:
                raise ValueError("Time step must be positive.")

            time_window = float(time_window)
            if time_window <= 0:
                raise ValueError("Time window must be positive.")

            n_channels = int(n_channels)
            if n_channels <= 0:
                raise ValueError("Number of channels must be positive.")

            feed_ch_names = [name.strip() for name in feed_ch_names.split(",") if name.strip()]
            if not feed_ch_names:
                raise ValueError("At least one channel name must be provided.")

            if selected_channel not in feed_ch_names:
                raise ValueError(f"Selected channel {selected_channel} not in provided channels.")

            low_freq = float(low_freq)
            if low_freq < 0:
                raise ValueError("Lower frequency must be non-negative.")

            high_freq = float(high_freq)
            if high_freq <= low_freq:
                raise ValueError("Upper frequency must be greater than lower frequency.")

            if not band_name.strip():
                raise ValueError("Frequency band name cannot be empty.")

            if not input_stream_name.strip():
                raise ValueError("Input stream name cannot be empty.")

            if not baseline_file or not os.path.exists(baseline_file):
                raise ValueError("Baseline file must be provided and exist.")

            notch_freq = float(notch_freq)
            if notch_freq <= 0:
                raise ValueError("Notch frequency must be positive.")

            return {
                'input_stream_name': input_stream_name,
                'baseline_file': baseline_file,
                'band_name': band_name,
                'low_freq': low_freq,
                'high_freq': high_freq,
                'epoch_duration': time_window,
                'selected_channel': selected_channel,
                'feed_ch_names': feed_ch_names,
                'montage_name': 'standard_1020',
                'notch_freq': notch_freq,
                'prob_threshold': 0.9,
                'max_chunk_size': 1.0
            }

        except ValueError as e:
            return str(e)

    def submit_parameters(self):
        input_stream_name = self.input_stream_input.text()
        baseline_file = self.baseline_file_input.text()
        step = self.step_input.text()
        time_window = self.time_window_input.text()
        n_channels = self.n_channels_input.value()
        feed_ch_names = self.feed_ch_names_input.text()
        selected_channel = self.channel_select.currentText() if self.channel_select.count() > 0 else ""
        low_freq = self.low_freq_input.text()
        high_freq = self.high_freq_input.text()
        band_name = self.band_name_input.text()
        notch_freq = self.notch_freq_input.text()

        self.channel_select.clear()
        channel_list = [name.strip() for name in feed_ch_names.split(",") if name.strip()]
        self.channel_select.addItems(channel_list)
        self.channel_select.setEnabled(bool(channel_list))
        if channel_list:
            selected_channel = channel_list[0]

        result = self.validate_parameters(step, time_window, n_channels, feed_ch_names,
                                        selected_channel, low_freq, high_freq, band_name,
                                        input_stream_name, baseline_file, notch_freq)

        if isinstance(result, str):
            QMessageBox.critical(self, "Input Error", f"Invalid input: {result}")
            self.start_button.setEnabled(False)
        else:
            self.validated_params = result
            self.plot_widget.set_band_name(band_name)
            self.start_button.setEnabled(True)
            QMessageBox.information(self, "Success", "Parameters validated. Ready to start streaming.")

    def start_streaming(self):
        if self.validated_params is None:
            QMessageBox.critical(self, "Error", "No validated parameters. Please submit parameters first.")
            return

        self.submit_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.plot_widget.plot_feedback(0, -1)

        self.stream_thread = StreamThread(self.validated_params)
        self.stream_thread.error_signal.connect(self.handle_stream_error)
        self.stream_thread.power_update_signal.connect(self.update_feedback)
        self.stream_thread.finished.connect(self.streaming_finished)
        self.stream_thread.start()

    def stop_streaming(self):
        if self.stream_thread and self.stream_thread.isRunning():
            self.stream_thread.stop()
            self.stream_thread.wait()
            self.streaming_finished()

    def streaming_finished(self):
        self.submit_button.setEnabled(True)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.feedback_label.setText("Power Change: Waiting for data...")

    def handle_stream_error(self, error_msg):
        QMessageBox.critical(self, "Streaming Error", f"Error: {error_msg}")
        self.streaming_finished()

    def update_feedback(self, power_change, epoch_count):
        self.feedback_label.setText(f"Epoch {epoch_count}: Power Change = {power_change:.2f}%")
        self.plot_widget.plot_feedback(power_change, epoch_count)

    def closeEvent(self, event):
        if self.stream_thread and self.stream_thread.isRunning():
            self.stream_thread.stop()
            self.stream_thread.wait()
        
        if self.unity_process and self.unity_process.poll() is None:
            try:
                self.unity_process.terminate()
                self.unity_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.unity_process.kill()
                self.unity_process.wait()
            except Exception as e:
                print(f"Error terminating Unity game process: {e}")
        
        if event is not None:
            event.accept()

def main():
    app = QApplication(sys.argv)
    window = EEGParameterWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
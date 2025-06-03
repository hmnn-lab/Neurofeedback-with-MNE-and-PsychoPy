import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QSpinBox, QPushButton, QMessageBox)
from PyQt6.QtCore import Qt

# Your existing function (included for reference, but can be imported if in another file)
def get_eeg_parameters(step, time_window, n_channels, feed_ch_names, low_freq, high_freq, band_name):
    """
    Validate EEG processing parameters and return them if valid.

    Returns:
    --------
    tuple or None
        Returns the validated parameters as a tuple or None if validation fails.
    """
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

        low_freq = float(low_freq)
        if low_freq < 0:
            raise ValueError("Lower frequency must be non-negative.")

        high_freq = float(high_freq)
        if high_freq <= low_freq:
            raise ValueError("Upper frequency must be greater than lower frequency.")

        if not band_name.strip():
            raise ValueError("Frequency band name cannot be empty.")

        return step, time_window, n_channels, feed_ch_names, low_freq, high_freq, band_name

    except ValueError as e:
        print(f"Error: {e}")
        return None

class EEGParameterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Neurofeedback Parameter Input")
        self.setGeometry(100, 100, 400, 400)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        # Input fields
        # Time step
        self.step_input = QLineEdit()
        self.step_input.setPlaceholderText("e.g., 0.01")
        layout.addWidget(QLabel("Time step (seconds):"))
        layout.addWidget(self.step_input)

        # Time window
        self.time_window_input = QLineEdit()
        self.time_window_input.setPlaceholderText("e.g., 5")
        layout.addWidget(QLabel("Time window (seconds):"))
        layout.addWidget(self.time_window_input)

        # Number of channels
        self.n_channels_input = QSpinBox()
        self.n_channels_input.setMinimum(1)
        self.n_channels_input.setValue(2)
        layout.addWidget(QLabel("Number of channels:"))
        layout.addWidget(self.n_channels_input)

        # Channel names
        self.feed_ch_names_input = QLineEdit()
        self.feed_ch_names_input.setPlaceholderText("e.g., O1,Pz")
        layout.addWidget(QLabel("Channel names (comma-separated):"))
        layout.addWidget(self.feed_ch_names_input)

        # Low frequency
        self.low_freq_input = QLineEdit()
        self.low_freq_input.setPlaceholderText("e.g., 8")
        layout.addWidget(QLabel("Lower frequency bound (Hz):"))
        layout.addWidget(self.low_freq_input)

        # High frequency
        self.high_freq_input = QLineEdit()
        self.high_freq_input.setPlaceholderText("e.g., 12")
        layout.addWidget(QLabel("Upper frequency bound (Hz):"))
        layout.addWidget(self.high_freq_input)

        # Frequency band name
        self.band_name_input = QLineEdit()
        self.band_name_input.setPlaceholderText("e.g., Alpha")
        layout.addWidget(QLabel("Frequency band name:"))
        layout.addWidget(self.band_name_input)

        # Submit button
        submit_button = QPushButton("Submit Parameters")
        submit_button.clicked.connect(self.submit_parameters)
        layout.addWidget(submit_button)

        # Add stretch to push content up
        layout.addStretch()

    def submit_parameters(self):
        # Collect inputs
        step = self.step_input.text()
        time_window = self.time_window_input.text()
        n_channels = self.n_channels_input.value()
        feed_ch_names = self.feed_ch_names_input.text()
        low_freq = self.low_freq_input.text()
        high_freq = self.high_freq_input.text()
        band_name = self.band_name_input.text()

        # Call the existing function with UI inputs
        result = get_eeg_parameters(step, time_window, n_channels, feed_ch_names,
                                  low_freq, high_freq, band_name)

        if result is None:
            # Show error message
            QMessageBox.critical(self, "Input Error",
                               "Invalid input. Please check all fields and try again.")
        else:
            # Unpack validated parameters
            step, time_window, n_channels, feed_ch_names, low_freq, high_freq, band_name = result
            # Here, pass the parameters to your neurofeedback processing pipeline
            # For now, show a confirmation (replace with your pipeline call)
            QMessageBox.information(self, "Success",
                                  f"Parameters accepted:\n"
                                  f"Step: {step}\n"
                                  f"Time window: {time_window}\n"
                                  f"Channels: {n_channels}\n"
                                  f"Channel names: {', '.join(feed_ch_names)}\n"
                                  f"Low freq: {low_freq}\n"
                                  f"High freq: {high_freq}\n"
                                  f"Band name: {band_name}")
            # Example: Call your processing function here
            # your_processing_function(step, time_window, n_channels, feed_ch_names,
            #                         low_freq, high_freq, band_name)

def main():
    app = QApplication(sys.argv)
    window = EEGParameterWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
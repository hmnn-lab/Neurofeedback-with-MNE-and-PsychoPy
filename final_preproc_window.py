import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QProgressBar,
    QMessageBox, QDialogButtonBox, QApplication, QPushButton, QHBoxLayout
)
from PyQt6.QtCore import pyqtSignal, Qt
import mne
# Assuming these imports point to your actual preprocessing and calculation functions
# It's good practice to ensure these are specific file paths/modules if not globally accessible
from preproc_initialize import preproc_flow
from cal_baseline_parameter import compute_baseline_modality
import pandas as pd

class PreprocessingDialog(QDialog):
    """
    A QDialog for EEG preprocessing and baseline parameter calculation.
    It displays progress and, upon successful completion, allows launching
    real-time feedback.
    """
    # Signal to emit results for RealTimePsdGui or other consuming components
    preprocessing_completed = pyqtSignal(dict)

    # Define EEG frequency bands as a class-level constant
    # This makes it accessible without instantiating the class, and emphasizes it's fixed.
    EEG_FREQ_BANDS = {
        'Delta': [1, 4],
        'Theta': [4, 8],
        'Alpha': [8, 12],
        'Beta': [12, 30],
        'Gamma': [30, 100]
    }

    def __init__(self, baseline_file_path, num_channels, channel_names, modality,
                 selected_channels=None, selected_channels_2=None, frequency_band=None,
                 band_1=None, band_2=None, phase_frequency=None, amplitude_frequency=None,
                 user_data_dir=None, parent=None):
        """
        Initializes the PreprocessingDialog with all necessary parameters
        for EEG data processing and modality calculation.

        :param baseline_file_path: Path to the MNE raw file for baseline.
        :param num_channels: Total number of channels in the EEG data.
        :param channel_names: List of all channel names.
        :param modality: The type of EEG metric to compute (e.g., "psd_auc", "pac").
        :param selected_channels: Primary channel(s) selected for the modality.
        :param selected_channels_2: Secondary channel(s) for modalities like coherence.
        :param frequency_band: Single frequency band name (e.g., 'Alpha') for PSD AUC.
        :param band_1: Dictionary for the first frequency band (e.g., for PSD Ratio).
        :param band_2: Dictionary for the second frequency band (e.g., for PSD Ratio).
        :param phase_frequency: Frequency for phase in PAC.
        :param amplitude_frequency: Frequency for amplitude in PAC.
        :param user_data_dir: Directory to save processed data/plots.
        :param parent: Parent QWidget for the dialog.
        """
        super().__init__(parent)
        self.setWindowTitle("Preprocessing & Baseline Calculation")
        self.setMinimumSize(600, 350) # Set a sensible default size

        # Store all these parameters as instance attributes for access during processing
        self.baseline_file_path = baseline_file_path
        self.num_channels = num_channels
        self.channel_names = channel_names
        self.modality = modality
        # Ensure these are always lists, even if None or [] passed initially
        self.selected_channels = selected_channels if isinstance(selected_channels, list) else []
        self.selected_channels_2 = selected_channels_2 if isinstance(selected_channels_2, list) else []

        print(f"\n--- Debugging PreprocessingDialog Init Channels ---")
        print(f"Modality: {self.modality}")
        print(f"self.selected_channels (in init): {self.selected_channels}")
        print(f"self.selected_channels_2 (in init): {self.selected_channels_2}")
        print(f"--- End PreprocessingDialog Init Channels ---")

        self.frequency_band = frequency_band # e.g., ['Alpha'] for PSD AUC/Coherence
        self.band_1 = band_1 # e.g., {'name': 'Alpha', 'low': 8, 'high': 12} for PSD Ratio
        self.band_2 = band_2 # e.g., {'name': 'Beta', 'low': 12, 'high': 30} for PSD Ratio
        self.phase_frequency = phase_frequency
        self.amplitude_frequency = amplitude_frequency
        self.user_data_dir = user_data_dir

        self.freq_bands_for_calculation = [] # Will store [[low, high]] pairs
        self.freq_band_names_display = []    # Will store ['Alpha', 'Beta'] for UI display

        self._process_frequency_parameters() # Helper method to clean up __init__

        self._preprocessed_params = {} # Stores results after successful processing
        self._init_ui()

    def _process_frequency_parameters(self):
        """
        Parses and prepares frequency band parameters for internal use
        and UI display based on the selected modality.
        """
        print(f"\n--- Debugging _process_frequency_parameters ---")
        print(f"Modality: {self.modality}")
        print(f"Raw self.frequency_band: {self.frequency_band} (Type: {type(self.frequency_band)})")
        print(f"Raw self.band_1: {self.band_1} (Type: {type(self.band_1)})")
        print(f"Raw self.band_2: {self.band_2} (Type: {type(self.band_2)})")
        print(f"Raw self.phase_frequency: {self.phase_frequency} (Type: {type(self.phase_frequency)})")
        print(f"Raw self.amplitude_frequency: {self.amplitude_frequency} (Type: {type(self.amplitude_frequency)})")

        self.freq_bands_for_calculation = []
        self.freq_band_names_display = []

        try:
            if self.modality == "psd_ratio":
                if not (self.band_1 and self.band_2):
                    raise ValueError("`band_1` and `band_2` must be provided for 'psd_ratio'.")
                if not all(isinstance(d, dict) and 'low' in d and 'high' in d for d in [self.band_1, self.band_2]):
                    raise ValueError("`band_1` and `band_2` must be dictionaries with 'low' and 'high' keys.")
                self.freq_bands_for_calculation = [
                    [float(self.band_1['low']), float(self.band_1['high'])],
                    [float(self.band_2['low']), float(self.band_2['high'])]
                ]
                self.freq_band_names_display = [
                    self.band_1.get('name', f"{self.band_1['low']}-{self.band_1['high']}Hz"),
                    self.band_2.get('name', f"{self.band_2['low']}-{self.band_2['high']}Hz")
                ]
                print(f"DEBUG: Configured for psd_ratio. Bands: {self.freq_bands_for_calculation}")

            elif self.modality in ["psd_auc", "coh"]:
                if not self.frequency_band:
                    raise ValueError(f"`frequency_band` must be provided for '{self.modality}'.")
                if not (isinstance(self.frequency_band, dict) and 'low' in self.frequency_band and 'high' in self.frequency_band):
                    raise ValueError(f"`frequency_band` must be a dictionary with 'low' and 'high' keys for '{self.modality}'.")
                low_freq = float(self.frequency_band['low'])
                high_freq = float(self.frequency_band['high'])
                band_name = self.frequency_band.get('name', f"{low_freq}-{high_freq}Hz")
                self.freq_bands_for_calculation = [[low_freq, high_freq]]
                self.freq_band_names_display = [band_name]
                print(f"DEBUG: Configured for {self.modality}. Band: {self.freq_bands_for_calculation}")

            elif self.modality == "pac":
                if not (self.phase_frequency and self.amplitude_frequency):
                    raise ValueError("`phase_frequency` and `amplitude_frequency` must be provided for 'pac'.")
                # Handle both dictionary and list/tuple formats
                phase_low, phase_high = self._parse_frequency(self.phase_frequency, "phase_frequency")
                amp_low, amp_high = self._parse_frequency(self.amplitude_frequency, "amplitude_frequency")
                self.freq_bands_for_calculation = [
                    [float(phase_low), float(phase_high)],
                    [float(amp_low), float(amp_high)]
                ]
                phase_name = (self.phase_frequency.get('name', f"Phase: {phase_low}-{phase_high}Hz")
                            if isinstance(self.phase_frequency, dict) else f"Phase: {phase_low}-{phase_high}Hz")
                amp_name = (self.amplitude_frequency.get('name', f"Amplitude: {amp_low}-{amp_high}Hz")
                            if isinstance(self.amplitude_frequency, dict) else f"Amplitude: {amp_low}-{amp_high}Hz")
                self.freq_band_names_display = [phase_name, amp_name]
                print(f"DEBUG: Configured for pac. Bands: {self.freq_bands_for_calculation}")

            else:
                raise ValueError(f"Unsupported modality: {self.modality}")

            # Validate frequency bands
            if not self.freq_bands_for_calculation:
                raise ValueError("No frequency bands configured for calculation.")
            for band in self.freq_bands_for_calculation:
                if not (isinstance(band, list) and len(band) == 2 and all(isinstance(x, (int, float)) for x in band)):
                    raise ValueError(f"Invalid frequency band format: {band}. Expected [low, high] with numeric values.")
                if band[0] >= band[1]:
                    raise ValueError(f"Invalid frequency band {band}: low ({band[0]}) must be less than high ({band[1]}).")

        except ValueError as e:
            print(f"DEBUG: Error in _process_frequency_parameters: {e}")
            self.freq_bands_for_calculation = []
            self.freq_band_names_display = ["N/A or Invalid"]
            raise

        print(f"Final self.freq_bands_for_calculation: {self.freq_bands_for_calculation}")
        print(f"Final self.freq_band_names_display: {self.freq_band_names_display}")
        print(f"--- End _process_frequency_parameters ---")

    def _parse_frequency(self, freq_param, param_name):
        """
        Helper method to parse frequency parameters (dict or list/tuple) into low, high values.
        """
        if isinstance(freq_param, dict):
            if 'low' not in freq_param or 'high' not in freq_param:
                raise ValueError(f"{param_name} dictionary must contain 'low' and 'high' keys.")
            return float(freq_param['low']), float(freq_param['high'])
        elif isinstance(freq_param, (list, tuple)) and len(freq_param) == 2:
            return float(freq_param[0]), float(freq_param[1])
        else:
            raise ValueError(f"Invalid {param_name} format: {freq_param}. Expected dict with 'low' and 'high' or list/tuple with two numeric values.")

    def _init_ui(self):
        """Initializes the user interface elements of the dialog."""
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel(f"EEG Preprocessing & {self.modality.upper()} Calculation")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #333; margin-bottom: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        file_name = os.path.basename(self.baseline_file_path) if self.baseline_file_path else "No file selected"
        self.file_label = QLabel(
            f"Selected file: <b style='color: #0056b3;'>{file_name}</b>"
            if self.baseline_file_path else "<b style='color: #e60000;'>No file selected</b>"
        )
        self.file_label.setStyleSheet("font-size: 14px; margin-bottom: 15px;")
        layout.addWidget(self.file_label)

        channels_info = f"Number of channels: {self.num_channels if self.num_channels else 'Not specified'}"
        self.channels_label = QLabel(channels_info)
        self.channels_label.setStyleSheet("font-size: 13px; margin-bottom: 10px;")
        layout.addWidget(self.channels_label)

        channel_names_text = ", ".join(self.channel_names) if self.channel_names else "Not specified"
        self.channel_names_label = QLabel(f"Channel names: {channel_names_text}")
        self.channel_names_label.setStyleSheet("font-size: 13px; margin-bottom: 10px;")
        layout.addWidget(self.channel_names_label)

        # Use freq_band_names_display for UI display
        self.status_label = QLabel(
            f"Modality: {self.modality}, Freq. Bands: {', '.join(self.freq_band_names_display)}, "
            f"Channels: {self.selected_channels or self.selected_channels_2 or 'N/A'}"
        )
        self.status_label.setStyleSheet("color: blue; margin-top: 10px; font-weight: bold;")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Progress: %p%")
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #c0c0c0; border-radius: 5px; text-align: center; background-color: #f0f0f0; }"
            "QProgressBar::chunk { background-color: #4CAF50; border-radius: 4px; }"
        )
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        layout.addSpacing(15)

        # --- Add "Real-time feedback" button ---
        self.button_realtime_feedback = QPushButton("Launch Real-time Feedback")
        self.button_realtime_feedback.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; padding: 8px 15px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #1e88e5; }"
            "QPushButton:disabled { background-color: #cccccc; }"
        )
        self.button_realtime_feedback.setEnabled(False) # Initially disabled
        self.button_realtime_feedback.hide() # Initially hidden
        self.button_realtime_feedback.clicked.connect(self._open_realtime_feedback)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self._handle_ok_button) # Connect to custom handler for run logic
        self.button_box.rejected.connect(self.reject)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText(f"Run Preprocessing + {self.modality.upper()}")
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; padding: 8px 15px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #cccccc; }"
        )
        self.button_box.button(QDialogButtonBox.StandardButton.Cancel).setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; padding: 8px 15px; border-radius: 5px; }"
            "QPushButton:hover { background-color: #da190b; }"
        )

        # Create a horizontal layout for the buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.button_realtime_feedback)
        button_layout.addStretch() # Pushes the feedback button to the left
        button_layout.addWidget(self.button_box)

        layout.addStretch(1) # Pushes content to the top
        layout.addLayout(button_layout) # Add the new button layout

        self.setLayout(layout)

    def _open_realtime_feedback(self):
        """
        Emits the preprocessing_completed signal with the processed results
        to trigger the opening of the RealTimePsdGui window in the parent.
        Then closes this dialog.
        """
        print("[DEBUG PreprocessingDialog] _open_realtime_feedback called.")
        if self._preprocessed_params: # This check ensures results are ready
            print("[DEBUG PreprocessingDialog] _preprocessed_params is populated. Emitting signal...")
            # Emit the signal with the stored results dictionary
            self.preprocessing_completed.emit(self._preprocessed_params)
            print("[DEBUG PreprocessingDialog] Signal emitted. Attempting to close dialog with self.accept()...")
            self.accept() # This will close the dialog and return QDialog.Accepted
            print("[DEBUG PreprocessingDialog] self.accept() call completed.")
        else:
            print("[DEBUG PreprocessingDialog] _preprocessed_params is empty. Cannot proceed with feedback.")
            QMessageBox.warning(self, "Warning", "Preprocessing results not available. Please run preprocessing first.")

    def accept(self):
        """
        Overrides the standard QDialog accept method.
        This is called when the dialog's result should be QDialog.Accepted.
        """
        print("[DEBUG PreprocessingDialog] Custom accept() method called.")
        super().accept() # Call the base class accept to close the dialog
        print("[DEBUG PreprocessingDialog] super().accept() completed.")

    def _handle_ok_button(self):
        """
        Handles the logic when the 'Run Preprocessing' (or 'Close') button is clicked.
        Initiates the preprocessing and baseline calculation.
        """
        print("[DEBUG PreprocessingDialog] _handle_ok_button called (Run Preprocessing/Close).")
        self.status_label.clear()
        self.progress_bar.setValue(0)

        # Disable buttons to prevent re-running or closing during processing
        self.button_box.setEnabled(False)
        self.button_realtime_feedback.setEnabled(False)
        self.button_realtime_feedback.hide()  # Ensure it's hidden before processing starts

        # If the button text is "Close", it means processing is already done,
        # so just accept the dialog.
        if self.button_box.button(QDialogButtonBox.StandardButton.Ok).text() == "Close":
            print("[DEBUG PreprocessingDialog] 'Close' button clicked. Accepting dialog.")
            self.accept()
            return

        if not self.baseline_file_path or not os.path.isfile(self.baseline_file_path):
            self.status_label.setText("Error: No valid baseline file provided.")
            self.status_label.setStyleSheet("color: red; margin-top: 10px; font-weight: bold;")
            self.button_box.setEnabled(True)  # Re-enable if there's an immediate error
            print("[DEBUG PreprocessingDialog] Error: No valid file. Re-enabling buttons.")
            return

        try:
            self.status_label.setText("Processing: Running EEG preprocessing flow...")
            self.status_label.setStyleSheet("color: blue; margin-top: 10px; font-weight: bold;")
            self.progress_bar.setFormat("Preprocessing: %p%")
            self.progress_bar.setValue(10)
            QApplication.processEvents()  # Update UI immediately

            # --- Run preprocessing ---
            raw_cleaned, bad_channels, asr, ica, artifact_components = preproc_flow(
                self.baseline_file_path,
                n_channels=self.num_channels,
                channel_names=self.channel_names
            )
            print(f"[DEBUG PreprocessingDialog] Preproc_flow returned: raw_cleaned={'None' if raw_cleaned is None else 'MNE Raw object'}, bad_channels={bad_channels}")

            self.progress_bar.setValue(50)
            QApplication.processEvents()

            self.status_label.setText(f"Processing: Calculating baseline {self.modality.upper()}...")
            self.status_label.setStyleSheet("color: blue; margin-top: 10px; font-weight: bold;")
            self.progress_bar.setFormat(f"{self.modality.upper()} Calculation: %p%")
            self.progress_bar.setValue(60)
            QApplication.processEvents()

            if raw_cleaned is None:
                raise RuntimeError("Preprocessing (preproc_flow) failed, raw_cleaned object is None.")

            # --- Validate freq_bands_for_calculation ---
            if not isinstance(self.freq_bands_for_calculation, list) or not self.freq_bands_for_calculation:
                raise ValueError("`freq_bands_for_calculation` must be a non-empty list of [low, high] ranges.")
            for band in self.freq_bands_for_calculation:
                if not (isinstance(band, (list, tuple)) and len(band) == 2 and all(isinstance(x, (int, float)) for x in band)):
                    raise ValueError(f"Invalid frequency band format: {band}. Expected [low, high] with numeric values.")
                if band[0] >= band[1]:
                    raise ValueError(f"Invalid frequency band {band}: low ({band[0]}) must be less than high ({band[1]}).")
            expected_bands = 2 if self.modality in ['psd_ratio', 'pac'] else 1
            if len(self.freq_bands_for_calculation) != expected_bands:
                raise ValueError(f"Modality '{self.modality}' requires {expected_bands} frequency band(s), but {len(self.freq_bands_for_calculation)} provided.")

            # --- Debug parameters ---
            print(f"\n--- Debugging PreprocessingDialog Call Params ---")
            print(f"Modality being passed: {self.modality}")
            print(f"Channels being passed: {self.selected_channels}")
            print(f"Channels_2 being passed: {self.selected_channels_2}")
            print(f"freq_bands_for_calculation: {self.freq_bands_for_calculation}, type: {type(self.freq_bands_for_calculation)}")
            print(f"--- End PreprocessingDialog Call Params ---")

            # --- Compute baseline modality ---
            df, excel_path, plot_path = compute_baseline_modality(
                raw_or_file=raw_cleaned,
                modality=self.modality,
                channels=self.selected_channels,
                channels_2=self.selected_channels_2,
                freq_bands_numeric=self.freq_bands_for_calculation,
                band_1_dict=self.band_1,
                band_2_dict=self.band_2,
                phase_freq_dict=self.phase_frequency,
                amp_freq_dict=self.amplitude_frequency,
                user_data_dir=self.user_data_dir
            )
            print(f"[DEBUG PreprocessingDialog] compute_baseline_modality returned: df={'None' if df is None else 'DataFrame'}, excel_path={excel_path}, plot_path={plot_path}")

            self.progress_bar.setValue(100)
            QApplication.processEvents()

            if df is None:
                raise RuntimeError(f"{self.modality.upper()} computation failed: Resulting DataFrame is None.")

            # --- Store results in _preprocessed_params ---
            self._preprocessed_params = {
                'raw_cleaned': raw_cleaned,
                'bad_channels': bad_channels,
                'asr': asr,
                'ica': ica,
                'artifact_components': artifact_components,
                'modality_result': df,
                'df_excel': excel_path,
                'plot': plot_path,
                'modality': self.modality,
                'info': raw_cleaned.info
            }
            print(f"[DEBUG PreprocessingDialog] _preprocessed_params populated with keys: {list(self._preprocessed_params.keys())}")
            print(f"[DEBUG PreprocessingDialog] raw_cleaned type: {type(self._preprocessed_params.get('raw_cleaned'))}")
            print(f"[DEBUG PreprocessingDialog] modality_result type: {type(self._preprocessed_params.get('modality_result'))}")

            self.status_label.setText(f"Preprocessing and {self.modality.upper()} completed successfully!")
            self.status_label.setStyleSheet("color: green; margin-top: 10px; font-weight: bold;")

            # --- Enable and show the new button after successful processing ---
            self.button_realtime_feedback.setEnabled(True)
            self.button_realtime_feedback.show()
            self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Close")
            self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
            print("[DEBUG PreprocessingDialog] Preprocessing successful. Real-time feedback button enabled/shown.")

        except Exception as e:
            print(f"[DEBUG PreprocessingDialog] An exception occurred in _handle_ok_button: {e}")
            import traceback
            traceback.print_exc()
            self.button_box.setEnabled(True)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Progress: %p%")
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: red; margin-top: 10px; font-weight: bold;")
            QMessageBox.critical(self, "Processing Error", f"An unexpected error occurred during processing:\n{str(e)}")
            self._preprocessed_params = {}
            print("[DEBUG PreprocessingDialog] Error handled. Buttons re-enabled, params cleared.")

    def get_preprocessed_params(self):
        """
        Returns the dictionary of preprocessed results.
        Note: This method is less critical now that `preprocessing_completed` signal
        is the primary way to get results to the MainWindow.
        """
        return self._preprocessed_params

    def reject(self):
        """
        Overrides the standard QDialog reject method.
        Called when the dialog is cancelled (e.g., via 'Cancel' button).
        """
        print("[DEBUG PreprocessingDialog] reject() called (Cancel).")
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Progress: %p%")
        self.status_label.clear()
        self.button_box.setEnabled(True) # Ensure buttons are re-enabled
        self.button_realtime_feedback.hide() # Hide/disable feedback button on cancel
        self.button_realtime_feedback.setEnabled(False)
        super().reject()
        print("[DEBUG PreprocessingDialog] Dialog rejected.")
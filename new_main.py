import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QStackedWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QWidget, QLabel, QLineEdit, QMessageBox, QDialog, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
import os
import inspect
from datetime import datetime, timezone, timedelta
import traceback

# Import your custom UI windows/dialogs
# Consider adding a comment for each import if the name isn't immediately obvious
from final_baseline_window import BaselineWindow
from paradigm_choice_window import ParadigmGui
from final_preproc_window import PreprocessingDialog # This is your PreprocessingDialog
from feedback_window import RealTimePsdGui

# --- Constants for Timezone (Optional, but good for consistency) ---
# Define IST timezone once
IST = timezone(timedelta(hours=5, minutes=30))

class WelcomePage(QWidget):
    """
    The initial welcome screen for the EEG Real-Time System.
    Collects user's name and ID.
    """
    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        """Initializes the user interface for the welcome page."""
        layout = QVBoxLayout()
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(15)

        title = QLabel("EEG Real-Time System")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(title)

        info = QLabel("An application for real-time EEG processing.")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setStyleSheet("font-size: 14px;")
        layout.addWidget(info)

        form_layout = QHBoxLayout()
        form_layout.setSpacing(20)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter your name")
        self.name_input.setMinimumWidth(200)
        self.name_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        form_layout.addWidget(self.name_input)

        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("Enter your ID")
        self.id_input.setMinimumWidth(200)
        self.id_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        form_layout.addWidget(self.id_input)

        layout.addLayout(form_layout)

        self.start_button = QPushButton("Start")
        self.start_button.setFixedWidth(140)
        self.start_button.setStyleSheet(
            "QPushButton { background-color: #007bff; color: white; padding: 10px 20px; "
            "border-radius: 8px; font-size: 16px; }"
            "QPushButton:hover { background-color: #0056b3; }"
        )
        layout.addWidget(self.start_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(layout)

        self.setMinimumWidth(500)
        self.setMinimumHeight(250)


class MainWindow(QMainWindow):
    """
    The main application window, managing the flow between different
    EEG processing stages using a QStackedWidget.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Processing App")
        self.setGeometry(100, 100, 800, 600) # Give it a larger default size

        # --- Instance Attributes Initialization ---
        # UI components
        self.stacked_widget = QStackedWidget()
        self.back_button = QPushButton("Back")

        # Data storage for preprocessing results and paradigm parameters
        self.paradigm_parameters = None
        self.asr_obj = None
        self.ica_obj = None
        self.artifact_components = []
        self.bad_channels = []
        self.eeg_info = None

        # Flags and window references
        self.realtime_feedback_launched = False
        self.user_data_dir = None # Will be set after welcome page
        self.user_name = None
        self.user_id = None

        # References to child windows/dialogs (initially None)
        self.welcome_page = None
        self.baseline_window = None
        self.paradigm_window = None
        self.preprocessing_dialog = None
        self.psd_gui = None # RealTimePsdGui instance

        self._init_ui() # Initialize UI elements
        self._setup_connections() # Set up signal/slot connections

    def _init_ui(self):
        """Initializes the main window's user interface."""
        # Navigation layout for the back button
        nav_layout = QHBoxLayout()
        self.back_button.setEnabled(False) # Disabled initially
        self.back_button.setFixedWidth(100) # Give it a fixed width
        nav_layout.addWidget(self.back_button)
        nav_layout.addStretch() # Pushes the button to the left

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(nav_layout)
        main_layout.addWidget(self.stacked_widget)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Add initial pages to stacked widget
        self.welcome_page = WelcomePage()
        self.stacked_widget.addWidget(self.welcome_page)

        # Set initial page
        self.stacked_widget.setCurrentWidget(self.welcome_page)
        self._update_back_button_state() # Update back button state on startup

    def _setup_connections(self):
        """Connects signals to slots for navigation and workflow."""
        self.welcome_page.start_button.clicked.connect(self.go_to_baseline)
        self.back_button.clicked.connect(self.go_back)

    def _log_debug(self, message):
        """Helper for consistent debug logging with timestamp."""
        timestamp = datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')
        print(f"[DEBUG {timestamp}] {message}")

    def _update_back_button_state(self):
        """
        Updates the enabled state of the global back button based on the
        current widget in the stacked layout.
        """
        is_welcome_or_realtime = (self.stacked_widget.currentWidget() == self.welcome_page or
                                  self.stacked_widget.currentWidget() == self.psd_gui)
        self.back_button.setEnabled(not is_welcome_or_realtime)

    def go_to_baseline(self):
        """
        Navigates to the BaselineWindow after collecting user details.
        Creates a user-specific directory for recordings.
        """
        self.user_name = self.welcome_page.name_input.text().strip()
        self.user_id = self.welcome_page.id_input.text().strip()

        if not self.user_name or not self.user_id:
            QMessageBox.warning(self, "Input Required", "Please enter both your name and ID before continuing.")
            return

        folder_name = f"{self.user_name}_{self.user_id}".replace(" ", "_")
        base_path = "./recordings"
        self.user_data_dir = os.path.join(base_path, folder_name)
        os.makedirs(self.user_data_dir, exist_ok=True)

        self._log_debug(f"User data directory created: {self.user_data_dir}")

        # Create or recreate BaselineWindow to ensure a fresh state
        if self.baseline_window:
            self.stacked_widget.removeWidget(self.baseline_window)
            self.baseline_window.deleteLater() # Safely delete the old instance
            self.baseline_window = None

        self.baseline_window = BaselineWindow()
        # Ensure signal is connected only once
        try:
            self.baseline_window.proceed_to_paradigm.disconnect(self.show_paradigm)
        except TypeError:
            pass # No prior connection to disconnect
        self.baseline_window.proceed_to_paradigm.connect(self.show_paradigm)
        self.stacked_widget.addWidget(self.baseline_window) # Add to stacked widget

        self.stacked_widget.setCurrentWidget(self.baseline_window)
        self._update_back_button_state() # Update button state

    @pyqtSlot(str, list, int)
    def show_paradigm(self, baseline_file, channel_names, num_channels):
        """
        Displays the ParadigmGui after baseline recording is complete.
        Connects necessary signals for further flow.
        """
        self._log_debug(f"ParadigmGui module: {inspect.getfile(ParadigmGui)}")
        self._log_debug(f"Received proceed_to_paradigm with file: {baseline_file}, channels: {channel_names}, num_channels: {num_channels}")

        # Create or recreate ParadigmGui to ensure a fresh state for new parameters
        if self.paradigm_window:
            self.stacked_widget.removeWidget(self.paradigm_window)
            self.paradigm_window.deleteLater() # Safely delete the old instance
            self.paradigm_window = None

        self.paradigm_window = ParadigmGui(baseline_file, channel_names, num_channels, self.user_data_dir, self)
        # Connect the new signal from ParadigmGui
        try:
            self.paradigm_window.launch_preprocessing_requested.disconnect(self.launch_preprocessing_dialog_from_paradigm)
        except TypeError:
            pass # No prior connection to disconnect
        self.paradigm_window.launch_preprocessing_requested.connect(self.launch_preprocessing_dialog_from_paradigm)

        self.stacked_widget.addWidget(self.paradigm_window) # Add to stacked widget

        self.stacked_widget.setCurrentWidget(self.paradigm_window)
        self._update_back_button_state() # Update button state

    @pyqtSlot(dict)
    def launch_preprocessing_dialog_from_paradigm(self, parameters):
        """
        Launches the PreprocessingDialog based on parameters selected in ParadigmGui.
        The dialog is modal, blocking interaction with MainWindow until closed.
        """
        self._log_debug(f"MainWindow received request to launch PreprocessingDialog with parameters: {parameters}")
        self.paradigm_parameters = parameters # Store parameters for RealTimePsdGui

        print(f"\n--- Debugging Parameters from ParadigmGui ---")
        print(f"Modality: {parameters.get('modality')}")
        print(f"Frequency Band (single): {parameters.get('frequency_band')}") # For PSD AUC/Coherence
        print(f"Band 1 (dict): {parameters.get('band_1')}") # For PSD Ratio/PAC
        print(f"Band 2 (dict): {parameters.get('band_2')}") # For PSD Ratio/PAC
        print(f"Phase Frequency (dict): {parameters.get('phase_frequency')}") # For PAC
        print(f"Amplitude Frequency (dict): {parameters.get('amplitude_frequency')}") # For PAC
        # Add channel parameters to this debug print to see what ParadigmGui sends
        print(f"Channel (raw from ParadigmGui): {parameters.get('channel')}")
        print(f"Channel_1 (raw from ParadigmGui): {parameters.get('channel_1')}")
        print(f"Channel_2 (raw from ParadigmGui): {parameters.get('channel_2')}")
        print(f"--- End ParadigmGui Parameters ---")

        # Reset flag before launching preprocessing for a new feedback session
        self.realtime_feedback_launched = False

        try:
            # --- START OF REQUIRED CHANGES FOR CHANNELS ---
            primary_channel_str = parameters.get('channel')
            if not primary_channel_str: # If 'channel' is not present or empty
                primary_channel_str = parameters.get('channel_1') # Try 'channel_1'

            # Ensure selected_channels is always a list, even if empty or None
            selected_channels_list = []
            if primary_channel_str: # If we have a valid channel string
                selected_channels_list = [primary_channel_str]

            # Ensure selected_channels_2 is always a list, even if empty or None
            secondary_channel_str = parameters.get('channel_2')
            selected_channels_2_list = []
            if secondary_channel_str: # If we have a valid second channel string
                selected_channels_2_list = [secondary_channel_str]

            print(f"\n--- Debugging MainWindow Processed Channels ---")
            print(f"Processed selected_channels: {selected_channels_list}")
            print(f"Processed selected_channels_2: {selected_channels_2_list}")
            print(f"--- End MainWindow Processed Channels ---")
            # --- END OF REQUIRED CHANGES FOR CHANNELS ---

            self.preprocessing_dialog = PreprocessingDialog(
                baseline_file_path=parameters["baseline_file"],
                num_channels=parameters["num_channels"],
                channel_names=parameters["channel_names"],
                modality=parameters["modality"], # Pass the backend modality string
                
                # *** THESE ARE THE LINES YOU NEED TO CHANGE ***
                selected_channels=selected_channels_list, # Pass the new list variable
                selected_channels_2=selected_channels_2_list, # Pass the new list variable
                
                frequency_band=parameters.get('frequency_band'),
                band_1=parameters.get('band_1'),
                band_2=parameters.get('band_2'),
                phase_frequency=parameters.get('phase_frequency'),
                amplitude_frequency=parameters.get('amplitude_frequency'),
                user_data_dir=self.user_data_dir,
                parent=self
            )

            # Disconnect previous connections if any, and connect for current dialog
            try:
                # IMPORTANT: Pass the `parameters` dictionary to the slot
                self.preprocessing_dialog.preprocessing_completed.disconnect() # Disconnect all previous
            except TypeError:
                pass # No prior connection to disconnect
            self.preprocessing_dialog.preprocessing_completed.connect(lambda results: self.handle_preprocessing_results(results, parameters))

            self._log_debug("PreprocessingDialog created and signal connected.")

            # Show PreprocessingDialog as a modal dialog
            dialog_result = self.preprocessing_dialog.exec()
            self._log_debug(f"PreprocessingDialog closed with result: {dialog_result}")

            # If dialog was accepted, check if feedback was actually launched from within it
            if dialog_result == QDialog.DialogCode.Accepted and not self.realtime_feedback_launched:
                QMessageBox.information(self, "Preprocessing Complete", "Preprocessing and baseline calculation finished. You can now select a different paradigm or close the application.")
                # Return to paradigm selection if feedback wasn't launched
                self.stacked_widget.setCurrentWidget(self.paradigm_window)
                self._update_back_button_state()
            elif dialog_result == QDialog.DialogCode.Rejected:
                QMessageBox.information(self, "Preprocessing Cancelled", "Preprocessing was cancelled.")
                self.stacked_widget.setCurrentWidget(self.paradigm_window) # Return to paradigm window on cancel
                self._update_back_button_state()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch or configure preprocessing dialog: {str(e)}")
            self._log_debug(f"Error launching preprocessing dialog: {e}")
            traceback.print_exc() # Use traceback module here for full stack trace
            self.stacked_widget.setCurrentWidget(self.paradigm_window) # Return to paradigm window on error
            self._update_back_button_state()

    @pyqtSlot(dict, dict)
    def handle_preprocessing_results(self, results, paradigm_parameters):
        """
        Receives preprocessing results from PreprocessingDialog and
        launches the RealTimePsdGui for real-time feedback.
        """
        self._log_debug("Entering handle_preprocessing_results method.")
        self._log_debug(f"Received results keys: {list(results.keys())}")
        self._log_debug(f"Received paradigm_parameters: {paradigm_parameters}")

        # Store preprocessing results
        self.asr_obj = results.get('asr')
        self.ica_obj = results.get('ica')
        self.artifact_components = results.get('artifact_components', [])
        self.bad_channels = results.get('bad_channels', [])
        self.eeg_info = results.get('info') # MNE Info object

        self._log_debug(f"Stored ASR: {self.asr_obj is not None}, ICA: {self.ica_obj is not None}, Info: {self.eeg_info is not None}")

        try:
            # Explicitly check for expected keys that are critical for RealTimePsdGui
            required_results_keys = ['modality_result', 'modality', 'info']
            if not all(key in results and results[key] is not None for key in required_results_keys):
                missing_keys = [key for key in required_results_keys if key not in results or results[key] is None]
                self._log_debug(f"Missing or None critical results for RealTimePsdGui: {missing_keys}")
                QMessageBox.critical(self, "Error", f"Missing critical preprocessing results for feedback: {', '.join(missing_keys)}")
                # Go back to paradigm window if critical data is missing
                self.stacked_widget.setCurrentWidget(self.paradigm_window)
                self._update_back_button_state()
                return

            modality_df = results['modality_result']
            modality_type_from_preproc = results['modality'] # This is the backend string (e.g., 'psd_auc')

            # --- Prepare parameters for RealTimePsdGui's configure_parameters method ---
            # These parameters define *what* to compute in real-time, based on paradigm choice
            feedback_params = {
                'modality': modality_type_from_preproc,
                'baseline_modality_result': modality_df,
                'user_data_dir': self.user_data_dir,
                'raw_info': self.eeg_info
            }

            # Add channels and frequency bands based on modality_type from the passed paradigm_parameters
            if paradigm_parameters['modality'] == "psd_auc":
                feedback_params['channel'] = paradigm_parameters.get('channel')
                feedback_params['frequency_band'] = paradigm_parameters.get('frequency_band')
            elif paradigm_parameters['modality'] == "psd_ratio":
                feedback_params['channel'] = paradigm_parameters.get('channel')
                feedback_params['band_1'] = paradigm_parameters.get('band_1')
                feedback_params['band_2'] = paradigm_parameters.get('band_2')
            elif paradigm_parameters['modality'] == "pac":
                feedback_params['channel'] = paradigm_parameters.get('channel')
                feedback_params['phase_frequency'] = paradigm_parameters.get('phase_frequency')
                feedback_params['amplitude_frequency'] = paradigm_parameters.get('amplitude_frequency')
            elif paradigm_parameters['modality'] == "coh":
                feedback_params['channel_1'] = paradigm_parameters.get('channel_1')
                feedback_params['channel_2'] = paradigm_parameters.get('channel_2')
                feedback_params['frequency_band'] = paradigm_parameters.get('frequency_band')
            else:
                QMessageBox.warning(self, "Feedback Error", f"Unsupported modality for feedback display: {paradigm_parameters['modality']}")
                self.stacked_widget.setCurrentWidget(self.paradigm_window)
                self._update_back_button_state()
                return

            self._log_debug(f"Prepared feedback_params: {feedback_params}")

            # Instantiate RealTimePsdGui, passing the required preprocessing objects and feedback parameters
            if self.psd_gui:
                # If RealTimePsdGui already exists, remove it from stacked widget and delete it
                self.stacked_widget.removeWidget(self.psd_gui)
                self.psd_gui.deleteLater() # Safely delete the old instance
                self.psd_gui = None

            self.psd_gui = RealTimePsdGui(
                asr=self.asr_obj,
                ica=self.ica_obj,
                artifact_components=self.artifact_components,
                bad_channels=self.bad_channels,
                info=self.eeg_info, # Pass the MNE Info object
                parameters=feedback_params, # Pass the selected paradigm parameters for real-time calculation
                parent=self
            )

            # Connect the back signal from RealTimePsdGui to a slot in MainWindow
            try:
                self.psd_gui.back_to_paradigm_signal.disconnect(self.show_paradigm_from_realtime)
            except TypeError:
                pass # No prior connection to disconnect
            self.psd_gui.back_to_paradigm_signal.connect(self.show_paradigm_from_realtime)

            # Add the new RealTimePsdGui widget to the stacked widget
            self.stacked_widget.addWidget(self.psd_gui)

            # Set RealTimePsdGui as the current visible widget
            self.stacked_widget.setCurrentWidget(self.psd_gui)

            # Disable the main back button while in RealTimePsdGui
            self.back_button.setEnabled(False)

            self.realtime_feedback_launched = True # Crucial: Set the flag here!
            self._log_debug("RealTimePsdGui launched and flag set to True.")

        except Exception as e:
            self._log_debug(f"An unhandled exception occurred during RealTimePsdGui launch: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Internal Error", f"An internal error occurred while processing results or launching feedback: {str(e)}")
            self.stacked_widget.setCurrentWidget(self.paradigm_window) # Return to paradigm window on error
            self._update_back_button_state()

    @pyqtSlot()
    def show_paradigm_from_realtime(self):
        """
        Slot to return to the ParadigmGui window after closing RealTimePsdGui.
        """
        self._log_debug("Returning to Paradigm Choice window from RealTimePsdGui.")
        if self.paradigm_window:
            self.stacked_widget.setCurrentWidget(self.paradigm_window)
            self._update_back_button_state() # Update back button state
        else:
            # Fallback if paradigm_window was somehow not initialized
            self._log_debug("ParadigmGui instance not found after RealTimePsdGui closed. Returning to Baseline.")
            self.stacked_widget.setCurrentWidget(self.baseline_window)
            self._update_back_button_state()

    def go_back(self):
        """
        Handles global back navigation. Prevents direct back from real-time feedback.
        """
        current_widget = self.stacked_widget.currentWidget()

        # Prevent going back if RealTimePsdGui is active
        if current_widget == self.psd_gui:
            QMessageBox.warning(self, "Navigation Blocked", "Please use the 'Back to Paradigm' button within the real-time feedback window to stop the stream before navigating back.")
            return

        # Handle navigation for other pages
        if self.stacked_widget.currentIndex() > 0:
            previous_index = self.stacked_widget.currentIndex() - 1
            self.stacked_widget.setCurrentIndex(previous_index)
            self._update_back_button_state()
        else:
            # If on the welcome page, the back button should be disabled, but a safety check
            self.back_button.setEnabled(False)

def main():
    """Main function to start the QApplication."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
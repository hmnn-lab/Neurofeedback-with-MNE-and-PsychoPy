from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton,
    QComboBox, QLineEdit, QLabel, QPushButton, QWidget, QStatusBar, QMessageBox, QDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QDateTime # QDateTime for more accurate timestamping
from datetime import datetime, timezone, timedelta
import os
from final_preproc_window import PreprocessingDialog

class ConfirmationDialog(QDialog):
    def __init__(self, parameters, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confirm Parameters")
        self.setMinimumSize(300, 250)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        params_label = QLabel("Selected Parameters:")
        params_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(params_label)

        params_text = ""
        for key, value in parameters.items():
            if isinstance(value, dict):
                params_text += f"<b>{key.replace('_', ' ').title()}:</b><br>"
                for sub_key, sub_value in value.items():
                    params_text += f"&nbsp;&nbsp;&nbsp;- {sub_key.replace('_', ' ').title()}: {sub_value}<br>"
            elif isinstance(value, list):
                params_text += f"<b>{key.replace('_', ' ').title()}:</b> {', '.join(value)}<br>"
            else:
                params_text += f"<b>{key.replace('_', ' ').title()}:</b> {value}<br>"

        params_display = QLabel(params_text)
        params_display.setWordWrap(True)
        params_display.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        params_display.setTextFormat(Qt.TextFormat.RichText) # Enable HTML-like formatting
        layout.addWidget(params_display)

        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.setFixedWidth(80)
        cancel_button.setFixedWidth(80)

        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        layout.addStretch()
        layout.addLayout(button_layout)
        self.setLayout(layout)

class ParadigmGui(QMainWindow):
    # This signal will be used by MainWindow to launch the PreprocessingDialog
    # and pass the collected parameters directly.
    # It carries the full dictionary of collected paradigm parameters.
    launch_preprocessing_requested = pyqtSignal(dict)

    def __init__(self, baseline_file, channel_names, num_channels, user_data_dir, main_window, parent=None):
        super().__init__(parent)
        self.baseline_file = baseline_file
        self.channel_names = channel_names
        self.num_channels = num_channels
        self.user_data_dir = user_data_dir
        self.main_window = main_window  # Store MainWindow reference
        self.frequency_bands = {
            "Delta": [1, 4],
            "Theta": [4, 8],
            "Alpha": [8, 12],
            "Beta": [12, 30],
            "Gamma": [30, 100]
        }
        self.modality_parameters = {
            "Single Frequency": ["Channel", "Frequency Band"],
            "Dual Frequency": ["Channel", "Band 1", "Band 2"],
            "Coherence": ["Channel 1", "Channel 2", "Frequency Band"],
            "Phase Amplitude Coupling": ["Channel", "Phase Frequency", "Amplitude Frequency"]
        }
        self.modality_mapping = {
            "Single Frequency": "psd_auc",
            "Dual Frequency": "psd_ratio",
            "Phase Amplitude Coupling": "pac",
            "Coherence": "coh" # Make sure this matches what compute_baseline_modality expects
        }
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Paradigm Choice")
        self.setGeometry(100, 100, 650, 500)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        title = QLabel("Paradigm Choice")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        main_layout.addWidget(title)

        file_label = QLabel(f"Baseline File: <b>{os.path.basename(self.baseline_file)}</b>")
        file_label.setTextFormat(Qt.TextFormat.RichText)
        main_layout.addWidget(file_label)

        channels_label = QLabel(f"Channels: <b>{', '.join(self.channel_names) if self.channel_names else 'None'}</b>")
        channels_label.setTextFormat(Qt.TextFormat.RichText)
        main_layout.addWidget(channels_label)

        num_channels_label = QLabel(f"Number of Channels: <b>{self.num_channels if self.num_channels else 'Not specified'}</b>")
        num_channels_label.setTextFormat(Qt.TextFormat.RichText)
        main_layout.addWidget(num_channels_label)

        modality_group = QGroupBox("Select Modality")
        modality_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        modality_layout = QVBoxLayout()
        modality_layout.setSpacing(8)
        modality_group.setLayout(modality_layout)

        self.radio_buttons = {}
        for modality in self.modality_parameters.keys():
            rb = QRadioButton(modality)
            self.radio_buttons[modality] = rb
            modality_layout.addWidget(rb)
        modality_layout.addStretch()

        self.parameters_group = QGroupBox("Parameters")
        self.parameters_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.parameters_layout = QVBoxLayout()
        self.parameters_layout.setSpacing(10)
        self.parameters_layout.setContentsMargins(10, 15, 10, 10)
        self.parameters_group.setLayout(self.parameters_layout)

        for modality, rb in self.radio_buttons.items():
            rb.toggled.connect(self.update_parameters)

        button_layout = QHBoxLayout()
        next_button = QPushButton("Next")
        next_button.setFixedWidth(100)
        # Change here: Connect to _confirm_and_launch which emits the signal
        next_button.clicked.connect(self._confirm_and_launch) 
        clear_button = QPushButton("Clear")
        clear_button.setFixedWidth(100)
        clear_button.clicked.connect(self.clear_parameters)
        button_layout.addStretch()
        button_layout.addWidget(next_button)
        button_layout.addWidget(clear_button)

        main_layout.addWidget(modality_group)
        main_layout.addWidget(self.parameters_group)
        main_layout.addStretch()
        main_layout.addLayout(button_layout)

        self.radio_buttons["Single Frequency"].setChecked(True)
        self.update_parameters()

    def update_parameters(self):
        while self.parameters_layout.count():
            child = self.parameters_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                self._clear_layout(child.layout())

        selected_modality = next(
            name for name, radio in self.radio_buttons.items() if radio.isChecked()
        )
        self.statusBar.showMessage(f"Configuring parameters for {selected_modality}")

        self.current_param_widgets = {}
        if selected_modality == "Single Frequency":
            self.create_single_frequency_params()
        elif selected_modality == "Dual Frequency":
            self.create_dual_frequency_params()
        elif selected_modality == "Coherence":
            self.create_coherence_params()
        elif selected_modality == "Phase Amplitude Coupling":
            self.create_pac_params()

    def _clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                elif item.layout():
                    self._clear_layout(item.layout())

    def create_frequency_band_widget(self, band_label="Frequency Band"):
        band_container_widget = QWidget()
        h_layout = QHBoxLayout(band_container_widget)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(10)

        band_combo_layout = QVBoxLayout()
        band_combo_layout.setContentsMargins(0, 0, 0, 0)
        band_combo_layout.setSpacing(4)
        band_combo_layout.addWidget(QLabel(f"{band_label}:"))
        band_combo = QComboBox()
        band_combo.addItems(self.frequency_bands.keys())
        band_combo.setFixedWidth(120)
        band_combo_layout.addWidget(band_combo)
        band_combo_layout.addStretch()

        freq_edits_layout = QVBoxLayout()
        freq_edits_layout.setContentsMargins(0, 0, 0, 0)
        freq_edits_layout.setSpacing(4)

        low_edit = QLineEdit()
        low_edit.setPlaceholderText("Low")
        low_edit.setFixedWidth(60)
        low_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        low_edit.setValidator(QDoubleValidator()) # Ensure numeric input
        low_edit.textEdited.connect(lambda: self.validate_frequency_input(low_edit))

        high_edit = QLineEdit()
        high_edit.setPlaceholderText("High")
        high_edit.setFixedWidth(60)
        high_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        high_edit.setValidator(QDoubleValidator()) # Ensure numeric input
        high_edit.textEdited.connect(lambda: self.validate_frequency_input(high_edit))

        low_h_layout = QHBoxLayout()
        low_h_layout.addWidget(QLabel("Lower (Hz):"))
        low_h_layout.addWidget(low_edit)
        low_h_layout.addStretch()

        high_h_layout = QHBoxLayout()
        high_h_layout.addWidget(QLabel("Upper (Hz):"))
        high_h_layout.addWidget(high_edit)
        high_h_layout.addStretch()

        freq_edits_layout.addLayout(low_h_layout)
        freq_edits_layout.addLayout(high_h_layout)
        freq_edits_layout.addStretch()

        h_layout.addLayout(band_combo_layout)
        h_layout.addLayout(freq_edits_layout)
        h_layout.addStretch()

        band_combo.currentTextChanged.connect(
            lambda: self.update_frequency_ranges(band_combo, low_edit, high_edit)
        )
        self.update_frequency_ranges(band_combo, low_edit, high_edit)

        return band_container_widget, band_combo, low_edit, high_edit

    def validate_frequency_input(self, line_edit):
        """Basic validation for frequency input to ensure it's a valid float."""
        text = line_edit.text()
        try:
            float(text)
            line_edit.setStyleSheet("") # Clear any error highlighting
        except ValueError:
            line_edit.setStyleSheet("border: 1px solid red;") # Highlight invalid input

    def create_single_frequency_params(self):
        main_h_layout = QHBoxLayout()
        main_h_layout.setSpacing(20)

        channel_group_widget = QWidget()
        channel_v_layout = QVBoxLayout(channel_group_widget)
        channel_v_layout.setContentsMargins(0, 0, 0, 0)
        channel_v_layout.addWidget(QLabel("Channel:"))
        channel_combo = QComboBox()
        channel_combo.addItems(self.channel_names)
        channel_combo.setMinimumWidth(150)
        channel_v_layout.addWidget(channel_combo)
        channel_v_layout.addStretch()

        main_h_layout.addWidget(channel_group_widget)
        self.current_param_widgets['channel_sf'] = channel_combo

        freq_band_widget, band_combo, low_edit, high_edit = self.create_frequency_band_widget("Frequency Band")
        main_h_layout.addWidget(freq_band_widget)
        self.current_param_widgets['freq_band_sf'] = {'combo': band_combo, 'low': low_edit, 'high': high_edit}

        main_h_layout.addStretch()
        self.parameters_layout.addLayout(main_h_layout)
        self.parameters_layout.addStretch()

    def create_dual_frequency_params(self):
        channel_group_widget = QWidget()
        channel_v_layout = QVBoxLayout(channel_group_widget)
        channel_v_layout.setContentsMargins(0, 0, 0, 0)
        channel_v_layout.addWidget(QLabel("Channel:"))
        channel_combo = QComboBox()
        channel_combo.addItems(self.channel_names)
        channel_combo.setMinimumWidth(150)
        channel_combo.setCurrentIndex(0)  # Set a default selection if channel_names is not empty
        channel_v_layout.addWidget(channel_combo)
        channel_v_layout.addStretch()
        self.parameters_layout.addWidget(channel_group_widget)
        self.current_param_widgets['channel_df'] = channel_combo

        band1_widget, band1_combo, band1_low_edit, band1_high_edit = self.create_frequency_band_widget("Band 1")
        self.parameters_layout.addWidget(band1_widget)
        self.current_param_widgets['band1_df'] = {'combo': band1_combo, 'low': band1_low_edit, 'high': band1_high_edit}

        band2_widget, band2_combo, band2_low_edit, band2_high_edit = self.create_frequency_band_widget("Band 2")
        self.parameters_layout.addWidget(band2_widget)
        self.current_param_widgets['band2_df'] = {'combo': band2_combo, 'low': band2_low_edit, 'high': band2_high_edit}

        self.parameters_layout.addStretch()

    def create_coherence_params(self):
        channel1_group_widget = QWidget()
        channel1_v_layout = QVBoxLayout(channel1_group_widget)
        channel1_v_layout.setContentsMargins(0, 0, 0, 0)
        channel1_v_layout.addWidget(QLabel("Channel 1:"))
        channel1_combo = QComboBox()
        channel1_combo.addItems(self.channel_names)
        channel1_combo.setMinimumWidth(150)
        channel1_combo.setCurrentIndex(0) # Default selection
        channel1_v_layout.addWidget(channel1_combo)
        channel1_v_layout.addStretch()
        self.parameters_layout.addWidget(channel1_group_widget)
        self.current_param_widgets['channel1_coh'] = channel1_combo

        channel2_group_widget = QWidget()
        channel2_v_layout = QVBoxLayout(channel2_group_widget)
        channel2_v_layout.setContentsMargins(0, 0, 0, 0)
        channel2_v_layout.addWidget(QLabel("Channel 2:"))
        channel2_combo = QComboBox()
        channel2_combo.addItems(self.channel_names)
        channel2_combo.setMinimumWidth(150)
        # Set to the next index if available, or 0 if only one channel
        if len(self.channel_names) > 1:
            channel2_combo.setCurrentIndex(1)
        else:
            channel2_combo.setCurrentIndex(0)
        channel2_v_layout.addWidget(channel2_combo)
        channel2_v_layout.addStretch()
        self.parameters_layout.addWidget(channel2_group_widget)
        self.current_param_widgets['channel2_coh'] = channel2_combo

        freq_band_widget, band_combo, low_edit, high_edit = self.create_frequency_band_widget("Frequency Band")
        self.parameters_layout.addWidget(freq_band_widget)
        self.current_param_widgets['freq_band_coh'] = {'combo': band_combo, 'low': low_edit, 'high': high_edit}
        self.parameters_layout.addStretch()

    def create_pac_params(self):
        channel_group_widget = QWidget()
        channel_v_layout = QVBoxLayout(channel_group_widget)
        channel_v_layout.setContentsMargins(0, 0, 0, 0)
        channel_v_layout.addWidget(QLabel("Channel:"))
        channel_combo = QComboBox()
        channel_combo.addItems(self.channel_names)
        channel_combo.setMinimumWidth(150)
        channel_combo.setCurrentIndex(0) # Default selection
        channel_v_layout.addWidget(channel_combo)
        channel_v_layout.addStretch()
        self.parameters_layout.addWidget(channel_group_widget)
        self.current_param_widgets['channel_pac'] = channel_combo

        phase_freq_widget, phase_combo, phase_low_edit, phase_high_edit = self.create_frequency_band_widget("Phase Frequency")
        self.parameters_layout.addWidget(phase_freq_widget)
        self.current_param_widgets['phase_freq_pac'] = {'combo': phase_combo, 'low': phase_low_edit, 'high': phase_high_edit}

        amp_freq_widget, amp_combo, amp_low_edit, amp_high_edit = self.create_frequency_band_widget("Amplitude Frequency")
        self.parameters_layout.addWidget(amp_freq_widget)
        self.current_param_widgets['amp_freq_pac'] = {'combo': amp_combo, 'low': amp_low_edit, 'high': amp_high_edit}

        self.parameters_layout.addStretch()

    def update_frequency_ranges(self, combo, low_edit, high_edit):
        band = combo.currentText()
        low, high = self.frequency_bands[band]
        low_edit.setText(str(low))
        high_edit.setText(str(high))

    def _confirm_and_launch(self):
        """
        Collects parameters, shows confirmation, and if accepted,
        emits launch_preprocessing_requested signal.
        """
        parameters = self.get_parameters()
        if parameters:
            # Add baseline and channel info to the parameters for completeness
            parameters["baseline_file"] = self.baseline_file
            parameters["channel_names"] = self.channel_names
            parameters["num_channels"] = self.num_channels
            parameters["user_data_dir"] = self.user_data_dir # Add user_data_dir here

            dialog = ConfirmationDialog(parameters, self)
            self.statusBar.showMessage("Opened confirmation dialog")
            
            if dialog.exec():
                self.statusBar.showMessage("Parameters confirmed")
                print(f"[DEBUG {QDateTime.currentDateTime().toString(Qt.DateFormat.ISODateWithMs)} IST] Parameters confirmed: {parameters}")
                # Emit the signal with all collected parameters
                self.launch_preprocessing_requested.emit(parameters)
                # The main_window will now handle showing the PreprocessingDialog
                # This ParadigmGui window should hide or close itself if needed.
                self.hide() # Or self.close() depending on desired flow
            else:
                self.statusBar.showMessage("Confirmation canceled")
        else:
            self.statusBar.showMessage("Error: Could not retrieve parameters. Please select a modality and ensure all fields are filled.")
            QMessageBox.critical(self, "Error", "Could not retrieve parameters. Please select a modality and ensure all fields are filled.")

    def get_parameters(self):
        selected_modality = next(
            (name for name, radio in self.radio_buttons.items() if radio.isChecked()), None
        )
        if not selected_modality:
            return None

        params = {"modality": self.modality_mapping[selected_modality]}
        try:
            if selected_modality == "Single Frequency":
                channel = self.current_param_widgets['channel_sf'].currentText()
                if not channel: raise ValueError("Channel not selected for Single Frequency.")
                freq_band_info = self.current_param_widgets['freq_band_sf']
                freq_band_name = freq_band_info['combo'].currentText()
                low_freq_str = freq_band_info['low'].text().strip()
                high_freq_str = freq_band_info['high'].text().strip()
                
                if not low_freq_str or not high_freq_str:
                    raise ValueError("Frequency range cannot be empty.")
                low_freq = float(low_freq_str)
                high_freq = float(high_freq_str)
                
                if low_freq >= high_freq:
                    raise ValueError("Lower frequency must be less than upper frequency.")
                
                params["channel"] = channel
                params["frequency_band"] = {"name": freq_band_name, "low": low_freq, "high": high_freq}

            elif selected_modality == "Dual Frequency":
                channel = self.current_param_widgets['channel_df'].currentText()
                if not channel: raise ValueError("Channel not selected for Dual Frequency.")
                band1_info = self.current_param_widgets['band1_df']
                band2_info = self.current_param_widgets['band2_df']

                band1_name = band1_info['combo'].currentText()
                band1_low_str = band1_info['low'].text().strip()
                band1_high_str = band1_info['high'].text().strip()
                band2_name = band2_info['combo'].currentText()
                band2_low_str = band2_info['low'].text().strip()
                band2_high_str = band2_info['high'].text().strip()

                if not all([band1_low_str, band1_high_str, band2_low_str, band2_high_str]):
                    raise ValueError("All frequency ranges must be filled for Dual Frequency.")

                band1_low = float(band1_low_str)
                band1_high = float(band1_high_str)
                band2_low = float(band2_low_str)
                band2_high = float(band2_high_str)

                if band1_low >= band1_high or band2_low >= band2_high:
                    raise ValueError("Lower frequency must be less than upper frequency for both bands.")
                
                params["channel"] = channel
                params["band_1"] = {"name": band1_name, "low": band1_low, "high": band1_high}
                params["band_2"] = {"name": band2_name, "low": band2_low, "high": band2_high}
            
            elif selected_modality == "Coherence":
                channel1 = self.current_param_widgets['channel1_coh'].currentText()
                channel2 = self.current_param_widgets['channel2_coh'].currentText()
                if not channel1 or not channel2: raise ValueError("Both Channel 1 and Channel 2 must be selected for Coherence.")
                if channel1 == channel2:
                    raise ValueError("Channel 1 and Channel 2 must be different for Coherence.")
                
                freq_band_info = self.current_param_widgets['freq_band_coh']
                freq_band_name = freq_band_info['combo'].currentText()
                low_freq_str = freq_band_info['low'].text().strip()
                high_freq_str = freq_band_info['high'].text().strip()

                if not low_freq_str or not high_freq_str:
                    raise ValueError("Frequency range cannot be empty for Coherence.")
                low_freq = float(low_freq_str)
                high_freq = float(high_freq_str)
                if low_freq >= high_freq:
                    raise ValueError("Lower frequency must be less than upper frequency for Coherence.")
                
                params["channel_1"] = channel1
                params["channel_2"] = channel2
                params["frequency_band"] = {"name": freq_band_name, "low": low_freq, "high": high_freq}

            elif selected_modality == "Phase Amplitude Coupling":
                channel = self.current_param_widgets['channel_pac'].currentText()
                if not channel: raise ValueError("Channel not selected for Phase Amplitude Coupling.")
                
                phase_freq_info = self.current_param_widgets['phase_freq_pac']
                amp_freq_info = self.current_param_widgets['amp_freq_pac']

                phase_name = phase_freq_info['combo'].currentText()
                phase_low_str = phase_freq_info['low'].text().strip()
                phase_high_str = phase_freq_info['high'].text().strip()
                amp_name = amp_freq_info['combo'].currentText()
                amp_low_str = amp_freq_info['low'].text().strip()
                amp_high_str = amp_freq_info['high'].text().strip()

                if not all([phase_low_str, phase_high_str, amp_low_str, amp_high_str]):
                    raise ValueError("All frequency ranges must be filled for Phase Amplitude Coupling.")

                phase_low = float(phase_low_str)
                phase_high = float(phase_high_str)
                amp_low = float(amp_low_str)
                amp_high = float(amp_high_str)

                if phase_low >= phase_high or amp_low >= amp_high:
                    raise ValueError("Lower frequency must be less than upper frequency for both phase and amplitude bands.")

                params["channel"] = channel
                params["phase_frequency"] = {"name": phase_name, "low": phase_low, "high": phase_high}
                params["amplitude_frequency"] = {"name": amp_name, "low": amp_low, "high": amp_high}
            
            else:
                self.statusBar.showMessage(f"Error: Unknown modality {selected_modality}.")
                return None

        except ValueError as e:
            self.statusBar.showMessage(f"Input Error: {str(e)}")
            QMessageBox.warning(self, "Input Error", str(e))
            return None
        except Exception as e:
            self.statusBar.showMessage(f"An unexpected error occurred while collecting parameters: {str(e)}")
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {str(e)}")
            return None

        return params

    def clear_parameters(self):
        # Reset to default selected modality
        self.radio_buttons["Single Frequency"].setChecked(True)
        # Force re-update parameters to clear any manual edits
        self.update_parameters() 
        self.statusBar.showMessage("Parameters cleared and reset to Single Frequency")

    # Removed the direct call to launch_preprocessing_dialog from here.
    # The _confirm_and_launch method now emits a signal, and MainWindow will handle it.
    # def launch_preprocessing_dialog(self, parameters):
    #    ... (This method is no longer directly called by ParadigmGui)

    def closeEvent(self, event):
        self.statusBar.showMessage("Closing Paradigm Choice window")
        event.accept()

# Required for QDoubleValidator in create_frequency_band_widget
from PyQt6.QtGui import QDoubleValidator
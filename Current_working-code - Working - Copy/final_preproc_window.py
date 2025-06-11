import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit,
    QMessageBox, QHBoxLayout, QTextEdit
)
from PyQt6.QtCore import pyqtSignal
from preproc_initialize import preproc_flow
from cal_baseline_psd import compute_baseline_psd


class PreprocessingGUI(QWidget):  
    def __init__(self, baseline_file_path=None):
        super().__init__()
        self.setWindowTitle("Preprocessing Window")
        self.setGeometry(300, 300, 500, 400)

        self.file_path = baseline_file_path
        self.preprocessed_params = {}

        layout = QVBoxLayout()

        title = QLabel("EEG Preprocessing")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        self.file_label = QLabel(
            f"Selected file: {os.path.basename(self.file_path)}" if self.file_path else "No file selected"
        )
        layout.addWidget(self.file_label)

        # Number of channels input
        channel_layout = QHBoxLayout()
        channel_label = QLabel("Number of channels:")
        self.channel_input = QLineEdit()
        self.channel_input.setPlaceholderText("e.g., 8 or 32")
        channel_layout.addWidget(channel_label)
        channel_layout.addWidget(self.channel_input)
        layout.addLayout(channel_layout)

        # Channel names input
        channel_names_label = QLabel("Channel names (comma separated):")
        self.channel_names_input = QTextEdit()
        self.channel_names_input.setPlaceholderText("e.g., Fp1, Fp2, F3, F4, C3, C4, P3, P4")
        self.channel_names_input.setFixedHeight(60)
        layout.addWidget(channel_names_label)
        layout.addWidget(self.channel_names_input)

        self.preproc_button = QPushButton("Run Preprocessing + PSD AUC")
        self.preproc_button.clicked.connect(self.run_preprocessing)
        layout.addWidget(self.preproc_button)

        self.psd_status_label = QLabel("")
        layout.addWidget(self.psd_status_label)

        self.next_button = QPushButton("Next")
        self.next_button.setEnabled(True)  # Disabled until preprocessing completes
        layout.addWidget(self.next_button)

        self.setLayout(layout)


    def run_preprocessing(self):
        if not self.file_path:
            QMessageBox.warning(self, "Input Error", "No baseline file provided.")
            return

        try:
            num_channels_text = self.channel_input.text().strip()
            if not num_channels_text.isdigit():
                raise ValueError("Please enter a valid number of channels (integer).")
            num_channels = int(num_channels_text)

            # Parse channel names input
            channel_names_text = self.channel_names_input.toPlainText().strip()
            if channel_names_text:
                channel_names = [name.strip() for name in channel_names_text.split(',') if name.strip()]
                if len(channel_names) != num_channels:
                    raise ValueError(f"Number of channel names ({len(channel_names)}) does not match number of channels ({num_channels}).")
            else:
                channel_names = None

            # Run preprocessing, get cleaned raw and others
            raw_cleaned, bad_channels, asr, ica, artifact_components = preproc_flow(
                self.file_path,
                n_channels=num_channels,
                channel_names=channel_names
            )

            # Compute PSD AUC on cleaned data directly
            psd_auc_result, excel_path, plot_path = compute_baseline_psd(raw_cleaned)

            # Store results
            self.preprocessed_params = {
                'raw_cleaned': raw_cleaned,
                'bad_channels': bad_channels,
                'asr': asr,
                'ica': ica,
                'artifact_components': artifact_components,
                'psd_auc': psd_auc_result,
                'psd_excel': excel_path,
                'psd_plot': plot_path,
            }

            self.psd_status_label.setText("PSD AUC calculated successfully.")
            QMessageBox.information(self, "Success", "Preprocessing and PSD AUC calculation completed.")
            self.next_button.setEnabled(True)


        except Exception as e:
            QMessageBox.critical(self, "Error", f"Processing Failed:\n{str(e)}")


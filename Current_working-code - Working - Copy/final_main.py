import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel,
    QLineEdit, QPushButton, QHBoxLayout, QSizePolicy, QStackedWidget,
    QMessageBox
)
from PyQt6.QtCore import Qt

# Import your other windows as before
from final_baseline_window import BaselineWindow
from final_preproc_window import PreprocessingGUI
from final_power_window import RealTimePsdGui
from final_visual_window import VisualGui


class WelcomePage(QWidget):
    def __init__(self):
        super().__init__()

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
        layout.addWidget(self.start_button, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setLayout(layout)

        self.setMinimumWidth(500)
        self.setMinimumHeight(250)


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("EEG Real-Time System")

        # Stack widget for all pages
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Step 0: Welcome Page
        self.welcome_page = WelcomePage()
        self.welcome_page.start_button.clicked.connect(self.go_to_baseline)
        self.stack.addWidget(self.welcome_page)

        # Step 1: Baseline window
        self.baseline_window = BaselineWindow()
        # Connect the new signal to handle baseline completion
        self.baseline_window.baseline_ready.connect(self.handle_baseline_ready)
        self.stack.addWidget(self.baseline_window)

        # Step 2: Preprocessing window placeholder (will be created dynamically)
        self.preproc_window = None

        # Step 3 & 4 windows initialized later
        self.realtime_window = None
        self.visual_window = None

        # User info and data folder
        self.user_name = None
        self.user_id = None
        self.user_data_dir = None

        self.resize(600, 320)
        self.setMinimumSize(500, 250)

    

    def go_to_baseline_page(self):
        self.stack.setCurrentWidget(self.baseline_window)

    def go_to_preprocessing_page(self):
        self.stack.setCurrentWidget(self.preproc_window)

    def go_to_realtime_page(self):
        self.stack.setCurrentWidget(self.realtime_window)



    def go_to_baseline(self):
        self.user_name = self.welcome_page.name_input.text().strip()
        self.user_id = self.welcome_page.id_input.text().strip()

        if not self.user_name or not self.user_id:
            QMessageBox.warning(self, "Input Required", "Please enter both your name and ID before continuing.")
            return

        folder_name = f"{self.user_name}_{self.user_id}".replace(" ", "_")
        base_path = "./recordings"
        self.user_data_dir = os.path.join(base_path, folder_name)
        os.makedirs(self.user_data_dir, exist_ok=True)

        self.stack.setCurrentWidget(self.baseline_window)

    def handle_baseline_ready(self, baseline_file_path):
        print(f"[DEBUG] MainApp received baseline file: {baseline_file_path}")
        # Store user_data_dir based on baseline file location
        self.user_data_dir = os.path.dirname(baseline_file_path)
        self.go_to_preprocessing(baseline_file_path)

    def go_to_preprocessing(self, baseline_file_path):
        # Create PreprocessingGUI with baseline file path
        self.preproc_window = PreprocessingGUI(baseline_file_path=baseline_file_path)
        # Connect preprocessing 'next' button to go to realtime PSD window
        self.preproc_window.next_button.clicked.connect(self.go_to_realtime_psd)
        self.stack.addWidget(self.preproc_window)
        self.stack.setCurrentWidget(self.preproc_window)

    def go_to_realtime_psd(self):
        try:
            asr = self.preproc_window.preprocessed_params.get('asr')
            ica = self.preproc_window.preprocessed_params.get('ica')
            artifact_comps = self.preproc_window.preprocessed_params.get('artifact_components')
            bad_channels = self.preproc_window.preprocessed_params.get('bad_channels')
            info = self.preproc_window.preprocessed_params.get('raw_cleaned').info

            lsl_stream_name = "SignalGenerator"  # example

            self.realtime_window = RealTimePsdGui(
                asr=asr,
                ica=ica,
                artifact_components=artifact_comps,
                bad_channels=bad_channels,
                info=info
            )
            # self.realtime_window.next_button.clicked.connect(self.go_to_visual)
            self.stack.addWidget(self.realtime_window)
            self.stack.setCurrentWidget(self.realtime_window)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot open PSD window:\n{e}")

    def go_to_visual(self):
        try:
            info = self.preproc_window.preprocessed_params.get('raw_cleaned').info
            lsl_stream_name = "PowerChangeStream"  # example

            self.visual_window = VisualGui(
                asr=self.preproc_window.preprocessed_params.get('asr'),
                ica=self.preproc_window.preprocessed_params.get('ica'),
                artifact_components=self.preproc_window.preprocessed_params.get('artifact_components'),
                bad_channels=self.preproc_window.preprocessed_params.get('bad_channels'),
                info=info,
                lsl_stream_name=lsl_stream_name
            )
            self.stack.addWidget(self.visual_window)
            self.stack.setCurrentWidget(self.visual_window)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot open visualization window:\n{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainApp()
    main_win.show()
    sys.exit(app.exec())

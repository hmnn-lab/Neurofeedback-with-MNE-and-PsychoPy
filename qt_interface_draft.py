import sys
import PyQt5
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QFormLayout, QLabel, QLineEdit, QSpinBox, QComboBox, QPushButton,
                             QFileDialog, QCheckBox, QStackedWidget, QDoubleSpinBox)

class GeneralSettings(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        layout = QFormLayout()
        
        self.session_duration = QSpinBox()
        self.session_duration.setRange(1, 120)
        self.session_duration.setSuffix(" min")
        
        self.sampling_rate = QComboBox()
        self.sampling_rate.addItems(["128 Hz", "256 Hz", "512 Hz", "1024 Hz"])
        
        self.data_save_path = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_save_path)
        
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.data_save_path)
        path_layout.addWidget(browse_btn)
        
        layout.addRow(QLabel("Session Duration:"), self.session_duration)
        layout.addRow(QLabel("Sampling Rate:"), self.sampling_rate)
        layout.addRow(QLabel("Data Save Path:"), path_layout)
        
        self.setLayout(layout)
    
    def browse_save_path(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if directory:
            self.data_save_path.setText(directory)

class NeurofeedbackSettings(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        self.training_type = QComboBox()
        self.training_type.addItems(["Alpha/Theta", "SMR", "Slow Cortical Potentials", "fNIRS"])
        self.training_type.currentIndexChanged.connect(self.update_settings)
        
        self.settings_stack = QStackedWidget()
        
        # Alpha/Theta settings
        alpha_theta = QWidget()
        a_layout = QFormLayout()
        self.alpha_threshold = QDoubleSpinBox()
        self.alpha_threshold.setRange(0.1, 100.0)
        a_layout.addRow(QLabel("Alpha Threshold (μV):"), self.alpha_threshold)
        alpha_theta.setLayout(a_layout)
        
        # SMR settings
        smr = QWidget()
        s_layout = QFormLayout()
        self.smr_low = QDoubleSpinBox()
        self.smr_low.setRange(12, 15)
        self.smr_high = QDoubleSpinBox()
        self.smr_high.setRange(15, 18)
        s_layout.addRow(QLabel("Low Frequency (Hz):"), self.smr_low)
        s_layout.addRow(QLabel("High Frequency (Hz):"), self.smr_high)
        smr.setLayout(s_layout)
        
        self.settings_stack.addWidget(alpha_theta)
        self.settings_stack.addWidget(smr)
        self.settings_stack.addWidget(QWidget())  # Placeholder for other types
        
        layout.addWidget(QLabel("Training Type:"))
        layout.addWidget(self.training_type)
        layout.addWidget(self.settings_stack)
        
        self.setLayout(layout)
    
    def update_settings(self, index):
        self.settings_stack.setCurrentIndex(index)

class SignalProcessingSettings(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        layout = QFormLayout()
        
        self.filter_type = QComboBox()
        self.filter_type.addItems(["Notch Filter", "Bandpass Filter", "Wavelet Transform"])
        
        self.filter_range_low = QDoubleSpinBox()
        self.filter_range_low.setRange(0.1, 100.0)
        self.filter_range_high = QDoubleSpinBox()
        self.filter_range_high.setRange(1.0, 200.0)
        
        range_layout = QHBoxLayout()
        range_layout.addWidget(self.filter_range_low)
        range_layout.addWidget(QLabel("to"))
        range_layout.addWidget(self.filter_range_high)
        
        self.artifact_handling = QCheckBox("Enable artifact handling")
        self.ocular_correction = QCheckBox("Ocular correction")
        
        layout.addRow(QLabel("Filter Type:"), self.filter_type)
        layout.addRow(QLabel("Frequency Range (Hz):"), range_layout)
        layout.addRow(self.artifact_handling)
        layout.addRow(self.ocular_correction)
        
        self.setLayout(layout)

class ModeSettings(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        self.mode = QComboBox()
        self.mode.addItems(["Training Mode", "Assessment Mode", "Simulation Mode"])
        self.mode.currentIndexChanged.connect(self.update_mode)
        
        self.simulation_options = QCheckBox("Generate synthetic data")
        self.simulation_options.hide()
        
        layout.addWidget(QLabel("Operation Mode:"))
        layout.addWidget(self.mode)
        layout.addWidget(self.simulation_options)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def update_mode(self, index):
        self.simulation_options.setVisible(index == 2)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Neurofeedback Configuration")
        self.setGeometry(100, 100, 600, 400)
        
        tab_widget = QTabWidget()
        
        # Create tabs
        tab_widget.addTab(GeneralSettings(), "General")
        tab_widget.addTab(NeurofeedbackSettings(), "Neurofeedback Training Type")
        tab_widget.addTab(SignalProcessingSettings(), "Signal Processing")
        tab_widget.addTab(ModeSettings(), "Mode")
        
        # Create buttons
        start_btn = QPushButton("Start")
        start_btn.clicked.connect(self.start_processing)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.close)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(start_btn)
        btn_layout.addWidget(cancel_btn)
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(tab_widget)
        main_layout.addLayout(btn_layout)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
    
    def start_processing(self):
        # Here you would collect all the parameters and start processing
        print("Starting processing with selected parameters...")
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
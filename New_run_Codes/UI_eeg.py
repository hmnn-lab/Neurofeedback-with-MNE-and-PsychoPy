import sys
import os
import re
import logging
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel,
                             QLineEdit, QPushButton, QMessageBox, QRadioButton, QButtonGroup,
                             QFileDialog, QProgressBar)
from PyQt6.QtCore import Qt
import mne
import numpy as np
from pyprep import NoisyChannels
from asrpy import ASR
from mne.preprocessing import ICA
from mne_icalabel import label_components
from recordnsave_eeg import record_eeg_stream
from fixation_display import run_fixation_display
from cal_baseline_psd import compute_psd_auc
import multiprocessing
from multiprocessing import Manager
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[logging.FileHandler('eeg_processing.log'), logging.StreamHandler()])

def preproc_flow(file_path, n_channels, montage_name='standard_1020', notch_freq=50, prob_threshold=0.9, max_chunk_size=1):
    try:
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose='warning')
        logging.debug(f"Loaded baseline file: {file_path} with {len(raw.ch_names)} channels")
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find baseline file: {file_path}")
    
    raw.crop(tmax=120)

    file_n_channels = len(raw.ch_names)
    if n_channels != file_n_channels:
        logging.warning(f"Input n_channels ({n_channels}) does not match file channels ({file_n_channels}). Using file channels.")
        n_channels = file_n_channels

    for ch in raw.ch_names:
        if raw.get_channel_types([ch])[0] not in ['eeg']:
            logging.debug(f"Setting channel {ch} to EEG type")
            raw.set_channel_types({ch: 'eeg'})

    channel_mapping = {
        'CZ': 'Cz', 'FP1': 'Fp1', 'FP2': 'Fp2', 'FPZ': 'Fpz',
        'FZ': 'Fz', 'PZ': 'Pz', 'OZ': 'Oz'
    }
    raw.rename_channels(channel_mapping)
    logging.debug(f"Renamed channels to: {raw.ch_names}")

    try:
        montage = mne.channels.make_standard_montage(montage_name)
        raw.set_montage(montage, on_missing='warn')
    except Exception as e:
        logging.error(f"Failed to set montage: {e}")
        raise ValueError("Could not set montage. Check channel names and montage compatibility.")

    montage = raw.get_montage()
    if montage:
        pos = montage.get_positions()['ch_pos']
        nan_channels = [ch for ch, coord in pos.items() if ch in raw.ch_names and np.any(np.isnan(coord))] 
        if nan_channels:
            logging.warning(f"NaN positions found for channels: {nan_channels}")
            print("Proceeding without RANSAC due to invalid channel positions.")
            bad_channels = []
        else:
            raw.filter(l_freq=1, h_freq=100, verbose=False)
            raw.notch_filter(freqs=notch_freq, verbose=False)
            raw.set_eeg_reference('average', projection=True, verbose=False)
            raw.apply_proj() 

            try:
                nd = NoisyChannels(raw, random_state=1337)
                nd.find_bad_by_ransac(channel_wise=True, max_chunk_size=max_chunk_size)
                bad_channels = nd.bad_by_ransac() or []
                raw.info['bads'].extend(bad_channels)
                logging.debug(f"Bad channels detected: {bad_channels}")
            except Exception as e:
                logging.warning(f"RANSAC failed: {e}. Proceeding without bad channel detection.")
                bad_channels = []
    else:
        logging.warning("No montage set. Skipping RANSAC.")
        bad_channels = []
        raw.filter(l_freq=1, h_freq=100, verbose=False)
        raw.notch_filter(freqs=notch_freq, verbose=False)
        raw.set_eeg_reference('average', projection=True, verbose=False)
        

    if bad_channels:
        raw.interpolate_bads(reset_bads=False)

    # Artifact Subspace Reconstruction (ASR)
    asr = ASR(sfreq=raw.info['sfreq'])
    asr.fit(raw)
    raw = asr.transform(raw)

    # ICA for artifact detection and removal - use extended infomax
    n_components = min(int(0.9 * n_channels), n_channels - len(bad_channels))
    if n_components < 1:
        logging.warning("No components available for ICA after bad channel removal. Setting n_components to 1.")
        n_components = 1
    ica = ICA(
        n_components=n_components, 
        method='infomax', 
        fit_params=dict(extended=True), 
        max_iter='auto', 
        random_state=42,
        verbose='warning'
    )
    logging.debug(f"Fitting ICA with {n_components} components")
    ica.fit(raw)

    labels = label_components(raw, ica, 'iclabel')
    component_labels = labels['labels']
    component_probs = labels['y_pred_proba']

    artifact_components = [i for i, (prob, label) in enumerate(zip(component_probs, component_labels))
                          if prob >= prob_threshold and label in ['muscle', 'eye']]
    logging.debug(f"Flagged artifact components: {artifact_components}")

    raw_cleaned = ica.apply(raw.copy(), exclude=artifact_components)
    return raw_cleaned, bad_channels, asr, ica, artifact_components, labels

def preprocess_realtime_stream(data, client_info, rename_dict, bad_channels, asr, ica, artifact_components, montage_name='standard_1020', notch_freq=50):
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array (n_channels x n_samples)")
    n_channels, n_samples = data.shape
    if n_channels != len(client_info['ch_names']):
        raise ValueError(f"Data channels ({n_channels}) do not match client_info channels ({len(client_info['ch_names'])})")
    if asr.sfreq != client_info['sfreq']:
        raise ValueError(f"ASR sampling rate ({asr.sfreq} Hz) does not match client_info ({client_info['sfreq']} Hz)")
    if ica.n_components > n_channels:
        raise ValueError(f"ICA components ({ica.n_components}) exceed number of channels ({n_channels})")

    try:
        raw_realtime = mne.io.RawArray(data, client_info, verbose='warning')
    except Exception as e:
        raise ValueError(f"Failed to create RawArray: {e}")

    try:
        raw_realtime.rename_channels(rename_dict)
        logging.debug(f"Renamed channels to: {raw_realtime.ch_names}")
    except Exception as e:
        raise ValueError(f"Channel renaming failed: {e}")

    try:
        montage = mne.channels.make_standard_montage(montage_name)
        raw_realtime.set_montage(montage, on_missing='warn')
    except Exception as e:
        logging.warning(f"Failed to set montage: {e}")

    try:
        raw_realtime.filter(l_freq=1, h_freq=100, verbose=False).notch_filter(freqs=notch_freq, verbose=False)
        raw_realtime.set_eeg_reference('average', projection=True, verbose=False)
        
    except Exception as e:
        raise RuntimeError(f"Filtering/referencing failed: {e}")

    if bad_channels:
        raw_realtime.info['bads'].extend(bad_channels)
        logging.debug(f"Interpolating bad channels: {bad_channels}")
        try:
            raw_realtime.interpolate_bads(reset_bads=False)
        except Exception as e:
            logging.warning(f"Bad channel interpolation failed: {e}")

    try:
        raw_realtime_asr = asr.transform(raw_realtime)
    except Exception as e:
        raise RuntimeError(f"ASR transformation failed: {e}")

    try:
        raw_realtime_processed = ica.apply(raw_realtime_asr.copy(), exclude=artifact_components)
    except Exception as e:
        raise RuntimeError(f"ICA application failed: {e}")

    return raw_realtime_processed

class ParadigmChoiceWindow(QMainWindow):
    def __init__(self, preprocessed_data=None, baseline_file=None):
        super().__init__()
        self.setWindowTitle("Paradigm Choice")
        self.setGeometry(100, 100, 400, 300)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        layout.addWidget(QLabel("Select Paradigm (To be defined):"))
        self.data_label = QLabel(f"Preprocessed data ready: {'Yes' if preprocessed_data is not None else 'No'}")
        layout.addWidget(self.data_label)
        self.file_label = QLabel(f"Baseline file: {baseline_file if baseline_file else 'None'}")
        layout.addWidget(self.file_label)

        self.status_label = QLabel("Ready to select paradigm.")
        layout.addWidget(self.status_label)

        layout.addStretch()

    def closeEvent(self, event):
        if event is not None:
            event.accept()

class PreprocessingWindow(QMainWindow):
    def __init__(self, baseline_file=None):
        super().__init__()
        self.setWindowTitle("EEG Preprocessing Parameters")
        self.setGeometry(100, 100, 400, 500)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        self.file_path_input = QLineEdit()
        self.file_path_input.setPlaceholderText("Select .fif file")
        if baseline_file:
            self.file_path_input.setText(baseline_file)
        layout.addWidget(QLabel("Path to baseline .fif file:"))
        layout.addWidget(self.file_path_input)
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_file)
        layout.addWidget(self.browse_button)

        self.stream_name_input = QLineEdit()
        self.stream_name_input.setPlaceholderText("e.g., Signal_generator")
        self.stream_name_input.setText("Signal_generator")
        layout.addWidget(QLabel("Stream name:"))
        layout.addWidget(self.stream_name_input)

        self.notch_freq_input = QLineEdit()
        self.notch_freq_input.setPlaceholderText("e.g., 50")
        self.notch_freq_input.setText("50")
        layout.addWidget(QLabel("Notch filter frequency (Hz):"))
        layout.addWidget(self.notch_freq_input)

        self.n_channels_input = QLineEdit()
        self.n_channels_input.setPlaceholderText("e.g., 8")
        self.n_channels_input.setText("8")
        layout.addWidget(QLabel("Number of EEG channels:"))
        layout.addWidget(self.n_channels_input)

        self.channel_names_input = QLineEdit()
        self.channel_names_input.setPlaceholderText("e.g., Fp1,Fp2,Cz,Pz,Fz,Oz,T7,T8")
        layout.addWidget(QLabel("Channel names (comma-separated):"))
        layout.addWidget(self.channel_names_input)

        self.prob_threshold_input = QLineEdit()
        self.prob_threshold_input.setPlaceholderText("e.g., 0.9")
        self.prob_threshold_input.setText("0.9")
        layout.addWidget(QLabel("Probability threshold for artifact components:"))
        layout.addWidget(self.prob_threshold_input)

        self.max_chunk_size_input = QLineEdit()
        self.max_chunk_size_input.setPlaceholderText("e.g., 1")
        self.max_chunk_size_input.setText("1")
        layout.addWidget(QLabel("Max chunk size for RANSAC (seconds):"))
        layout.addWidget(self.max_chunk_size_input)

        self.start_button = QPushButton("Start Preprocessing")
        self.start_button.clicked.connect(self.start_preprocessing)
        layout.addWidget(self.start_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(QLabel("Preprocessing Progress:"))
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel(f"Ready to preprocess baseline file: {baseline_file if baseline_file else 'None'}")
        layout.addWidget(self.status_label)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.open_paradigm_window)
        self.next_button.setEnabled(False)
        layout.addWidget(self.next_button)

        layout.addStretch()

        self.preprocessed_data = None
        self.baseline_file = baseline_file
        self.bad_channels = None
        self.asr = None
        self.ica = None
        self.artifact_components = None
        self.client_info = None
        self.rename_dict = None
        self.progress_value = 0

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Baseline .fif File", "", "FIF Files (*.fif)")
        if file_path:
            self.file_path_input.setText(file_path)
            self.baseline_file = file_path
            self.status_label.setText(f"Ready to preprocess baseline file: {file_path}")
            # Update channel count based on file
            try:
                raw = mne.io.read_raw_fif(file_path, preload=False, verbose='warning')
                self.n_channels_input.setText(str(len(raw.ch_names)))
            except Exception as e:
                logging.warning(f"Could not read channels from {file_path}: {e}")

    def validate_inputs(self):
        try:
            file_path = self.file_path_input.text().strip()
            if not file_path or not os.path.exists(file_path) or not file_path.endswith('.fif'):
                raise ValueError("Valid .fif file path required.")

            raw = mne.io.read_raw_fif(file_path, preload=False, verbose='warning')
            file_n_channels = len(raw.ch_names)

            stream_name = self.stream_name_input.text().strip()
            if not stream_name:
                raise ValueError("Stream name cannot be empty.")

            notch_freq = float(self.notch_freq_input.text().strip())
            if notch_freq <= 0:
                raise ValueError("Notch frequency must be positive.")

            n_channels = int(self.n_channels_input.text().strip())
            if n_channels <= 0:
                raise ValueError("Number of channels must be positive.")
            if n_channels != file_n_channels:
                QMessageBox.warning(self, "Channel Mismatch",
                                    f"Input channels ({n_channels}) do not match baseline file channels ({file_n_channels}). Using file channels.")
                n_channels = file_n_channels
                self.n_channels_input.setText(str(n_channels))

            channel_names = [ch.strip() for ch in self.channel_names_input.text().strip().split(',')]
            if not channel_names or len(channel_names) != n_channels:
                raise ValueError(f"Channel names must match number of channels ({n_channels}).")

            prob_threshold = float(self.prob_threshold_input.text().strip())
            if not 0 <= prob_threshold <= 1:
                raise ValueError("Probability threshold must be between 0 and 1.")

            max_chunk_size = float(self.max_chunk_size_input.text().strip())
            if max_chunk_size <= 0:
                raise ValueError("Max chunk size must be positive.")

            return stream_name, file_path, notch_freq, n_channels, channel_names, prob_threshold, max_chunk_size
        except ValueError as e:
            return str(e)
        except Exception as e:
            return f"Error reading baseline file: {e}"

    def update_progress(self, value, status_text):
        self.progress_value = value
        self.progress_bar.setValue(value)
        self.status_label.setText(status_text)
        QApplication.processEvents()

    def preprocess_task(self):
        try:
            self.update_progress(10, f"Loading and validating baseline file: {self.baseline_file or 'None'}...")
            result = self.validate_inputs()
            if isinstance(result, str):
                raise ValueError(result)
            stream_name, file_path, notch_freq, n_channels, channel_names, prob_threshold, max_chunk_size = result

            self.baseline_file = file_path
            self.update_progress(20, f"Preprocessing baseline file: {self.baseline_file}...")
            raw_cleaned, bad_channels, asr, ica, artifact_components, labels = preproc_flow(
                file_path, n_channels, notch_freq=notch_freq, prob_threshold=prob_threshold, max_chunk_size=max_chunk_size
            )

            self.client_info = mne.create_info(ch_names=channel_names, sfreq=raw_cleaned.info['sfreq'], ch_types='eeg')
            self.client_info.set_montage('standard_1020', on_missing='warn')
            self.rename_dict = {ch: ch for ch in channel_names}

            self.update_progress(80, "Simulating real-time stream preprocessing...")
            n_samples = int(raw_cleaned.info['sfreq'] * 10)
            data = raw_cleaned.get_data()[:, :n_samples]
            raw_realtime_processed = preprocess_realtime_stream(
                data, self.client_info, self.rename_dict, bad_channels, asr, ica, artifact_components,
                notch_freq=notch_freq
            )

            self.preprocessed_data = raw_realtime_processed
            self.bad_channels = bad_channels
            self.asr = asr
            self.ica = ica
            self.artifact_components = artifact_components

            self.update_progress(100, f"Preprocessing completed for {self.baseline_file}.")
            self.next_button.setEnabled(True)
            QMessageBox.information(self, "Success", f"Preprocessing completed for {self.baseline_file}.\nProceed to paradigm selection.")
        except Exception as e:
            self.update_progress(0, f"Preprocessing failed: {e}")
            QMessageBox.critical(self, "Preprocessing Error", f"Preprocessing failed: {e}")
            logging.error(f"Preprocessing error: {e}")
        finally:
            self.start_button.setEnabled(True)

    def start_preprocessing(self):
        self.start_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.status_label.setText(f"Starting preprocessing for {self.baseline_file or 'None'}...")
        self.progress_bar.setValue(0)
        QApplication.processEvents()

        self.preprocess_task()

    def open_paradigm_window(self):
        self.paradigm_window = ParadigmChoiceWindow(preprocessed_data=self.preprocessed_data, baseline_file=self.baseline_file)
        self.paradigm_window.show()
        self.close()

    def closeEvent(self, event):
        if event is not None:
            event.accept()

class EEGRecordingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Baseline Recording")
        self.setGeometry(100, 100, 400, 300)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        layout.addWidget(QLabel("Do you need to record a new baseline?"))
        self.radio_group = QButtonGroup()
        self.radio_yes = QRadioButton("Yes, record new baseline")
        self.radio_no = QRadioButton("No, use existing baseline")
        self.radio_group.addButton(self.radio_yes)
        self.radio_group.addButton(self.radio_no)
        self.radio_yes.setChecked(True)
        layout.addWidget(self.radio_yes)
        layout.addWidget(self.radio_no)

        self.stream_name_input = QLineEdit()
        self.stream_name_input.setPlaceholderText("e.g., Signal_generator")
        self.stream_name_input.setText("Signal_generator")
        layout.addWidget(QLabel("Input stream name:"))
        layout.addWidget(self.stream_name_input)

        self.duration_input = QLineEdit()
        self.duration_input.setPlaceholderText("e.g., 60")
        layout.addWidget(QLabel("Recording duration (seconds):"))
        layout.addWidget(self.duration_input)

        self.start_button = QPushButton("Start Recording")
        self.start_button.clicked.connect(self.start_recording)
        layout.addWidget(self.start_button)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.open_preprocessing_window)
        self.next_button.setEnabled(False)
        layout.addWidget(self.next_button)

        self.status_label = QLabel("Select whether to record a new baseline.")
        layout.addWidget(self.status_label)

        layout.addStretch()

        self.recording_process = None
        self.manager = None
        self.result_dict = None
        self.start_event = None
        self.baseline_file = None

        self.radio_yes.toggled.connect(self.toggle_inputs)
        self.toggle_inputs()

    def toggle_inputs(self):
        recording_needed = self.radio_yes.isChecked()
        self.stream_name_input.setEnabled(recording_needed)
        self.duration_input.setEnabled(recording_needed)
        self.start_button.setEnabled(recording_needed)
        self.next_button.setEnabled(not recording_needed)

    def generate_filename(self):
        # Generate incremental filename and return both path and number
        script_dir = os.path.dirname(os.path.abspath(__file__))
        baseline_folder = os.path.join(script_dir, "baseline_recordings")
        os.makedirs(baseline_folder, exist_ok=True)
        existing_files = os.listdir(baseline_folder)
        pattern = r'baseline_(\d{3})_raw\.fif'  # Updated pattern for _eeg.fif
        max_num = 0
        for fname in existing_files:
            match = re.match(pattern, fname)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)
        file_num = max_num + 1
        filename = os.path.join(baseline_folder, f"baseline_{file_num:03d}_eeg.fif")  # Use _eeg.fif
        return filename, f"{file_num:03d}"

    def validate_inputs(self, stream_name, duration):
        try:
            if not stream_name.strip():
                raise ValueError("Stream name cannot be empty.")
            duration = float(duration)
            if duration <= 0:
                raise ValueError("Duration must be positive.")
            return stream_name, duration
        except ValueError as e:
            return str(e)

    def fixation_task(self, duration, start_event):
        logging.debug("Starting fixation task")
        try:
            run_fixation_display(duration, start_event)
            logging.debug("Fixation task completed")
        except Exception as e:
            logging.error(f"Fixation task failed: {e}")
            raise

    @staticmethod
    def recording_task(stream_name, duration, filename, start_event, result_dict):
        logging.debug("Initializing recording task")
        try:
            import mne_lsl
            logging.debug("MNE-LSL imported successfully")
            if start_event is not None:
                logging.debug("Waiting for start event")
                start_event.wait()
            logging.debug(f"Connecting to LSL stream: {stream_name}")
            fif_path = record_eeg_stream(stream_name, duration, filename)
            if not os.path.exists(fif_path):
                raise FileNotFoundError(f"Failed to save .fif file at {fif_path}")
            logging.debug(f"record_eeg_stream saved: {fif_path}")
            result_dict["fif_path"] = fif_path
            logging.debug(f"Recording task completed: {fif_path}")
        except Exception as e:
            logging.error(f"Recording task failed: {e}")
            raise

    def start_recording(self):
        if not self.radio_yes.isChecked():
            self.status_label.setText("No recording needed. Proceed to Next.")
            self.next_button.setEnabled(True)
            return

        stream_name = self.stream_name_input.text()
        duration = self.duration_input.text()

        result = self.validate_inputs(stream_name, duration)
        if isinstance(result, str):
            QMessageBox.critical(self, "Input Error", f"Invalid input: {result}")
            return

        stream_name, duration = result
        try:
            filename, file_num = self.generate_filename()
            logging.debug(f"Generated filename: {filename}, file_num: {file_num}")
        except Exception as e:
            self.status_label.setText(f"Failed to generate filename: {e}")
            QMessageBox.critical(self, "File Error", f"Failed to generate filename: {e}")
            return

        self.start_button.setEnabled(False)
        self.status_label.setText(f"Recording to {filename}...")
        QApplication.processEvents()

        try:
            self.manager = Manager()
            self.result_dict = self.manager.dict()
            self.start_event = multiprocessing.Event()

            self.recording_process = multiprocessing.Process(
                target=EEGRecordingWindow.recording_task,
                args=(stream_name, duration, filename, self.start_event, self.result_dict)
            )

            logging.debug("Starting recording process")
            self.recording_process.start()

            logging.debug("Running fixation task in main process")
            self.fixation_task(duration, self.start_event)
            QApplication.processEvents()

            logging.debug("Waiting for recording process to complete")
            self.recording_process.join()
            QApplication.processEvents()

            if self.recording_process.exitcode != 0:
                logging.error(f"Recording process failed with exit code: {self.recording_process.exitcode}")
                raise RuntimeError(f"Recording process failed with exit code: {self.recording_process.exitcode}")

            self.status_label.setText("Fixation display and EEG recording completed.")
            QApplication.processEvents()

            fif_path = self.result_dict.get("fif_path")
            if fif_path and os.path.exists(fif_path):
                self.baseline_file = fif_path
                self.status_label.setText("Computing PSD and saving results...")
                QApplication.processEvents()
                try:
                    df, xlsx_path, plot_path = compute_psd_auc(fif_path, file_num)
                    if not os.path.exists(xlsx_path):
                        raise FileNotFoundError(f"PSD Excel file not saved at {xlsx_path}")
                    if not os.path.exists(plot_path):
                        raise FileNotFoundError(f"PSD plot not saved at {plot_path}")
                    logging.debug(f"Saved PSD Excel at {xlsx_path}, plot at {plot_path}")
                    self.status_label.setText(
                        f"Recording complete.\nFIF: {fif_path}\nPSD AUC: {xlsx_path}\nPlot: {plot_path}"
                    )
                    QApplication.processEvents()
                    QMessageBox.information(
                        self, "Success",
                        f"Baseline recorded successfully.\nFIF: {fif_path}\nPSD AUC: {xlsx_path}\nPlot: {plot_path}\nProceed to preprocessing."
                    )
                    self.next_button.setEnabled(True)
                except Exception as e:
                    self.status_label.setText(f"PSD computation failed: {e}")
                    QApplication.processEvents()
                    QMessageBox.critical(self, "Processing Error", f"PSD computation failed: {e}")
                    logging.error(f"PSD computation error: {e}")
            else:
                self.status_label.setText(f"EEG file not found at {fif_path}. PSD computation skipped.")
                QApplication.processEvents()
                QMessageBox.critical(self, "File Error", f"EEG file not found at {fif_path}. PSD computation skipped.")
                logging.error(f"EEG file not found at {fif_path}")

        except Exception as e:
            self.status_label.setText(f"Recording failed: {e}")
            QApplication.processEvents()
            QMessageBox.critical(self, "Recording Error", f"Recording failed: {e}")
            logging.error(f"Recording process error: {e}")

        finally:
            if self.recording_process and self.recording_process.is_alive():
                self.recording_process.terminate()
                self.recording_process.join()
            self.start_button.setEnabled(True)
            self.recording_process = None
            self.manager = None
            self.result_dict = None
            self.start_event = None
            logging.debug("Processes cleaned up")
            QApplication.processEvents()

    def open_preprocessing_window(self):
        self.preprocessing_window = PreprocessingWindow(baseline_file=self.baseline_file)
        self.preprocessing_window.show()
        self.close()

    def closeEvent(self, event):
        if self.recording_process and self.recording_process.is_alive():
            self.recording_process.terminate()
            self.recording_process.join()
        if event is not None:
            event.accept()

def main():
    multiprocessing.set_start_method('spawn', force=True)
    app = QApplication(sys.argv)
    window = EEGRecordingWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
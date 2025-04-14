# Import necessary libraries
import pandas as pd
import numpy as np
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
from mne_realtime import LSLClient
from pyprep.find_noisy_channels import NoisyChannels
from asrpy import ASR
import pylsl
import time
import os
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eeg_streaming.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------ Configuration ------------------
CONFIG = {
    'HOST': 'Signal_generator_EEG_16_250.0_float32_Vgram',
    'WAIT_MAX': 5,  # Seconds to wait for LSL connection
    'SFREQ_EXPECTED': 250,  # Expected sampling rate (Hz)
    'LOW_FREQ': 8,  # Alpha band lower bound (Hz)
    'HIGH_FREQ': 12,  # Alpha band upper bound (Hz)
    'TIME_WINDOW': 5,  # Time window for PSD (seconds)
    'UPDATE_INTERVAL': 0.5,  # Feedback update interval (seconds)
    'FEED_CH_NAMES': ['O1'],  # Channels for feedback
    'MIN_FEEDBACK': 2,  # Min feedback value (for Unity)
    'MAX_FEEDBACK': 10,  # Max feedback value (for Unity)
    'BASELINE_PATH': r"C:\Users\varsh\OneDrive\Desktop\NFB-MNE-Psy\base.fif",
    'OUTPUT_EXCEL': r"C:\Users\varsh\OneDrive\Desktop\NFB-MNE-Psy\feedback_values.xlsx",
    'OUTPUT_PLOT': r"C:\Users\varsh\OneDrive\Desktop\NFB-MNE-Psy\power_change_plot.png",
    'RENAME_DICT': {
        'EEG 001': 'Pz', 'EEG 002': 'T8', 'EEG 003': 'P3', 'EEG 004': 'C3',
        'EEG 005': 'Fp1', 'EEG 006': 'Fp2', 'EEG 007': 'O1', 'EEG 008': 'P4',
        'EEG 009': 'Fz', 'EEG 010': 'F7', 'EEG 011': 'C4', 'EEG 012': 'O2',
        'EEG 013': 'F3', 'EEG 014': 'F4', 'EEG 015': 'Cz', 'EEG 016': 'T3',
    }
}

# ------------------ Baseline Preprocessing ------------------
def preprocess_baseline():
    """Load and preprocess baseline EEG data."""
    try:
        logger.info("Loading baseline data from %s", CONFIG['BASELINE_PATH'])
        raw = mne.io.read_raw_fif(CONFIG['BASELINE_PATH'], preload=True, verbose=False)
        
        # Apply filters
        logger.info("Applying notch (50 Hz) and bandpass (0.1–40 Hz) filters")
        raw.notch_filter(50, picks='eeg', verbose=False)
        raw.filter(l_freq=0.1, h_freq=40, picks='eeg', verbose=False)
        
        # Set average reference
        raw.set_eeg_reference('average', verbose=False)
        
        # Rename channels
        logger.info("Renaming channels")
        raw.rename_channels(CONFIG['RENAME_DICT'])
        
        # Set montage
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='warn')
        
        # Detect bad channels
        logger.info("Detecting bad channels with pyprep")
        nd = NoisyChannels(raw, random_state=1337)
        nd.find_bad_by_ransac(channel_wise=True, max_chunk_size=1)
        raw.info['bads'].extend(nd.bad_by_ransac)
        logger.info("Bad channels: %s", raw.info['bads'])
        
        # Apply ASR
        logger.info("Fitting ASR")
        asr = ASR(sfreq=raw.info['sfreq'])
        asr.fit(raw)
        
        # Apply ICA
        n_channels = len(raw.ch_names)
        n_components = min(n_channels - len(raw.info['bads']), 15)  # Limit components
        logger.info("Fitting ICA with %d components", n_components)
        ica = ICA(n_components=n_components, method='infomax', max_iter=500, random_state=42)
        ica.fit(raw, verbose=False)
        
        # Label and exclude artifacts
        logger.info("Labeling ICA components")
        labels = label_components(raw, ica, 'iclabel')
        artifact_components = [
            i for i, (prob, label) in enumerate(zip(labels['y_pred_proba'], labels['labels']))
            if prob >= 0.9 and label in ['muscle', 'eye']
        ]
        logger.info("Artifact components: %s", artifact_components)
        
        # Apply ICA
        raw_cleaned = ica.apply(raw.copy(), exclude=artifact_components, verbose=False)
        
        return raw_cleaned, asr, ica, artifact_components, raw.info['bads']
    
    except Exception as e:
        logger.error("Baseline preprocessing failed: %s", e)
        raise

# ------------------ Real-Time Processing ------------------
def main():
    """Stream EEG data, compute alpha power, and send feedback via LSL."""
    # Preprocess baseline
    logger.info("Starting baseline preprocessing")
    _, asr, ica, artifact_components, bad_channels = preprocess_baseline()
    
    # Initialize variables
    epoch_count = 0
    running = True
    time_points = []
    pow_change_values = []
    feedback_values = []
    
    # Create LSL outlet for Unity
    logger.info("Creating LSL outlet: FeedbackStream")
    outlet_info = pylsl.StreamInfo(
        name='FeedbackStream',
        type='Feedback',
        channel_count=1,
        nominal_srate=1/CONFIG['UPDATE_INTERVAL'],
        channel_format='float32',
        source_id='FeedbackGenerator'
    )
    outlet = pylsl.StreamOutlet(outlet_info)
    
    # Initialize timer
    update_timer = time.time()
    
    # Start LSL client
    logger.info("Connecting to EEG stream: %s", CONFIG['HOST'])
    try:
        with LSLClient(host=CONFIG['HOST'], wait_max=CONFIG['WAIT_MAX']) as client:
            client_info = client.get_measurement_info()
            sfreq = client_info['sfreq']
            stream_ch_names = [ch['ch_name'] for ch in client_info['chs']]
            
            # Validate channels
            expected_ch_names = list(CONFIG['RENAME_DICT'].values())
            if stream_ch_names != expected_ch_names:
                logger.warning(
                    "Stream channels %s differ from expected %s",
                    stream_ch_names, expected_ch_names
                )
                # Update rename dict for stream
                stream_rename_dict = {
                    stream_ch_names[i]: expected_ch_names[i]
                    for i in range(min(len(stream_ch_names), len(expected_ch_names)))
                }
            else:
                stream_rename_dict = CONFIG['RENAME_DICT']
            
            # Check sampling rate
            if abs(sfreq - CONFIG['SFREQ_EXPECTED']) > 1:
                logger.warning(
                    "Stream sfreq (%.2f Hz) differs from expected (%.2f Hz)",
                    sfreq, CONFIG['SFREQ_EXPECTED']
                )
            
            while running:
                try:
                    # Check for exit
                    if mne.utils.check_escape():
                        logger.info("Escape key detected")
                        running = False
                        break
                    
                    # Process new data
                    if time.time() - update_timer >= CONFIG['UPDATE_INTERVAL']:
                        logger.info("Processing epoch %d", epoch_count)
                        
                        # Get epoch
                        epoch = client.get_data_as_epoch(n_samples=int(sfreq))
                        epoch.apply_baseline(baseline=(0, None), verbose=False)
                        data = np.squeeze(epoch.get_data())
                        
                        # Create Raw object
                        raw_realtime = mne.io.RawArray(data, client_info, verbose=False)
                        raw_realtime.rename_channels(stream_rename_dict)
                        
                        # Apply preprocessing
                        raw_realtime.info['bads'] = bad_channels
                        raw_realtime_asr = asr.transform(raw_realtime)
                        raw_realtime_clean = ica.apply(
                            raw_realtime_asr, exclude=artifact_components, verbose=False
                        )
                        
                        # Compute PSD
                        psd = raw_realtime_clean.compute_psd(
                            tmin=0, tmax=None, picks=CONFIG['FEED_CH_NAMES'],
                            method='welch', average=False, verbose=False
                        )
                        freqs = psd.freqs
                        power = np.squeeze(psd.get_data())
                        power_whole = power.sum()
                        
                        # Calculate alpha power
                        low_freq_ind = np.argmin(np.abs(freqs - CONFIG['LOW_FREQ']))
                        high_freq_ind = np.argmin(np.abs(freqs - CONFIG['HIGH_FREQ']))
                        power_range = power[low_freq_ind:high_freq_ind].sum()
                        power_change = (power_range / power_whole) * 100
                        
                        # Scale to feedback
                        feedback_value = np.interp(
                            power_change,
                            [0, 100],
                            [CONFIG['MIN_FEEDBACK'], CONFIG['MAX_FEEDBACK']]
                        )
                        
                        # Store data
                        time_points.append(epoch_count)
                        pow_change_values.append(power_change)
                        feedback_values.append(feedback_value)
                        
                        # Send feedback to Unity
                        outlet.push_sample([feedback_value])
                        logger.info("Feedback value: %.2f", feedback_value)
                        
                        # Update state
                        update_timer = time.time()
                        epoch_count += 1
                    
                except Exception as e:
                    logger.error("Epoch %d failed: %s", epoch_count, e)
                    continue
                
    except Exception as e:
        logger.error("Streaming failed: %s", e)
        running = False
    
    finally:
        # Save results
        if feedback_values:
            logger.info("Saving data to %s", CONFIG['OUTPUT_EXCEL'])
            feedback_df = pd.DataFrame({
                'Epoch': time_points,
                'Power_Change (%)': pow_change_values,
                'Feedback_Value': feedback_values
            })
            os.makedirs(os.path.dirname(CONFIG['OUTPUT_EXCEL']), exist_ok=True)
            feedback_df.to_excel(CONFIG['OUTPUT_EXCEL'], index=False)
            
            # Plot power change
            plt.figure(figsize=(10, 4))
            plt.plot(time_points, pow_change_values, label='Alpha Power Change (%)')
            plt.xlabel('Epoch')
            plt.ylabel('Power Change (%)')
            plt.title('Real-Time Alpha Power Change')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            os.makedirs(os.path.dirname(CONFIG['OUTPUT_PLOT']), exist_ok=True)
            plt.savefig(CONFIG['OUTPUT_PLOT'])
            plt.close()
            logger.info("Saved plot to %s", CONFIG['OUTPUT_PLOT'])
        else:
            logger.warning("No data to save")
        
        logger.info("Streaming complete")

# ------------------ Run Script ------------------
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Terminated by user")
    except Exception as e:
        logger.error("Unexpected error: %s", e)
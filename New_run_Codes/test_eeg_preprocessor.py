import numpy as np
import mne
import sys
import os
from eeg_pproc import EEGPreproc

# Debug: Print environment information
print("Python version:", sys.version)
print("Python path:", sys.path)
print("Current working directory:", os.getcwd())
print("Checking for eeg_pproc.py:", os.path.exists('eeg_pproc.py'))

def load_baseline_data(file_path, preload=True):
    """
    Load baseline EEG data from a file and set a montage.
    
    Parameters:
    -----------
    file_path : str
        Path to the EEG data file (e.g., .fif).
    preload : bool
        Whether to preload the data into memory.
        
    Returns:
    --------
    raw : mne.io.Raw
        Loaded EEG data with montage set.
    """
    try:
        if file_path.endswith('.fif'):
            raw = mne.io.read_raw_fif(file_path, preload=preload)
        elif file_path.endswith('.edf'):
            raw = mne.io.read_raw_edf(file_path, preload=preload)
        elif file_path.endswith('.gdf'):
            raw = mne.io.read_raw_gdf(file_path, preload=preload)
        else:
            raise ValueError("Unsupported file format. Supported formats: .fif, .edf, .gdf")
        
        print(f"Loaded baseline data: {len(raw.ch_names)} channels, {raw.n_times} samples, {raw.info['sfreq']} Hz")
        print(f"Channel names: {raw.ch_names}")
        
        # Set montage (e.g., standard_1020) if not already set
        if raw.info.get('dig') is None or not any(ch['loc'][:3].any() for ch in raw.info['chs']):
            print("No valid montage found. Setting standard_1020 montage.")
            try:
                raw.set_montage('standard_1020', match_case=False, on_missing='warn')
            except Exception as e:
                print(f"Failed to set montage: {e}")
                print("Ensure channel names match standard_1020 or provide a custom montage.")
                raise
        else:
            print("Montage already set or channel positions available.")
        
        return raw
    except Exception as e:
        print(f"Failed to load baseline data: {e}")
        raise

def test_eeg_preprocessor(baseline_file_path, n_samples_realtime=250):
    """
    Test the EEGPreprocessor class using a baseline EEG recording.
    
    Parameters:
    -----------
    baseline_file_path : str
        Path to the baseline EEG file.
    n_samples_realtime : int
        Number of samples for real-time simulation (default: 250, ~1s at 250 Hz).
    """
    # Load baseline data
    offline_raw = load_baseline_data(baseline_file_path)
    sfreq = offline_raw.info['sfreq']
    ch_names = offline_raw.ch_names
    n_channels = len(ch_names)
    
    # Define a rename dictionary (identity mapping or customize as needed)
    rename_dict = {ch: ch for ch in ch_names}  # Adjust if channel names need remapping
    # Example: rename_dict = {'Ch1': 'F3', 'Ch2': 'F4', ...} if names don't match standard_1020
    
    # Initialize EEGPreprocessor
    print("\n=== Testing EEGPreprocessor Initialization ===")
    pre_processor = EEGPreproc(sfreq=sfreq, n_channels=n_channels, random_state=42)
    print("EEGPreprocessor initialized successfully")

    # Test preprocess_offline
    print("\n=== Testing Offline Preprocessing ===")
    try:
        raw_cleaned, bad_channels, asr, ica, artifact_components = pre_processor.preproc_offline(
            offline_raw, rename_dict=rename_dict
        )
        print("Offline preprocessing successful")
        print(f"Bad channels detected: {bad_channels}")
        print(f"Artifact components: {artifact_components}")
        print(f"Cleaned data shape: {raw_cleaned.get_data().shape}")
    except Exception as e:
        print(f"Offline preprocessing failed: {e}")
        raise

    # Simulate real-time data by extracting a small chunk from baseline
    print("\n=== Testing Real-time Preprocessing ===")
    try:
        # Ensure enough samples are available
        if n_samples_realtime > offline_raw.n_times:
            raise ValueError(f"Requested {n_samples_realtime} samples, but only {offline_raw.n_times} available.")
        realtime_data = offline_raw.get_data(start=0, stop=n_samples_realtime)
        client_info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        
        raw_realtime_processed = pre_processor.preproc_realtime(
            data=realtime_data,
            client_info=client_info,
            rename_dict=rename_dict
        )
        print("Real-time preprocessing successful")
        print(f"Processed real-time data shape: {raw_realtime_processed.get_data().shape}")
    except Exception as e:
        print(f"Real-time preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    # Specify the path to your baseline EEG file
    baseline_file_path = r'C:\Users\varsh\NFB_Spyder\New_Runs\Baseline_codes\baseline_recordings\baseline_002.fif'  # Replace with actual path
    
    try:
        test_eeg_preprocessor(baseline_file_path=baseline_file_path, n_samples_realtime=250)
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"Test failed: {e}")
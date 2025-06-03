import mne
import numpy as np
import pandas as pd
import os
from mne_lsl.stream import StreamLSL, EpochsStream
from psychopy import core

# Import your functions (assumed to be in separate files)
from eeg_parameters import get_eeg_parameters
from load_and_filter import load_and_filter_base
from rename_chan import rename_eeg_channels
from preproc_initialize import preproc_flow
from preproc_apply import preprocess_realtime_stream
#from visual_1freq import visualize_eeg_feedback
from power_auc_cal import compute_band_auc_epochs

def main():
    """
    Main function to process offline EEG data and real-time LSL stream for neurofeedback.
    """
    # Step 1: Get EEG parameters
    result = get_eeg_parameters()
    if not result:
        print("Failed to retrieve EEG parameters. Exiting.")
        return
    step, time_window, n_channels, feed_ch_names, band_name, low_freq, high_freq = result

    # Step 2: Load and preprocess offline baseline data
    raw = load_and_filter_base()

    # Step 3: Rename channels
    raw, n_channels = rename_eeg_channels(raw)

    # Step 4: Apply advanced preprocessing (bad channels, ASR, ICA)
    raw_cleaned, bad_channels, asr, ica, artifact_components = preproc_flow(raw, n_channels)

    # Step 5: Set up real-time LSL stream
    try:
        # Connect to EEG stream
        #eeg_stream = Stream(name="MyEEGStream", source_id="my_source_id")
        
        # Create EpochsStream
        epochs_stream = EpochsStream(
            stream="Signal_generator",
            bufsize=time_window,
            event_id=None,  # Use all events if not specified
            event_channels=None,  # Adjust based on your stream
            tmin=0.0,
            tmax=1.0,
            baseline=None,
            picks=feed_ch_names,
            reject= None,
            detrend=1  # Linear detrending
        )
        
        # Start acquisition
        epochs_stream.connect()

        # Step 6: Real-time processing loop
        auc_values = []  # Initialize auc_values
        epoch_count = 1  # Initialize epoch count
        print("Starting real-time processing...")
        
        while True:
            # Get next epoch
            epoch = epochs_stream.next()
            if epoch is not None:
                epoch_count += 1
                
                # Apply real-time preprocessing
                data = epoch.get_data(picks=feed_ch_names)[0]  # Single epoch, shape: n_channels x n_times
                client_info = mne.create_info(
                    ch_names=feed_ch_names,
                    sfreq=raw.info['sfreq'],
                    ch_types='eeg'
                )
                raw_realtime_processed = preprocess_realtime_stream(
                    data, client_info, rename_eeg_channels.__defaults__[0],  # Default rename_dict
                    bad_channels, asr, ica, artifact_components
                )
                
                # Convert processed raw back to epochs for compute_band_auc_epochs
                #events = np.array([[0, 0, 1]])  # Dummy event for single epoch
                processed_epoch = mne.Epochs(
                    raw_realtime_processed, events=None, event_id=None,
                    tmin=0, tmax=1.0, baseline=(None, 0), preload=True
                )
                
                # Compute AUC for the band
                power_change, output_path = compute_band_auc_epochs(
                    processed_epoch, feed_ch_names, epoch_count, auc_values,
                    band_name=band_name, low_freq=low_freq, high_freq=high_freq
                )
                
                # Visualize feedback
                #visualize_eeg_feedback()
            
            # Avoid busy-waiting
            core.wait(0.1)
            
    except KeyboardInterrupt:
        print("Stopping real-time stream...")
    except ValueError as e:
        print(f"Error in stream setup: {e}")
    finally:
        # Clean up
        epochs_stream.disconnect()
        core.quit()

if __name__ == "__main__":
    main()
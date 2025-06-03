import numpy as np
import pandas as pd
import time
import os
from mne_lsl.stream import StreamLSL
from mne_lsl.lsl import StreamInfo, StreamOutlet
from mne import create_info, Epochs, io
from preproc_initialize_copy import preproc_flow
from preproc_apply_copy import preprocess_realtime_stream
from power_auc_cal import compute_band_auc_epochs

def stream_power_change(
    input_stream_name='Signal_generator',
    band_name='Alpha',
    low_freq=8.0,
    high_freq=12.0,
    epoch_duration=1.0,
    output_stream_name='BandPowerChange'
):
    """Stream EEG data, preprocess it, and calculate power changes in a specified band."""
    # Step 1: Run offline preprocessing
    offline_file_path = r'C:\Users\varsh\NFB_Spyder\New_Runs\Baseline_codes\baseline_recordings\baseline_002.fif'
    n_channels = 16
    try:
        # FIX 1: Only unpack 5 values if using original function, or 6 if using modified function
        # Option A: If using original function (returns 5 values)
        raw_cleaned, bad_channels, asr, ica, artifact_components, labels = preproc_flow(
        offline_file_path, n_channels, montage_name='standard_1020'
        )

        
        # FIX 2: Create rename_dict from the preprocessing function or define it here
        # The rename_dict should map from your actual channel names to standard montage names
        # You need to get this from your data or define it based on your setup
        rename_dict = {
            'CZ': 'Cz', 'FP1': 'Fp1', 'FP2': 'Fp2', 'FPZ': 'Fpz',
            'FZ': 'Fz', 'PZ': 'Pz', 'OZ': 'Oz'
            # Add more mappings as needed for your 16 channels
        }
        
        print("Offline preprocessing completed!")
        print(f"Bad channels detected: {bad_channels}")
        print(f"Artifact components: {artifact_components}")
        
    except Exception as e:
        print(f"Offline preprocessing failed: {e}")
        return

    # Step 2: Connect to LSL stream
    try:
        stream = StreamLSL(bufsize=epoch_duration * 2, name=input_stream_name)
        stream.connect()
        info = stream.info
        ch_names = info['ch_names']
        sfreq = info['sfreq']
        print(f"Connected to stream: {input_stream_name}, sfreq: {sfreq}")
        print(f"Available channels: {ch_names}")
    except Exception as e:
        print(f"Failed to connect to stream: {e}")
        return

    # FIX 3: Check if selected channel exists AFTER renaming
    selected_channel = 'F3'
    # First check if the channel exists in original names or renamed names
    original_selected = None
    for orig, renamed in rename_dict.items():
        if renamed == selected_channel and orig in ch_names:
            original_selected = orig
            break
    
    if selected_channel not in ch_names and original_selected is None:
        print(f"Invalid channel {selected_channel}. Available: {ch_names}")
        print(f"After renaming would be: {[rename_dict.get(ch, ch) for ch in ch_names]}")
        return

    # Step 3: Create LSL output stream
    power_info = StreamInfo(
        name=output_stream_name, stype='PowerChange', n_channels=1,
        sfreq=1.0 / epoch_duration, dtype='float32', source_id='power_change_stream'
    )
    power_outlet = StreamOutlet(power_info)
    print(f"Streaming {band_name} power changes [{low_freq}-{high_freq} Hz] for {selected_channel}...")

    # Step 4: Initialize buffers and results
    buffer = []
    timestamps = []
    epoch_samples = int(epoch_duration * sfreq)
    epoch_count = 0
    auc_values = []
    results_df = pd.DataFrame(columns=['Epoch', 'AUC_Value', 'Power_Change'])
    target_sfreq = 250  # Matches preproc_flow's resampling rate

    try:
        while True:
            data, ts = stream.get_data()
            if data.size == 0:
                time.sleep(0.01)
                continue

            buffer.append(data)
            timestamps.extend(ts)

            buffered_data = np.hstack(buffer)
            if buffered_data.shape[1] >= epoch_samples:
                epoch_data = buffered_data[:, :epoch_samples]
                
                # FIX 4: Handle buffer remainder properly
                if buffered_data.shape[1] > epoch_samples:
                    buffer = [buffered_data[:, epoch_samples:]]
                    timestamps = timestamps[epoch_samples:]
                else:
                    buffer = []
                    timestamps = []

                # FIX 5: Resample to target_sfreq (250 Hz) if needed
                if sfreq != target_sfreq:
                    try:
                        raw_temp = io.RawArray(epoch_data, create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg'))
                        raw_temp.resample(target_sfreq, verbose=False)
                        epoch_data = raw_temp.get_data()
                        current_sfreq = target_sfreq
                    except Exception as e:
                        print(f"Resampling failed: {e}")
                        continue
                else:
                    current_sfreq = sfreq

                # FIX 6: Create proper client_info with correct channel names
                try:
                    client_info = create_info(ch_names=ch_names, sfreq=current_sfreq, ch_types='eeg')
                    raw_processed = preprocess_realtime_stream(
                        epoch_data, client_info, rename_dict, bad_channels, asr, ica, artifact_components,
                        montage_name='standard_1020'
                    )
                except Exception as e:
                    print(f"Real-time preprocessing failed: {e}")
                    continue

                # FIX 7: Handle channel selection after renaming
                try:
                    if selected_channel in raw_processed.ch_names:
                        single_channel_data = raw_processed.get_data(picks=[selected_channel])
                    else:
                        print(f"Channel {selected_channel} not found after preprocessing. Available: {raw_processed.ch_names}")
                        continue
                        
                    info_single = create_info(ch_names=[selected_channel], sfreq=current_sfreq, ch_types='eeg')
                    raw_single = io.RawArray(single_channel_data, info_single)
                    raw_single.set_montage('standard_1020', match_case=False, on_missing='warn')
                except Exception as e:
                    print(f"Channel selection failed: {e}")
                    continue

                # FIX 8: Create epochs properly
                try:
                    events = np.array([[0, 0, 1]])
                    epochs = Epochs(
                        raw_single, events, event_id=1, tmin=0, 
                        tmax=epoch_duration - 1 / current_sfreq,
                        baseline=None, preload=True, verbose=False
                    )
                    epoch = epochs[0]
                except Exception as e:
                    print(f"Epoch creation failed: {e}")
                    continue

                # FIX 9: Calculate power change
                try:
                    power_change, auc_value = compute_band_auc_epochs(
                        epoch, ch_names=[selected_channel], epoch_count=epoch_count,
                        auc_values=auc_values, output_path="./psd_results/Band_power_change1.xlsx",
                        band_name=band_name, low_freq=low_freq, high_freq=high_freq
                    )
                    print(f"Epoch {epoch_count}: Power Change = {power_change:.2f}% (Channel: {selected_channel})")

                    # Stream the result
                    power_outlet.push_sample(np.array([power_change], dtype=np.float32))
                    
                    # Save to DataFrame
                    new_row = pd.DataFrame([{
                        'Epoch': epoch_count,
                        'AUC_Value': auc_value,
                        'Power_Change': power_change
                    }])
                    results_df = pd.concat([results_df, new_row], ignore_index=True)

                    epoch_count += 1
                    
                except Exception as e:
                    print(f"Power calculation failed: {e}")
                    continue

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping streaming and saving results...")
    except Exception as e:
        print(f"Streaming error: {e}")
    finally:
        # Save results
        try:
            os.makedirs("./psd_results", exist_ok=True)
            results_df.to_excel("./psd_results/band_power_change1.xlsx", index=False)
            print("Results saved to ./psd_results/band_power_change1.xlsx")
        except Exception as e:
            print(f"Failed to save results: {e}")
        
        # Disconnect stream
        try:
            stream.disconnect()
            print("Stream disconnected.")
        except:
            pass

if __name__ == "__main__":
    stream_power_change(
        input_stream_name="Signal_generator",
        band_name="Alpha",
        low_freq=8,
        high_freq=12,
        epoch_duration=1.0
    )
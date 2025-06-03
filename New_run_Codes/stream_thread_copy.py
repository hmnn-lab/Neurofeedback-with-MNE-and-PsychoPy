import numpy as np
import pandas as pd
import time
import os
import re
from PyQt6.QtCore import QThread, pyqtSignal
from mne_lsl.stream import StreamLSL
from mne_lsl.lsl import StreamInfo, StreamOutlet
import mne
from power_auc_cal import compute_band_auc_epochs

class StreamThreadCopy(QThread):
    error_signal = pyqtSignal(str)
    power_update_signal = pyqtSignal(float, int)  # Signal for power change and epoch count

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        try:
            stream_power_change(
                **self.params,
                power_update_signal=self.power_update_signal,
                running_flag=lambda: self.running
            )
        except Exception as e:
            self.error_signal.emit(str(e))

def stream_power_change(
    input_stream_name=None,
    band_name='Alpha',
    low_freq=8.0,
    high_freq=12.0,
    epoch_duration=1.0,
    output_stream_name='BandPowerChange',
    selected_channel=None,
    feed_ch_names=None,
    power_update_signal=None,
    running_flag=None,
    montage_name='standard_1020'
):
    # Connect to LSL stream
    stream = StreamLSL(bufsize=epoch_duration * 2, name=input_stream_name)
    stream.connect()
    info = stream.info
    ch_names = info['ch_names']
    sfreq = info['sfreq']
    
    # Validate parameters
    if feed_ch_names is None or not feed_ch_names:
        raise ValueError("At least one channel name must be provided.")
    if selected_channel not in feed_ch_names:
        raise ValueError(f"Selected channel {selected_channel} not in provided channel names: {feed_ch_names}")
    if selected_channel not in ch_names:
        raise ValueError(f"Selected channel {selected_channel} not in stream channels: {ch_names}")
    
    print(f"Connected to stream: {input_stream_name}, sfreq: {sfreq}, selected channel: {selected_channel}")

    # Create client_info for the selected channel
    client_info = mne.create_info(ch_names=[selected_channel], sfreq=sfreq, ch_types='eeg')

    # Create LSL output stream for power change values
    power_info = StreamInfo(
        name=output_stream_name,
        stype='PowerChange',
        n_channels=1,
        sfreq=1.0 / epoch_duration,
        dtype='float32',
        source_id='power_change_stream'
    )
    power_outlet = StreamOutlet(power_info)
    print(f"Streaming power change values in '{band_name}' band [{low_freq}-{high_freq} Hz] for channel {selected_channel}...")

    # Determine next file number for results
    os.makedirs("./psd_results", exist_ok=True)
    existing_files = os.listdir("./psd_results")
    pattern = r'band_power_change_(\d{3})\.xlsx'
    max_num = 0
    for fname in existing_files:
        match = re.match(pattern, fname)
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)
    output_file = f"./psd_results/band_power_change_{max_num + 1:03d}.xlsx"

    buffer = []
    timestamps = []
    epoch_samples = int(epoch_duration * sfreq)
    epoch_count = 0
    auc_values = []
    
    # Prepare DataFrame for saving results
    results_df = pd.DataFrame(columns=['Epoch', 'AUC_Value', 'Power_Change'])

    try:
        while running_flag() if running_flag else True:
            data, ts = stream.get_data()
            if data.size == 0:
                time.sleep(0.01)
                continue

            buffer.append(data)
            timestamps.extend(ts)

            # Process buffered data when enough samples are collected
            buffered_data = np.hstack(buffer)
            if buffered_data.shape[1] >= epoch_samples:
                epoch_data = buffered_data[:, :epoch_samples]
                buffer = [buffered_data[:, epoch_samples:]]
                timestamps = timestamps[epoch_samples:]

                # Select data for the specified channel
                ch_index = ch_names.index(selected_channel)
                selected_data = epoch_data[ch_index:ch_index+1, :]  # Keep as 2D array

                # Create raw object for the selected channel
                raw_single = mne.io.RawArray(selected_data, client_info)

                # Create epoch for power computation
                events = np.array([[0, 0, 1]])
                epochs = mne.Epochs(
                    raw_single, events, event_id=1, tmin=0, tmax=epoch_duration - 1 / sfreq,
                    baseline=None, preload=True
                )
                epoch = epochs[0]

                # Compute power change
                power_change, auc_value = compute_band_auc_epochs(
                    epoch, ch_names=[selected_channel], epoch_count=epoch_count, 
                    auc_values=auc_values, output_path=output_file,
                    band_name=band_name, low_freq=low_freq, high_freq=high_freq
                )
                print(f"Epoch {epoch_count}: Power Change = {power_change:.2f}% (Channel: {selected_channel})")

                # Stream power change
                power_outlet.push_sample([float(power_change)])
                
                # Emit signal for UI
                if power_update_signal:
                    power_update_signal.emit(power_change, epoch_count)
                
                # Store results
                new_row = pd.DataFrame([{
                    'Epoch': epoch_count,
                    'AUC_Value': auc_value,
                    'Power_Change': power_change
                }])
                results_df = pd.concat([results_df, new_row], ignore_index=True)
                
                epoch_count += 1

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping streaming and saving results...")
    finally:
        # Save results to Excel
        os.makedirs("./psd_results", exist_ok=True)
        results_df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
        stream.disconnect()
        print("Stream disconnected.")
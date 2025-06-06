# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 16:21:28 2024

@author: varsh
"""
# Import the necessary libraries
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import pandas as pd  # For saving data to Excel

import mne_realtime
from mne_realtime import LSLClient, MockLSLStream

import pylsl
import pyxdf
import mne
import keyboard  # For exiting with 'Esc' key

mne.viz.set_browser_backend('qt')

# this is the host id that identifies your stream on LSL
host = 'HMNN-LSL'
wait_max = 5  # Max wait time for client connection

# Load EEG file
xdf_fpath = "C:/Users/varsh/.spyder-py3/eye-open-mc.xdf"
streams, header = pyxdf.load_xdf(xdf_fpath)
eeg_stream = streams[0]

# User-defined parameters
step = 0.1  # in seconds
time_window = 1  # in seconds
n_channels = 2
ch_names = ['Ch1', 'Ch2']
low_freq = 4  # Theta band (4-7 Hz)
high_freq = 7

data = streams[0]["time_series"].T
data = data[:n_channels]

sfreq = float(streams[0]["info"]["nominal_srate"][0])
info = mne.create_info(ch_names, sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(data, info)
raw.apply_function(lambda x: x * 1e-6, picks="eeg")

raw.notch_filter(50, picks='eeg').filter(l_freq=0.1, h_freq=40)

# Initialize storage lists
running = True
epoch_count = 1
time_points = []
pow_change_values = []

_, ax = plt.subplots(1)

# Output file path
output_path = r"C:\Users\varsh\OneDrive\Desktop\NFB-MNE-Psy\Rel_power_data.xlsx"

try:
    if __name__ == '__main__':
        with MockLSLStream(host, raw, 'eeg', step):
            with LSLClient(info=raw.info, host=host, wait_max=wait_max) as client:
                client_info = client.get_measurement_info()
                sfreq = int(client_info['sfreq'])

                while running:
                    if keyboard.is_pressed('esc'):  # Press ESC to stop
                        print("Exit requested by user (ESC key).")
                        break

                    print(f'Got epoch {epoch_count}')
                    epoch = client.get_data_as_epoch(n_samples=int(time_window * sfreq))

                    # Clear the previous plot
                    ax.clear()
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Relative Power Change (%)')
                    ax.set_title('Relative Theta Power Change')

                    # Compute power spectrum
                    psd = epoch.compute_psd(tmin=0, tmax=250, picks="eeg", method='welch', average=False)
                    power = (psd._data).reshape(n_channels, max((psd._data).shape))
                    power_whole = power.sum(axis=1)
                    
                    # Select frequency indices
                    freq = psd._freqs
                    low_freq_ind = (abs(freq - low_freq)).argmin()
                    high_freq_ind = (abs(freq - high_freq)).argmin()
                    
                    # Compute relative power change
                    power_range = power[:, low_freq_ind:high_freq_ind].sum(axis=1)
                    power_change = (power_range / power_whole) * 100  # Relative power change
                    
                    # Store values
                    time_points.append(epoch_count)
                    pow_change_values.append(power_change.tolist())  # Convert to list for Excel storage

                    # Update plot
                    ax.plot(time_points, pow_change_values, label=ch_names)
                    ax.legend(loc='upper right')
                    plt.draw()
                    plt.pause(0.1)  # Real-time update

                    epoch_count += 1
                
                plt.show()

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    # 🔹 Ensure data is saved even if script is stopped
    if time_points and pow_change_values:
        power_data = {
            'Epoch Count': time_points,
            'Power Change (%)': pow_change_values
        }
        df = pd.DataFrame(power_data)
        df.to_excel(output_path, index=False)
        print(f"Data saved successfully to {output_path}")

    print('Streams closed')

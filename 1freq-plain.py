# Import necessary libraries
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylsl  # Lab Streaming Layer (LSL) for Unity communication

import mne_realtime
from mne_realtime import LSLClient, MockLSLStream
import pyxdf
import mne
import keyboard  # For ESC key to stop

mne.viz.set_browser_backend('qt')

# LSL Stream Setup for Unity
stream_name = "PowerChangeStream"
lsl_info = pylsl.StreamInfo(stream_name, "EEG", 1, 10, pylsl.cf_float32, "power_change_001")
lsl_outlet = pylsl.StreamOutlet(lsl_info)

# EEG File Path
xdf_fpath = "C:/Users/varsh/.spyder-py3/eye-closed-rest.xdf"
streams, header = pyxdf.load_xdf(xdf_fpath)
eeg_stream = streams[0]

# User-defined parameters
step = 0.5  # seconds
time_window = 1  # seconds
n_channels = 2
ch_names = ['Ch1', 'Ch2']
low_freq, high_freq = 8, 12  # ALpha band 

data = streams[0]["time_series"].T[:n_channels]
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
output_path = r"C:\Users\varsh\OneDrive\Desktop\NFB-MNE-Psy\power_change_values.xlsx"

try:
    if __name__ == '__main__':
        with MockLSLStream("HMNN-LSL", raw, 'eeg', step):
            with LSLClient(info=raw.info, host="HMNN-LSL", wait_max=5) as client:
                client_info = client.get_measurement_info()
                sfreq = int(client_info['sfreq'])

                while running:
                    if keyboard.is_pressed('esc'):
                        print("Exit requested by user (ESC key).")
                        break

                    print(f'Got epoch {epoch_count}')
                    epoch = client.get_data_as_epoch(n_samples=int(time_window * sfreq))

                    # Clear the plot
                    ax.clear()
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Relative Power Change (%)')
                    ax.set_title('Relative Alpha Power Change')

                    # Compute Power Spectrum
                    psd = epoch.compute_psd(tmin=0, tmax=250, picks="eeg", method='welch', average=False)
                    power = (psd._data).reshape(n_channels, max((psd._data).shape))
                    power_whole = power.sum(axis=1)

                    # Frequency indices
                    freq = psd._freqs
                    low_freq_ind = (abs(freq - low_freq)).argmin()
                    high_freq_ind = (abs(freq - high_freq)).argmin()
                    power_range = power[:, low_freq_ind:high_freq_ind].sum(axis=1)

                    # Compute Relative Power Change
                    power_change = (power_range / power_whole) * 100
                    avg_power_change = np.mean(power_change)  # Single value for LSL

                    # Send power change to Unity via LSL
                    lsl_outlet.push_sample([avg_power_change])

                    # Store values
                    time_points.append(epoch_count)
                    pow_change_values.append(avg_power_change)

                    # Update plot
                    ax.plot(time_points, pow_change_values, label=ch_names)
                    ax.legend(loc='upper right')
                    plt.draw()
                    plt.pause(0.1)

                    epoch_count += 1
                
                plt.show()

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    if time_points and pow_change_values:
        power_data = {'Epoch Count': time_points, 'Power Change (%)': pow_change_values}
        df = pd.DataFrame(power_data)
        df.to_excel(output_path, index=False)
        print(f"Data saved successfully to {output_path}")

    print('Streams closed')

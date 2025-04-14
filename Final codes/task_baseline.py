<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:23:17 2024

@author: varsh
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg') 

import psychopy
from psychopy import visual, core, event
from psychopy.hardware import keyboard

import mne_realtime
from mne_realtime import LSLClient, MockLSLStream
#from mne.time_frequency import 
import pylsl
import pyxdf
from pylsl import resolve_stream, StreamInlet
import mne
mne.viz.set_browser_backend('qt')
import numpy as np
import time
import os
import pandas as pd
from pyprep.find_noisy_channels import NoisyChannels


# this is the host id that identifies your stream on LSL
host = 'Signal_generator'
# this is the max wait time in seconds until client connection
wait_max = 5

# Load a file to stream raw data


ch_names = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8', 'Ch9', 'Ch10', 'Ch11', 'Ch12', 'Ch13', 'Ch14', 'Ch15', 'Ch16']
#ch_names = ['Pz','T8','P3','C3','Fp1','Fp2','O1','P4','Fz','F7','C4','O2','F3','F4','Cz','T3']
running = True
duration = 120 #in seconds
epoch_count = 1

streams = resolve_stream()  # Resolves all available streams
for stream in streams:
    print(f"Stream Name: {stream.name()}, Type: {stream.type()}")

streams = resolve_stream('EEG')

if not streams:
    raise RuntimeError("No EEG streams found. Make sure the LSL stream is running.")

# Create an inlet to pull data
inlet = StreamInlet(streams[0])
print("Successfully connected to the EEG stream!")

# Create a mywindow
mywin = visual.Window([500, 500], monitor="TestMonitor", color=[0, 0, 0], fullscr=False, units="deg")
fixation = visual.ShapeStim(mywin, vertices=((0, -1), (0, 1), (0, 0), (-1, 0), (1, 0)), lineWidth=2, closeShape=False, lineColor="white")

# Create keyboard component
kb = keyboard.Keyboard()

# Create subplots: One row per channel
fig, axs = plt.subplots(len(ch_names), 1, figsize=(10, 12), sharex=True)
plt.subplots_adjust(hspace=0.5)

raw_data = []
start_time = time.time()

if __name__ == '__main__':
    with LSLClient(info=None, host=streams, wait_max=wait_max) as client:
        client_info = client.get_measurement_info()
        sfreq = int(client_info['sfreq'])

        while running:
            elapsed_time = time.time() - start_time
            if elapsed_time >= duration:
                break
            print(f'Got epoch {epoch_count}')

            epoch = client.get_data_as_epoch(n_samples=sfreq)
            inlet.append(epoch.get_data())

            epoch_count += 1
            fixation.draw()
            mywin.flip()

info = mne.create_info(ch_names=client_info['ch_names'], 
                       sfreq=client_info['sfreq'], 
                       ch_types='eeg')

data = np.concatenate(inlet, axis=2)
data = np.squeeze(data)
raw = mne.io.RawArray(data, info)

# Save baseline recording
baseline_path = r'C:\Users\varsh\OneDrive\Desktop\NFB-MNE-Psy\dummy-baseline.fif'
raw.save(baseline_path, overwrite=True)

# ---------- PSD CALCULATION ----------
psds, freqs = mne.time_frequency.psd_array_welch(raw, info["sfreq"], n_fft=2048)
psds_db = 10 * np.log10(psds)

bands = {
    'Delta (0.5–4 Hz)': (0.5, 4),
    'Theta (4–8 Hz)': (4, 8),
    'Alpha (8–13 Hz)': (8, 13),
    'Beta (13–30 Hz)': (13, 30),
    'Gamma (30–100 Hz)': (30, 100),
}

band_powers = []
plot_folder = 'psd_plots'
os.makedirs(plot_folder, exist_ok=True)

# Calculate band powers and generate plots
for ch_idx, ch_name in enumerate(raw.info['ch_names']):
    band_power = {'Channel': ch_name}
    for band_name, (fmin, fmax) in bands.items():
        idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
        mean_power = psds_db[ch_idx, idx_band].mean()
        band_power[band_name] = mean_power
    band_powers.append(band_power)

    # Plot PSD for channel
    plt.figure(figsize=(6, 4))
    plt.plot(freqs, psds_db[ch_idx, :], label=ch_name)
    plt.title(f'PSD - {ch_name}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(plot_folder, f'{ch_name}_psd.png')
    plt.savefig(plot_path)
    plt.close()

# Save to Excel
df = pd.DataFrame(band_powers)
excel_path = r'C:\Users\varsh\OneDrive\Desktop\NFB-MNE-Psy\Baseline_PSD.xlsx'

with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    # Write band powers
    df.to_excel(writer, sheet_name='Band_Powers', index=False)

    # Access the workbook and worksheet
    workbook = writer.book
    worksheet = workbook.add_worksheet('PSD_Plots')
    writer.sheets['PSD_Plots'] = worksheet

    row, col = 0, 0
    for ch_idx, ch_name in enumerate(raw.info['ch_names']):
        img_path = os.path.join(plot_folder, f'{ch_name}_psd.png')
        worksheet.insert_image(row, col, img_path, {'x_scale': 0.6, 'y_scale': 0.6})
        row += 20  # Adjust spacing for next image

print('Baseline recording and PSD with plots saved.')
mywin.close()
core.quit()
=======
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:23:17 2024

@author: varsh
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg') 

import psychopy
from psychopy import visual, core, event
from psychopy.hardware import keyboard

import mne_realtime
from mne_realtime import LSLClient, MockLSLStream
import pylsl
import pyxdf
from pylsl import resolve_stream, StreamInlet
import mne
mne.viz.set_browser_backend('qt')
import numpy as np
import time
from pyprep.find_noisy_channels import NoisyChannels


# this is the host id that identifies your stream on LSL
host = 'Signal_generator'
# this is the max wait time in seconds until client connection
wait_max = 5

# Load a file to stream raw data


#ch_names = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8', 'Ch9', 'Ch10', 'Ch11', 'Ch12', 'Ch13', 'Ch14', 'Ch15', 'Ch16']
ch_names = ['Pz','T8','P3','C3','Fp1','Fp2','O1','P4','Fz','F7','C4','O2','F3','F4','Cz','T3']
running = True
duration = 60 #in seconds
epoch_count = 1

streams = resolve_stream()  # Resolves all available streams
for stream in streams:
    print(f"Stream Name: {stream.name()}, Type: {stream.type()}")

streams = resolve_stream('EEG')

if not streams:
    raise RuntimeError("No EEG streams found. Make sure the LSL stream is running.")

# Create an inlet to pull data
inlet = StreamInlet(streams[0])
print("Successfully connected to the EEG stream!")

# Create a mywindow
mywin = visual.Window([500, 500], monitor="TestMonitor", color=[0, 0, 0], fullscr=False, units="deg")
fixation = visual.ShapeStim(mywin, vertices=((0, -1), (0, 1), (0, 0), (-1, 0), (1, 0)), lineWidth=2, closeShape=False, lineColor="white")

# Create keyboard component
kb = keyboard.Keyboard()

# Create subplots: One row per channel
fig, axs = plt.subplots(len(ch_names), 1, figsize=(10, 12), sharex=True)
plt.subplots_adjust(hspace=0.5)

raw_data = []
start_time = time.time()
# Main function to enable script as own program
if __name__ == '__main__':
   
    with LSLClient(info=None, host=streams, wait_max=wait_max) as client:
        client_info = client.get_measurement_info()
        sfreq = int(client_info['sfreq'])
            
            # Let's loop the data as a real-time stream
        while running:
            elapsed_time = time.time() - start_time
            if elapsed_time >= duration:
                break
            print(f'Got epoch {epoch_count}')
                
                # Get epoch data
            epoch = client.get_data_as_epoch(n_samples=sfreq)
            inlet.append(epoch.get_data())
            
                
            #     # Plot each channel individually
            # for i, ax in enumerate(axs):
            #     ax.clear()  # Clear previous data
            #     ax.plot(epoch.times, epoch.get_data(picks=[i])[0].T)
            #     ax.set_title(f'Channel: {ch_names[i]}')
            #     ax.set_ylabel('Amplitude (µV)')
                
            # axs[-1].set_xlabel('Time (s)')
                
            # plt.pause(0.1)
            epoch_count += 1
            fixation.draw()  # Draw fixation cross
                
            mywin.flip()  # Flip window to reflect changes

        # plt.draw()
info = mne.create_info(ch_names=client_info['ch_names'], 
                   sfreq=client_info['sfreq'], 
                   ch_types='eeg')
data = np.concatenate(inlet, axis=2)
data = np.squeeze(data)
raw = mne.io.RawArray(data, info)
raw.save(r'C:\Users\varsh\OneDrive\Desktop\NFB-MNE-Psy\dummy-baeline.fif', overwrite=True)
print('Streams closed')
mywin.close() #closing the psychopy window
core.quit()
>>>>>>> c1b84c90a17209315577e444e290d79e00963874

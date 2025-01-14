# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:52:36 2024

@author: varsh
"""

# Import the necessary libraries 
import psychopy
from psychopy import visual, core, event
from psychopy.hardware import keyboard
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.preprocessing import ICA 
from mne_icalabel import label_components
from mne import EpochsArray
from mne.baseline import rescale
from mne_realtime import LSLClient
import matplotlib 
matplotlib.use('QtAgg')
from pyprep.find_noisy_channels import NoisyChannels
from asrpy import ASR

## BASELINE CALIBRATION ##
# The host id that identifies the stream of interest on LSL
host = 'openbcigui'
# This is the max wait time in seconds until client connection
wait_max = 5

raw = mne.io.read_raw_fif(r"", preload=True) # Load the raw EEG baseline calibration
# Apply EEG re-referencing
raw.set_eeg_reference('average')
# Create a dictionary for renaming the electrodes to fit according to standard montage
rename_dict = {
    'EEG 001': 'Pz',
    'EEG 002': 'T8',
    'EEG 003': 'P3',
    'EEG 004': 'C3',
    'EEG 005': 'Fp1',
    'EEG 006': 'Fp2',
    'EEG 007': 'O1',
    'EEG 008': 'P4',
    'EEG 009': 'Fz',
    'EEG 0010': 'F7',
    'EEG 0011': 'C4',
    'EEG 0012': 'O2',
    'EEG 0013': 'F3',
    'EEG 0014': 'F4',
    'EEG 0015': 'Cz',
    'EEG 0016': 'T3',
    }
raw.rename_channels(rename_dict)
# Store the number of channels 
n_channels = len(rename_dict.keys())
# Set the montage 
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

# Apply notch filter to remove powerline noise and bandpass filter
raw.notch_filter(50, picks='eeg').filter(l_freq=0.1, h_freq=40)

# PREPROCESSING
# Bad channels detection and rejection using PREP pipeline RANSAC algorithm
nd = NoisyChannels(raw, random_state=1337)

nd.find_bad_ransac(channel_wise=True, max_chunk_size=1)
bad_channels = nd.bad_by_ransac
raw.info['bads'].extend(bad_channels)

# Artifact Subspace Reconstruction (ASR) to detect and reject non-bio artifacts
asr = ASR(sfreq=raw.info['sfreq']) 
asr.fit(raw)
raw = asr.tranform(raw)

# ICA to detect and remove independent components like eye-blinks, ECG, muscle artifacts
ica = ICA(n_components=n_channels-len(bad_channels), method='infomax', max_iter=500, random_state=42)
ica.fit(raw)

labels = label_components(raw, ica, 'iclabel')
component_labels = labels['labels']
component_probs = labels['y_pred_proba']
artifact_components = []
for i, prob in enumerate(component_probs):
    if prob >= 0.9 and prob <= 1 and component_labels[i] in ['muscle', 'eye']:
        print(f"Component (i) is likely an artifact")
        artifact_components.append(i)

ica_weights = ica.unmixing_matrix_
ica_inverse_weights = ica.mixing_matrix_
print("flagged artifact components: ", artifact_components)

raw_cleaned = ica.apply(raw.copy(), exclude=artifact_components)

ch_names = list(rename_dict.values())

fig, axs = plt.subplots(len(ch_names), 1, figsize=(10,12), sharex=True)
plt.subplots_adjust(hspace=0.5)

## VISUALIZATION ##
# Creating the window
mywin = visual.Window([1280, 720], monitor="TestMonitor", color=[-1, -1, -1], fullscr=False, units="cm")

# Creating the quadrant lines
line1 = visual.Line(win=mywin, start=(20, 0), end=(-20, 0), units='cm', lineWidth=2.0, pos=(0, 0), color=(-1, -1, -1), name='X-axis')
line2 = visual.Line(win=mywin, start=(0, 20), end=(0, -20), units='cm', lineWidth=2.0, pos=(0, 0), color=(-1, -1, -1), name='Y-axis')

# Creating a moving red dot 
dot = visual.Circle(win=mywin, radius=0.4, edges=128, fillColor='red', lineColor='white', pos=(0,0))

# Creating keyboard component
kb = keyboard.Keyboard()

# Craeting clock to control time and animation speed
clock = core.Clock()

# EEG parameters (user input)
step = 0.01 # in seconds
time_window = 5 # in seconds
n_channels = 2
feed_ch_names = ['O1']
freq_band_1 = (4, 7) # First freq band (theta)
high_freq = (8, 12) # second freq band (alpha)
frequency_bands = [freq_band_1, high_freq]

# Timer for updating the circle radius every 0.5 seconds
update_interval = 0.5
update_timer = core.CountdownTimer(update_interval)

## MAIN LOOP FOR REALTIME STREAM ##
# Main loop variable
epoch_count = 0  # Initialize epoch counter
running = True
feedback_data = []  # Initialize feedback data list

with LSLClient(info=None, host=host, wait_max=wait_max) as client:
    client_info = client.get_measurement_info()
    sfreq = int(client_info['sfreq'])
    print("LSLClient started, sampling frequency:", sfreq)
    
    while running:
        print(f'Got epoch {epoch_count}')
        
        keys = event.getKeys()
        
        if 'escape' in keys:
            running = False
            #Save the feedback value to an Excel file
            feedback_df = pd.DataFrame(feedback_data, columns=['Feedback value'])
            feedback_df.to_excel(r"", index=False)
            print("Exiting and Saving data.")
            break

        # Timer-based epoch fetching
        if update_timer.getTime() <= 0:
            print(f'Got epoch {epoch_count}')

            # Fetch epoch data
            epoch = client.get_data_as_epoch(n_samples=sfreq))
            #print(f"Received epoch {epoch_count} with {epoch.get_data().shape[1]} samples")

            #Apply baseline correction: subtract the mean of first 100ms
            epoch.apply_baseline(baseline=(0, None))
            data = np.squeeze(epoch.get_data())

            # preprocessing of realtime stream based on the calibarated data
            raw_realtime = mne.io.RawArray(data,client_info)
            raw_realtime.rename_channels(rename_dict)
            raw_realtime.notch_filter(50, picks='eeg').filter(l_freq=0.1, h_freq=40)
            raw_realtime.info['bads'].extend(bad_channels)
            raw_realtime_asr = asr.transform(raw_realtime)
            raw_realtime_asr_ica = ica.apply(raw_realtime_asr,exclude = artifact_components)

            # Compute power spectrum
            psd = raw_realtime_asr_ica.compute_psd(tmin=0, tmax=250, picks="eeg", method='welch', average=False)
            power = np.squeeze(psd.get_data(feed_ch_names))
            power_whole = power.sum()

            # Frequency indices and power calculations
            freq = psd.freqs  # Use correct property for frequencies
            power_changes = []
            for band in frequency_bands:
                low, high = band
                low_idx = np.where(freq >= low)[0][0]
                high_idx = np.where(freq <= high)[0][-1]
                power_band = power[low_idx:high_idx + 1].sum()
                power_change = (power_band / power_whole) * 100
                power_changes.append(power_change)

            # Append feedback data
            feedback_data.append({
                'Epoch':  epoch_count,
                **{f"Band {i + 1} Power Change (%)": power_changes[i] for i in range(len(frequency_bands))}
            })
            # Reset update timer and increment epoch count
            update_timer.reset(update_interval)
            epoch_count += 1

        # Update visualization based on power changes
        if len(power_changes) >= 2:
            x_pos = np.interp(power_changes[0], [0, 100], [-5, 5])
            y_pos = np.interp(power_changes[1], [0, 100], [-5, 5])
            dot.setPos((x_pos, y_pos))

        # Draw the stimuli
        dot.draw()
        line1.draw()
        line2.draw()
            
        mywin.flip()

    print('Streams closed')
mywin.close()
core.quit()


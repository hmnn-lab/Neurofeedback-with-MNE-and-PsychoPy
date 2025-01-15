# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:23:13 2024

@author: varsh
"""

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

# The host id that identifies the stream of interest on LSL
host = 'openbcigui'
# This is the max wait time in seconds until client connection
wait_max = 5

raw = mne.io.read_raw_fif(r"") # Load the raw EEG baseline calibration
# Apply notch filter to remove power-line noise and bandpass filter the signal
raw.notch_filter(50, picks='eeg').filter(l_freq=0.1, h_freq=40)
# Apply EEG re-referencing
raw.set_eeg_reference('average')
# Create a dictionary for renaming the electrodes to fit according to standard montage
rename_dict = {
    'EEG 001': 'Pz',
    'EEG 002': 'T8',
    'EEG 003': 'T5',
    'EEG 004': 'C3',
    'EEG 005': 'Fp1',
    'EEG 006': 'Fp2',
    'EEG 007': 'O1',
    'EEG 008': 'P4',
    'EEG 009': 'Fz',
    'EEG 010': 'F7',
    'EEG 011': 'C4',
    'EEG 012': 'T4',
    'EEG 013': 'F3',
    'EEG 014': 'F4',
    'EEG 015': 'Cz',
    'EEG 016': 'T3',
    }
raw.rename_channels(rename_dict)
# Store the number of channels 
n_channels = len(rename_dict.keys())
# Set the montage 
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

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

# Loop variables for visual output of the clean raw EEG data in realtime
ch_names = list(rename_dict.values())
# Initialize epoch couter
epoch_count = 0
running = True

fig, axs = plt.subplots(len(ch_names), 1, figsize=(10,12), sharex=True)
plt.subplots_adjust(hspace=0.5)

# Main loop for streaming of clean preprocessed EEG thru LSL
with LSLClient(info=None, host=host, wait_max=wait_max) as client:
    client_info = client.get_measurement_info()
    sfreq = int(client_info['sfreq'])
    
    while running:
        print(f'Got epoch {epoch_count}')
        
        #Getting data from LSL as an epoch window of 1s
        epoch = client.get_data_as_epoch(n_samples=sfreq)
        
        #Applying baseline correction i.e. subtract the mean of first 100ms
        epoch.apply_baseline(baseline=(0, None))
        data = np.squeeze(epoch.get_data())
        # Applying all the required preprocessing steps on realtime stream 
        raw_realtime = mne.io.RawArray(data, client_info)
        raw_realtime.rename_channels(rename_dict)
        raw_realtime.info['bads'].extend(bad_channels)
        raw_realtime_asr = asr.transform(raw_realtime)
        raw_realtime_asr_ica = ica.apply(raw_realtime_asr, exclude=artifact_components)
        
        #Plotting individual channels
        for i, ax in enumerate(axs):
            ax.clear()
            ax.plot(raw_realtime_asr_ica.times, raw_realtime_asr_ica.get_data(picks=[i])[0].T)
            ax.set_title(f'Channel: {ch_names[i]}')
            ax.set_ylabel('Amplitude (μV)')
        
        axs[-1].set_xlabel('Time (s)')
        plt.pause(0.1)
        
        epoch_count += 1
    plt.draw()
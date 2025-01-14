# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:34:18 2024

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

# BASELINE CALIBRATION
# The host id that identifies the stream of interest on LSL
host = 'openbcigui'
# This is the max wait time in seconds until client connection
wait_max = 5

raw = mne.io.read_raw_fif(r"", preload=True) # Load the raw EEG baseline calibration
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
    'EEG 0010': 'F7',
    'EEG 0011': 'C4',
    'EEG 0012': 'T4',
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

# VISUALIZATION 
# Create a window
mywin = visual.Window([1920, 1080], monitor="TestMonitor", color=[0,0,0], fullscr=True, units="deg")
# Create fixation cross
fixation = visual.ShapeStim(mywin, vertices=((0, -1), (0, 0), (-1, 0), (1, 0)), lineWidth=2, closShape=False, lineColor="white")

# Create square window and circle
sqr = visual.Rect(win=mywin, size=[20, 20], fillColor='black', pos=(0.0, 0.0))
mycir = visual.Circle(win=mywin, radius=5.0, edges=128, lineWidth=1.5, lineColor='White', fillColor='red', pos(0, 0))

# Parameters for smooth radius change
max_radius = 10
min_radius = 2

# Create keyboard component
kb = keyboard.Keyboard()

#EEG parameters
step = 0.01 # in seconds
time_window = 5 # in seconds
n_channels = 2
feed_ch_names = ['O1']
low_freq = 8 # alpha band
high_freq = 12

# Timer for updating the circle radius every 0.5 seconds
update_interval = 0.5
update_timer = core.CountdownTimer(update_interval)

#MAIN LOOP FOR REALTIME STREAM
# Main loop variable
epoch_count = 0 # Initialize epoch couter
running = True
time_points = []
pow_change_values =[]
feedback_values = []

# Main loop for streaming of clean preprocessed EEG thru LSL
with LSLClient(info=None, host=host, wait_max=wait_max) as client:
    client_info = client.get_measurement_info()
    sfreq = int(client_info['sfreq'])
    
    while running:
        print(f'Got epoch {epoch_count}')
        
        keys = event.getKeys()
        if 'escape' in keys:
            running = False
            # Saving the feedback values to an excel file
            feedback_df = pd.DataFrame(feedback_values, columns=['Feedback Radius'])
            feedback_df.to_excel(r"C:/Users/Admin/Desktop/Varsha/mne-psychopy-codes/feedback_values.xlsx", index=False)
            print("Exiting and Saving data...")
            break
        
        if update_timer.getTime() <= 0:
            print(f'Got epoch {epoch_count}')
            
            # Getting data from LSL as an epoch
            epoch = client.get_data_as_epoch(n_samples=sfreq)
            
            # Applying baseline correction 
            epoch.apply_baseline(baseline=(0, None)) 
            data = np.squeeze(epoch.get_data())

            # Applying all the required preprocessing steps on realtime stream 
            raw_realtime = mne.io.RawArray(data, client_info)
            raw_realtime.rename_channels(rename_dict)
            raw_realtime.info['bads'].extend(bad_channels)
            raw_realtime_asr = asr.transform(raw_realtime)
            raw_realtime_asr_ica = ica.apply(raw_realtime_asr, exclude=artifact_components)
            
            # Getting epoch data and computing power value (PSD)
            # epoch = client.get_data_as_epoch(n_samples=int(time_window*sfreq))
            psd = raw_realtime_asr_ica.compute_psd(tmin=0, tmax=250, picks="eeg", method='welch', average=False)
            
            # Reshaping power data
            power = np.squeeze(psd.get_data(feed_ch_names))
            power_whole = power.sum()
            
            # Frequency indices and boundaries
            frq = psd._freqs
            low_freq_ind = (abs(freq - low_freq)).argmin()
            high_frq_ind = (abs(freq - high_freq)).argmin()
            
            # Summing power in the chosen freq band
            power_range = (power[low_freq_ind:high_freq_ind]).sum()
            
            # Computing relative power change
            power_change = (power_range/power_whole) * 100
            time_points.append(epoch_count)
            pow_change_values.append(power_change)
            
            # Reset update timer
            update_timer.reset(update_interval)
            epoch_count += 1
        
        # Updating the radius
        if len(pow_change_values) > 0:
            current_power_change = pow_change_values[-1]
            new_radius = np.interp(np.mean(current_power_change), [0, 100], [min_radius, max_radius])
            mycir.radius = new_radius
            feedback_values.append(new_radius) # Stores the feedback value
            print(f'The feedback value is: {new_radius}')
            
        # Drawing the stimuli            
        sqr.draw()
        mycir.draw()
        fixation.draw()
        
        mywin.flip() # to reflect the changes
mywin.close()
core.quit()

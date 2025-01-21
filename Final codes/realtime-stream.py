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


ch_names = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8', 'Ch9', 'Ch10', 'Ch11', 'Ch12', 'Ch13', 'Ch14', 'Ch15', 'Ch16']



running = True
duration = 120 #in seconds
epoch_count = 1

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
   
    with LSLClient(info=None, host=host, wait_max=wait_max) as client:
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
            raw_data.append(epoch.get_data())
            
                
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
mywin.close() #closing the psychopy window
core.quit()

info = mne.create_info(ch_names=client_info['ch_names'], 
                   sfreq=client_info['sfreq'], 
                   ch_types='eeg')
data = np.concatenate(raw_data, axis=2)
data = np.squeeze(data)
raw = mne.io.RawArray(data, info)
raw.save(r'C:/Users/Admin/Desktop/Varsha/mne-psychopy-codes/baseline-data-priya-raw.fif', overwrite=True)
print('Streams closed')

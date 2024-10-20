# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 20:03:21 2024

@author: varsh
"""
#%%#
import psychopy
from psychopy import visual, core, event
from psychopy.hardware import keyboard
import pyxdf
import numpy as np

import matplotlib.pyplot as plt


import mne_realtime
from mne_realtime import LSLClient, MockLSLStream

import pylsl

import mne
#%%
# Create a mywindow
mywin = visual.Window([500, 500], monitor="TestMonitor", color=[0, 0, 0], fullscr=False, units="deg")
""" Creating quadrants
line1 = visual.Line(win=mywin, start=(10, 0), end=(-10, 0), units='cm', lineWidth=2.0,pos=(0, 0), size=1.0, anchor='center', ori=0.0, opacity=1.0, contrast=0.5, draggable=True, name='X-ais')
line1.setColor((1, 1, 1))
line2 = visual.Line(win=mywin, start=(0, 10), end=(0, -10), units='cm', lineWidth=2.0, pos=(0, 0), size=1.0, anchor='center', ori=0.0, opacity=1.0, contrast=0.5, draggable=True, name='Y-axis')
line2.setColor((1, 1, 1))
"""
#Create square window
sqr = visual.Rect(win=mywin, size=[20, 20], fillColor='black', pos=(0.0,0.0))

# Create a circle stimulus
mycir = visual.Circle(win=mywin, radius=5.0, edges=128, lineWidth=1.5, lineColor='white', fillColor='red', colorSpace='rgb', pos=(0, 0), size=1.0)

# Parameters for smooth radius change
max_radius = 10
min_radius = 2

# Create keyboard component
kb = keyboard.Keyboard()

# Clock to control time and animation speed
clock = core.Clock()

# this is the host id that identifies your stream on LSL
host = 'HMNN-LSL'
# this is the max wait time in seconds until client connection
wait_max = 5

#Loading the .xdf file i.e. raw EEG data
xdf_fpath = "C:/Users/varsh/.spyder-py3/eye-open-mc.xdf"

streams, header = pyxdf.load_xdf(xdf_fpath)
eeg_stream = streams[0]

# User input
step = 0.01  # in seconds
time_window = 5  # in seconds
n_channels = 2
ch_names = ['Ch1', 'Ch2']
low_freq = 4
high_freq = 7
data = streams[0]["time_series"].T
data = data[:n_channels]

sfreq = float(streams[0]["info"]["nominal_srate"][0])
info = mne.create_info(ch_names, sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(data, info)
#
raw.apply_function(lambda x: x*1e-6, picks="eeg")

raw.notch_filter(50, picks='eeg').filter(l_freq=0.1, h_freq=40)

# Variables for tracking
running = True
epoch_count = 1
time_points = []
pow_change_values = []

# Timer for updating the circle radius every 0.5 seconds
update_interval = 0.5  # seconds
update_timer = core.CountdownTimer(update_interval)

########## MAIN LOOP NOT WORKING -  PSYCHOPY WINDOW NOT RESPONDING #########
if __name__ == '__main__':
    with MockLSLStream(host, raw, 'eeg', step):
        with LSLClient(info=raw.info, host=host, wait_max=wait_max) as client:
            client_info = client.get_measurement_info()
            sfreq = int(client_info['sfreq'])
            
            while running:
                # Poll for keyboard events
                keys = event.getKeys()
                if 'escape' in keys:
                    running = False
                    np.save(r"C:\Users\varsh\OneDrive\Desktop\Neurofeedback MNE-PsychoPy\pow-change.npy", pow_change_values)
                    print("Exiting and saving data")
                    break
                # Get new data every time the update timer reaches zero
                if update_timer.getTime() <= 0:
                    print(f'Got epoch {epoch_count}')
                    
                    # Get epoch data and compute power spectrum
                    epoch = client.get_data_as_epoch(n_samples=int(time_window * sfreq))
                    psd = epoch.compute_psd(tmin=0, tmax=250, picks="eeg", method='welch', average=False)
                    
                    # Reshape power data
                    power = (psd._data).reshape(n_channels, max((psd._data).shape))
                    power_whole = power.sum(axis=1)
                    
                    # Frequency indices and boundaries
                    freq = psd._freqs
                    low_freq_ind = (abs(freq - low_freq)).argmin()
                    high_freq_ind = (abs(freq - high_freq)).argmin()
                    
                    # Sum power in the chosen frequency band for both channels
                    power_range = (power[:, low_freq_ind:high_freq_ind]).sum(axis=1)
                    
                    # Compute relative power change 
                    power_change = (power_range / power_whole) * 100
                    
                    # Append the current epoch count and power_change to the lists
                    time_points.append(epoch_count)
                    pow_change_values.append(power_change)
                    
                    # Reset update timer for the next 0.5 seconds
                    #update_timer.reset(update_interval)
                    
                    epoch_count += 1
                
                # Check if 'escape' is pressed to exit
                # if event.getKeys(['escape']):
                #     running = False
                #     np.save(r"C:\Users\varsh\OneDrive\Desktop\Neurofeedback MNE-PsychoPy\pow-change.npy", pow_change_values)
                #     break
                
                # Draw the circle with the updated radius if there's power change data
                if len(pow_change_values) > 0:
                    current_power_change = pow_change_values[-1]  # Select the last power change value
                    new_radius = np.interp(np.mean(current_power_change), [0, 100], [min_radius, max_radius])
                    #print(f'Updated radius: {new_radius}')
                    
                    # Update the circle's radius as feedback value
                    feedbackVal = print(f'The feedback value is: {new_radius}')
                    
                # Draw the updated circle and flip the window to make changes visible
                sqr.draw()
                mycir.draw()
                #line1.draw()
                #line2.draw()
                
                mywin.flip()

print('Streams closed')
mywin.close()
core.quit()

#%%
while running:
    if event.getKeys(['escape']):
        running = False
        np.save(r"C:\Users\varsh\OneDrive\Desktop\Neurofeedback MNE-PsychoPy\pow-change.npy", pow_change_values)
        break


# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:36:42 2024

@author: varsh
"""

import psychopy
from psychopy import visual, core, event
from psychopy.hardware import keyboard
import pyxdf
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import mne_realtime
from mne_realtime import LSLClient, MockLSLStream
import pylsl
import mne


# Create a window
mywin = visual.Window([720, 720], monitor="TestMonitor", color=[-1, -1, -1], fullscr=False, units="cm")

# Create quadrant lines
line1 = visual.Line(win=mywin, start=(20, 0), end=(-20, 0), units='cm', lineWidth=2.0, pos=(0, 0), color=(-1, -1, -1), name='X-axis')
line2 = visual.Line(win=mywin, start=(0, 20), end=(0, -20), units='cm', lineWidth=2.0, pos=(0, 0), color=(-1, -1, -1), name='Y-axis')

# Create a red dot to represent power change
dot = visual.Circle(win=mywin, radius=0.4, edges=128, fillColor='red', lineColor='white', pos=(0, 0))

# Create keyboard component
kb = keyboard.Keyboard()

# Clock to control time and animation speed
clock = core.Clock()

# LSL setup
host = 'HMNN-LSL'
wait_max = 5

# Loading the .xdf file (raw EEG data)
xdf_fpath = "C:/Users/varsh/.spyder-py3/eye-open-mc.xdf"
streams, header = pyxdf.load_xdf(xdf_fpath)
eeg_stream = streams[0]

# User input for frequency bands
step = 0.01  # in seconds
time_window = 5  # in seconds
n_channels = 2
ch_names = ['Ch1', 'Ch2']

# Define two frequency bands
freq_band_1 = (4, 7)  # First band (Theta)
freq_band_2 = (8, 12)  # Second band (Alpha)


data = streams[0]["time_series"].T
data = data[:n_channels]

sfreq = float(streams[0]["info"]["nominal_srate"][0])
info = mne.create_info(ch_names, sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(data, info)

# Preprocess raw data
raw.apply_function(lambda x: x * 1e-6, picks="eeg")
raw.notch_filter(50, picks='eeg').filter(l_freq=0.1, h_freq=40)

# Tracking variables
running = True
epoch_count = 1
time_points = []
pow_change_values_band_1 = []
pow_change_values_band_2 = []

# Timer for updating the red dot position
update_interval = 0.5  # seconds
update_timer = core.CountdownTimer(update_interval)

# EEG parameters dictionary
eeg_params = {
    "Step (s)": step,
    "Time Window (s)": time_window,
    "Number of Channels": n_channels,
    #"Feed Channels": ', '.join(feed_ch_names),
    "Frequency Band 1 (Theta)": f"{freq_band_1}",
    "Frequency Band 2 (Alpha)": f"{freq_band_2}",
    "Update Interval (s)": update_interval
}

# Prepare feedback data storage
feedback_data = []

if __name__ == '__main__':
    with MockLSLStream(host, raw, 'eeg', step):
        with LSLClient(info=raw.info, host=host, wait_max=wait_max) as client:
            client_info = client.get_measurement_info()
            sfreq = int(client_info['sfreq'])
            
            while running:
                # Check for keyboard input
                keys = kb.getKeys()
                if 'escape' in keys:
                    running = False
                    break

                # Check if timer has elapsed to update the red dot
                if update_timer.getTime() <= 0:
                    print(f'Got epoch {epoch_count}')
                    
                    # Get epoch data and compute power spectrum
                    epoch = client.get_data_as_epoch(n_samples=int(time_window * sfreq))
                    psd = epoch.compute_psd(tmin=0, tmax=250, picks="eeg", method='welch', average=False)
                    
                    # Reshape power data
                    power = (psd._data).reshape(n_channels, max((psd._data).shape))
                    power_whole = power.sum(axis=1)
                    
                    # Frequency indices for the first and second bands
                    freq = psd._freqs
                    low_freq_ind_1 = (abs(freq - freq_band_1[0])).argmin()
                    high_freq_ind_1 = (abs(freq - freq_band_1[1])).argmin()
                    low_freq_ind_2 = (abs(freq - freq_band_2[0])).argmin()
                    high_freq_ind_2 = (abs(freq - freq_band_2[1])).argmin()
                    
                    # Sum power in the chosen frequency bands for both channels
                    power_range_1 = (power[:, low_freq_ind_1:high_freq_ind_1]).sum(axis=1)
                    power_range_2 = (power[:, low_freq_ind_2:high_freq_ind_2]).sum(axis=1)
                    
                    # Compute relative power change
                    power_change_band_1 = (power_range_1 / power_whole) * 100
                    power_change_band_2 = (power_range_2 / power_whole) * 100
                    
                    # Append power change values
                    pow_change_values_band_1.append(power_change_band_1)
                    pow_change_values_band_2.append(power_change_band_2)
                    
                    # Save feedback data (power changes) for Excel export
                    feedback_data.append({
                        'Epoch': epoch_count,
                        'Power Change Band 1': np.mean(power_change_band_1),
                        'Power Change Band 2': np.mean(power_change_band_2)
                    })
                    
                    # Reset update timer
                    update_timer.reset(update_interval)
                    epoch_count += 1
                
                # Update the red dot's position based on the latest power changes
                if len(pow_change_values_band_1) > 0 and len(pow_change_values_band_2) > 0:
                    current_power_change_1 = pow_change_values_band_1[-1]  # Last power change for band 1
                    current_power_change_2 = pow_change_values_band_2[-1]  # Last power change for band 2
                    
                    # Interpolate the power change values to window coordinates (use smaller range)
                    x_pos = np.interp(np.mean(current_power_change_1), [0, 100], [-5, 5])  # X-axis
                    y_pos = np.interp(np.mean(current_power_change_2), [0, 100], [-5, 5])  # Y-axis
                    
                    # Update dot position
                    dot.setPos((x_pos, y_pos))
                
                # Draw the quadrant lines and red dot
                dot.draw()
                line1.draw()
                line2.draw()
                
                # Flip the window to update the display
                mywin.flip()

# Save feedback data to Excel
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"EEG_Feedback_{timestamp}.xlsx"

# Convert feedback data to a DataFrame
df_feedback = pd.DataFrame(feedback_data)

# Save EEG parameters and feedback data
with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
    # Save EEG parameters in the first sheet
    df_params = pd.DataFrame(list(eeg_params.items()), columns=["Parameter", "Value"])
    df_params.to_excel(writer, sheet_name="EEG Parameters", index=False)

    # Save feedback data in another sheet
    df_feedback.to_excel(writer, sheet_name="Feedback Data", index=False)

# Close streams and PsychoPy window
print('Streams closed')
mywin.close()
core.quit()


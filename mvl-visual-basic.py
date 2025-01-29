# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:39:05 2024

@author: varsh
"""
import numpy as np
import mne
from scipy.signal import hilbert
import pyxdf
import psychopy
from psychopy import visual, core
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from mne_realtime import LSLClient, MockLSLStream
import pylsl

# Load EEG data from the XDF file
xdf_fpath = "C:/Users/varsh/.spyder-py3/random-block-1.xdf"
streams, header = pyxdf.load_xdf(xdf_fpath)

# Extract EEG stream
eeg_stream = streams[0]
data = eeg_stream["time_series"].T  # Transpose to get channels x samples
sfreq = float(eeg_stream["info"]["nominal_srate"][0])  # Sampling frequency

# LSL setup
host = 'HMNN-LSL'
wait_max = 5
running = True
epoch_count = 1
update_interval = 0.5  # seconds
update_timer = core.CountdownTimer(update_interval)



# User-defined parameters
step = 0.1  # Step size in seconds
time_window = 1  # Time window in seconds
n_channels = 2  # Number of EEG channels to use
ch_names = ['F3', 'O1']  # EEG channel names
low_freq_band = (4, 8)  # Theta band for phase
high_freq_band = (80, 120)  # High gamma band for amplitude

# Limit the data to the first `n_channels`
data = data[:n_channels]

# Create MNE Raw object
info = mne.create_info(ch_names, sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(data, info)
raw.apply_function(lambda x: x * 1e-6, picks="eeg")  # Convert to microvolts if needed

# DEBUG: Checking if we are receiving data
print("Starting the LSLClient and preparing to receive data...")

if __name__ == '__main__':
    # Using MockLSLStream for testing, replace with real EEG stream for live data
    with MockLSLStream(host, raw, 'eeg', step):
        with LSLClient(info=raw.info, host=host, wait_max=wait_max) as client:
            client_info = client.get_measurement_info()
            sfreq = int(client_info['sfreq'])
            
            # DEBUG: Check if LSLClient has started correctly
            print("LSLClient started, sampling frequency:", sfreq)
    
                # Check if timer has elapsed to update the red dot
            if update_timer.getTime() <= 0:
                print(f'Fetching epoch {epoch_count}')
                    
                    # Fetch epoch data

# Extract data for a specific time range (adjust as needed)
tmin, tmax = 0, 10  # Time range in seconds
start, stop = raw.time_as_index([tmin, tmax])
data, times = raw[:, start:stop]

# Bandpass filtering for low (phase) and high (amplitude) frequency bands
low_signal = mne.filter.filter_data(data[0], sfreq, low_freq_band[0], low_freq_band[1], verbose=False)
high_signal = mne.filter.filter_data(data[0], sfreq, high_freq_band[0], high_freq_band[1], verbose=False)

# Hilbert transform to get instantaneous phase and amplitude
low_phase = np.angle(hilbert(low_signal))  # Phase of the low-frequency signal
high_amp = np.abs(hilbert(high_signal))   # Amplitude envelope of the high-frequency signal

# Compute the normalized Mean Vector Length (ρ̂_D)
N = len(low_phase)
complex_mvl = (1 / np.sqrt(N)) * np.sum(high_amp * np.exp(1j * low_phase)) / np.sqrt(np.sum(high_amp**2))

# Absolute value (strength of coupling)
mvl_abs = np.abs(complex_mvl)

# Angle (phase of the coupling)
mvl_angle = np.angle(complex_mvl)

# Convert polar coordinates to Cartesian for plotting
x = mvl_abs * np.cos(mvl_angle)  # X-coordinate
y = mvl_abs * np.sin(mvl_angle)  # Y-coordinate

# PsychoPy Visualization
win = visual.Window([800, 800], color="white", units="pix")  # Create a PsychoPy window

# Draw polar plot background with concentric circles
for radius in np.linspace(50, 300, 6):  # Create concentric circles
    circle = visual.Circle(win, radius=radius, edges=128, lineColor="lightgray", fillColor=None, pos=(0, 0))
    circle.draw()

# Draw radial lines for angles (0° to 360° at 30° intervals)
for angle in range(0, 360, 30):
    angle_rad = np.radians(angle)
    line = visual.Line(
        win, 
        start=(0, 0), 
        end=(300 * np.cos(angle_rad), 300 * np.sin(angle_rad)), 
        lineColor="lightgray"
    )
    line.draw()

    # Add angle text
    angle_text = visual.TextStim(
        win,
        text=f"{angle}°",
        pos=(320 * np.cos(angle_rad), 320 * np.sin(angle_rad)),
        color="black",
        height=15,
        alignText="center",
    )
    angle_text.draw()

# Plot the MVL as a red dot
mvl_dot = visual.Circle(
    win, radius=10, fillColor="red", lineColor="red", pos=(x * 300, y * 300)
)
mvl_dot.draw()

# Add radial strength labels
for radius, label in zip(np.linspace(50, 300, 6), np.linspace(0, 0.02, 6)):
    text = visual.TextStim(
        win,
        text=f"{label:.3f}",
        pos=(radius, 10),  # Adjusted for readability
        color="black",
        height=15,
    )
    text.draw()

# Add text for MVL information
text = visual.TextStim(
    win,
    text=f"Theta: {np.degrees(mvl_angle):.2f}°\nR: {mvl_abs:.6f}",
    pos=(0, -350),
    color="black",
    height=20,
    alignText="center",
)
text.draw()

# Show the plot
win.flip()
core.wait(5)  # Display the window for 5 seconds
win.close()

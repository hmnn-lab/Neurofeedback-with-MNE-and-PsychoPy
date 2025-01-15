# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:39:38 2024

@author: varsh
"""

import numpy as np
import mne
import scipy
from scipy.signal import hilbert
from mne_realtime import LSLClient, MockLSLStream
from psychopy import visual, core
import pyxdf

# Load EEG data from the .xdf file
xdf_fpath = "C:/Users/varsh/.spyder-py3/random-block-1.xdf"
streams, header = pyxdf.load_xdf(xdf_fpath)

# Extract EEG stream
eeg_stream = streams[0]
data = streams[0]["time_series"].T
ch_names = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8']
sfreq = float(streams[0]["info"]["nominal_srate"][0])
info = mne.create_info(ch_names, sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(data, info)
raw.apply_function(lambda x: x*1e-6, picks="eeg")
raw.notch_filter(50, picks='eeg').filter(l_freq=0.1, h_freq=40)
# User-defined parameters
low_freq_band = (4, 8)  # Theta band
#high_freq_band = (80, 120)  # High gamma band
high_freq_band = (8, 12)  # Alpha band
update_interval = 0.5  # Update visualization every 0.5 seconds
time_window = 1  # Time window for epochs in seconds


# PsychoPy visualization setup
win = visual.Window([800, 800], color="white", units="pix")

# Function to draw the static polar plot background
def draw_polar_background():
    # Draw concentric circles
    for radius in np.linspace(50, 300, 6):
        circle = visual.Circle(win, radius=radius, edges=128, lineColor="lightgray", fillColor=None, pos=(0, 0))
        circle.draw()

    # Draw radial lines and angle labels
    for angle in range(0, 360, 30):
        angle_rad = np.radians(angle)
        line = visual.Line(win, start=(0, 0), end=(300 * np.cos(angle_rad), 300 * np.sin(angle_rad)), lineColor="lightgray")
        line.draw()
        angle_text = visual.TextStim(win, text=f"{angle}°", pos=(320 * np.cos(angle_rad), 320 * np.sin(angle_rad)), color="black", height=15, alignText="center")
        angle_text.draw()

# Draw the static background once
draw_polar_background()
win.flip()

# LSL setup
host = 'MNE'
wait_max = 5
running = True
epoch_count = 1

# Real-time processing loop
if __name__ == '__main__':
    with MockLSLStream(host, raw, 'eeg') as mock_stream:
        with LSLClient(info=raw.info, host=host, wait_max=wait_max) as client:
            client_info = client.get_measurement_info()
            sfreq = int(client_info['sfreq'])
            print("LSLClient connected, sampling frequency:", sfreq)

            update_timer = core.CountdownTimer(update_interval)  # Initialize timer

            while running:
                # Check if the timer has elapsed
                if update_timer.getTime() <= 0:
                    print(f"Fetching epoch {epoch_count}")

                    # Fetch real-time EEG data as epoch
                    epoch = client.get_data_as_epoch(n_samples=int(time_window * sfreq))
                    data, times = epoch.get_data(picks="eeg")[0], epoch.times

                    # Bandpass filtering for low (phase) and high (amplitude) bands
                    low_signal = mne.filter.filter_data(data[0], sfreq, low_freq_band[0], low_freq_band[1], verbose=False)
                    high_signal = mne.filter.filter_data(data[0], sfreq, high_freq_band[0], high_freq_band[1], verbose=False)

                    # Hilbert transform for phase and amplitude
                    low_phase = np.angle(hilbert(low_signal))
                    high_amp = np.abs(hilbert(high_signal))

                    # MVL calculation
                    N = len(low_phase)
                    complex_mvl = (1 / np.sqrt(N)) * np.sum(high_amp * np.exp(1j * low_phase)) / np.sqrt(np.sum(high_amp**2))
                    mvl_abs = np.abs(complex_mvl)
                    mvl_angle = np.angle(complex_mvl)

                    # Cartesian coordinates for visualization
                    x = mvl_abs * np.cos(mvl_angle) * 300
                    y = mvl_abs * np.sin(mvl_angle) * 300

                    # Clear the window and redraw the background
                    win.clearBuffer()
                    draw_polar_background()

                    # Draw MVL dot
                    mvl_dot = visual.Circle(win, radius=10, fillColor="red", lineColor="red", pos=(x, y))
                    mvl_dot.draw()

                    # Add MVL information text
                    mvl_text = visual.TextStim(win, text=f"Theta: {np.degrees(mvl_angle):.2f}°\nR: {mvl_abs:.6f}", pos=(0, -350), color="black", height=20, alignText="center")
                    mvl_text.draw()

                    # Refresh display
                    win.flip()

                    # Reset timer
                    update_timer.reset(update_interval)
                    epoch_count += 1

                # Escape key to stop
                if 'escape' in [key.name for key in visual.event.getKeys()]:
                    running = False

# Close the PsychoPy window
win.close()
core.quit()

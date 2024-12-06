# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:39:00 2024

@author: varsh
"""

import numpy as np
import mne
from scipy.signal import butter, freqz
from mne_realtime import LSLClient, MockLSLStream
from psychopy import visual, core, event
import pyxdf
import pandas as pd
from datetime import datetime

# Function for Endpoint Corrected Hilbert Transform
def echt(xr, filt_lf, filt_hf, Fs, n=None):
    if not np.isrealobj(xr):
        print("Warning: Ignoring imaginary part of input signal.")
        xr = np.real(xr)
    if n is None:
        n = len(xr)
    x = np.fft.fft(xr, n=n)
    h = np.zeros(n, dtype=float)
    if n > 0 and n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:n // 2] = 2
    elif n > 0:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    x *= h
    filt_order = 2
    b, a = butter(filt_order, [filt_lf / (Fs / 2), filt_hf / (Fs / 2)], btype='bandpass')
    T = 1 / Fs * n
    filt_freq = np.fft.fftfreq(n, d=1 / Fs)
    filt_coeff = freqz(b, a, worN=filt_freq, fs=Fs)[1]
    x = np.fft.fftshift(x)
    x *= filt_coeff
    x = np.fft.ifftshift(x)
    analytic_signal = np.fft.ifft(x)
    phase = np.angle(analytic_signal)
    amplitude = np.abs(analytic_signal)
    return analytic_signal, phase, amplitude

# Load EEG data from the .xdf file
xdf_fpath = "C:/Users/varsh/.spyder-py3/random-block-1.xdf"
streams, header = pyxdf.load_xdf(xdf_fpath)
eeg_stream = next(stream for stream in streams if 'EEG' in stream['info']['type'][0])
data = eeg_stream["time_series"].T
sfreq = float(eeg_stream["info"]["nominal_srate"][0])
ch_names = ['Ch1', 'Ch2']
n_channels = len(ch_names)

# User-defined parameters
low_freq_band = (4, 8)
high_freq_band = (80, 120)
update_interval = 0.5
time_window = 1

# Create an MNE Raw object
info = mne.create_info(ch_names, sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(data[:n_channels], info)
raw.notch_filter(50, picks='eeg').filter(l_freq=0.1, h_freq=40)

# PsychoPy visualization setup
# Create a window
mywin = visual.Window([500, 500], monitor="TestMonitor", color=[-1,-1,-1], fullscr=False, units="cm")
# Create fixation cross
fixation = visual.ShapeStim(mywin, vertices=((0, -1), (0, 1), (0, 0), (-1, 0), (1, 0)), lineWidth=2, closeShape=False, lineColor="white")
# Create a dynamic circle for visualization
dynamic_circle = visual.Circle(win=mywin, radius=5, fillColor="cyan", lineColor="white", pos=(0, 0))

# Parameters for smooth radius change
max_radius = 10
min_radius = 2 # Minimum visual radius of the circle

# Timer for updating the circle radius every 0.5 seconds
update_interval = 0.5
update_timer = core.CountdownTimer(update_interval)

# LSL setup
host = 'MNE'
wait_max = 5
running = True
epoch_count = 1
results = []

if __name__ == '__main__':
    try:
        with MockLSLStream(host, raw, 'eeg') as mock_stream:
            with LSLClient(info=raw.info, host=host, wait_max=wait_max) as client:
                client_info = client.get_measurement_info()
                sfreq = int(client_info['sfreq'])
                print("LSLClient connected, sampling frequency:", sfreq)

                update_timer = core.CountdownTimer(update_interval)

                while running:
                    if update_timer.getTime() <= 0:
                        print(f"Fetching epoch {epoch_count}")
                        epoch = client.get_data_as_epoch(n_samples=int(time_window * sfreq))
                        data, times = epoch.get_data(picks="eeg")[0], epoch.times

                        # Use echt for phase and amplitude extraction
                        _, low_phase, _ = echt(data[0], low_freq_band[0], low_freq_band[1], sfreq)
                        _, _, high_amp = echt(data[0], high_freq_band[0], high_freq_band[1], sfreq)

                        # MVL calculation
                        N = len(low_phase)
                        complex_mvl = (1 / np.sqrt(N)) * np.sum(high_amp * np.exp(1j * low_phase)) / np.sqrt(np.sum(high_amp**2))
                        mvl_abs = np.abs(complex_mvl)

                        # Scale the radius based on mvl_abs
                        scaled_radius = min_radius + (mvl_abs * (max_radius - min_radius))

                        # Update the circle's radius and redraw
                        dynamic_circle.radius = scaled_radius
                        mywin.clearBuffer()
                        dynamic_circle.draw()
                        fixation.draw()
                        mywin.flip()

                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        results.append({
                            "Epoch": epoch_count,
                            "Timestamp": timestamp,
                            "MVL Absolute Value": mvl_abs,
                        })

                        update_timer.reset(update_interval)
                        epoch_count += 1

                    if 'escape' in event.getKeys():
                        running = False

    except Exception as e:
        print("Error occurred:", e)

    finally:
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_excel(r"C:\Users\varsh\NFB_Spyder\real_time_mvl_results.xlsx", index=False)
            print("Results saved to 'real_time_mvl_results.xlsx'.")
        mywin.close()
        core.quit()

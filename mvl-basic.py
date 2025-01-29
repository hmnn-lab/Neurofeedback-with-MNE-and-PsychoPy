# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:19:01 2024

@author: varsh
"""

import numpy as np
import mne
from scipy.signal import hilbert
import pyxdf

# Load EEG data from the XDF file
xdf_fpath = "C:/Users/varsh/.spyder-py3/eye-closed-mc.xdf"
streams, header = pyxdf.load_xdf(xdf_fpath)

# Extract EEG stream
eeg_stream = streams[0]
data = eeg_stream["time_series"].T  # Transpose to get channels x samples
sfreq = float(eeg_stream["info"]["nominal_srate"][0])  # Sampling frequency

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

# Output the results
print(f"Normalized MVL (absolute value): {mvl_abs:.4f}")
print(f"Normalized MVL (angle in radians): {mvl_angle:.4f}")
print(f"Normalized MVL (angle in degrees): {np.degrees(mvl_angle):.2f}")



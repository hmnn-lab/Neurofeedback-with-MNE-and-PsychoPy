import numpy as np
from scipy.signal import coherence, welch
import matplotlib.pyplot as plt

# Simulation parameters
sfreq = 256  # Sampling frequency (Hz)
duration = 5  # Duration of the signal (seconds)
n_samples = sfreq * duration
freq_band_1 = (4, 7)  # Theta band
freq_band_2 = (8, 12)  # Alpha band
frequency_bands = [freq_band_1, freq_band_2]

# Generate synthetic EEG signals for two channels
time = np.arange(0, duration, 1 / sfreq)

# Channel O1: Sinusoidal signal with theta and alpha components + noise
signal_O1 = (
    np.sin(2 * np.pi * 6 * time) +  # Theta component
    0.5 * np.sin(2 * np.pi * 10 * time) +  # Alpha component
    0.2 * np.random.randn(len(time))  # Noise
)

# Channel F3: Similar signal with slight phase shift and different noise
signal_F3 = (
    np.sin(2 * np.pi * 6 * time + np.pi / 4) +  # Theta component with phase shift
    0.5 * np.sin(2 * np.pi * 10 * time) +  # Alpha component
    0.2 * np.random.randn(len(time))  # Noise
)

# Power spectrum analysis using Welch's method
f_O1, psd_O1 = welch(signal_O1, fs=sfreq, nperseg=sfreq)
f_F3, psd_F3 = welch(signal_F3, fs=sfreq, nperseg=sfreq)

# Coherence calculation
f_coh, coh = coherence(signal_O1, signal_F3, fs=sfreq, nperseg=sfreq)

# Calculate band-specific power and coherence
power_changes = []
coherence_data = {}

for band_num, band in enumerate(frequency_bands):
    low, high = band

    # Power calculations
    low_idx = np.where(f_O1 >= low)[0][0]
    high_idx = np.where(f_O1 <= high)[0][-1]
    power_band_O1 = psd_O1[low_idx:high_idx + 1].sum()
    power_band_F3 = psd_F3[low_idx:high_idx + 1].sum()
    total_power_O1 = psd_O1.sum()
    power_change = (power_band_O1 / total_power_O1) * 100
    power_changes.append(power_change)

    # Coherence calculations
    band_coh = np.mean(coh[(f_coh >= low) & (f_coh <= high)])
    coherence_data[f"Band {band_num + 1} Coherence"] = band_coh

# Display results
print("Power Changes (%):", power_changes)
print("Coherence Data:", coherence_data)

# Plot the results
plt.figure(figsize=(12, 6))

# Plot power spectra
plt.subplot(2, 1, 1)
plt.plot(f_O1, psd_O1, label="O1")
plt.plot(f_F3, psd_F3, label="F3")
plt.title("Power Spectral Density")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.legend()

# Plot coherence
plt.subplot(2, 1, 2)
plt.plot(f_coh, coh, label="Coherence")
plt.title("Magnitude Squared Coherence")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Coherence")
plt.axvspan(freq_band_1[0], freq_band_1[1], color='red', alpha=0.2, label="Theta Band")
plt.axvspan(freq_band_2[0], freq_band_2[1], color='blue', alpha=0.2, label="Alpha Band")
plt.legend()

plt.tight_layout()
plt.show()

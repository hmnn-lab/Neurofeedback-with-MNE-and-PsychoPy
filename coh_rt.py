import numpy as np
import scipy.signal
import scipy.linalg

def bandpass_filter(data, sfreq, freq_band, order=4):
    """Applies a bandpass filter to extract a specific frequency band."""
    nyquist = 0.5 * sfreq
    low, high = freq_band[0] / nyquist, freq_band[1] / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return scipy.signal.filtfilt(b, a, data)

def compute_msc(eeg_band1, eeg_band2, sfreq, nperseg=256, noverlap=128):
    """
    Computes MSC between two filtered EEG signals using the correct formula.
    Returns a single MSC value (between 0 and 1).
    """
    f, Pxx = scipy.signal.welch(eeg_band1, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
    f, Pyy = scipy.signal.welch(eeg_band2, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
    f, Pxy = scipy.signal.csd(eeg_band1, eeg_band2, fs=sfreq, nperseg=nperseg, noverlap=noverlap)

    # Convert Pxx to a proper 2D diagonal matrix
    Pxx = np.diag(Pxx)  # Shape: (129,129)

    # Convert Pxy to a column vector
    Pxy = Pxy[:, np.newaxis]  # Shape: (129,1)

    # Compute pseudo-inverse of Pxx (size: 129x129)
    Pxx_inv = np.linalg.pinv(Pxx)

    # Corrected MSC computation
    numerator = np.real(np.conj(Pxy).T @ Pxx_inv @ Pxy)  # Ensure real values
    denominator = np.real(Pyy).sum()  # Correctly normalize with Pyy

    msc = np.abs(numerator) / denominator  # Ensure value is between 0 and 1

    return float(np.clip(msc, 0, 1))  # Clip to valid range

# -------- User Input for Frequency Bands --------
sfreq = 250  # Sampling frequency
freq_band1 = (8, 12)   # Alpha band
freq_band2 = (15, 30)  # Beta band

# -------- Simulated EEG Data --------
np.random.seed(42)
n_samples = 1024  
eeg_data = np.random.randn(n_samples)

# Bandpass filter to extract frequency bands
eeg_band1 = bandpass_filter(eeg_data, sfreq, freq_band1)
eeg_band2 = bandpass_filter(eeg_data, sfreq, freq_band2)

# Compute MSC between the two bands
msc_value = compute_msc(eeg_band1, eeg_band2, sfreq)

# -------- Output --------
print(f"MSC between {freq_band1} Hz and {freq_band2} Hz: {msc_value:.3f}")

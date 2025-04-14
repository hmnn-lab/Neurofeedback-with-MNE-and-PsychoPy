import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

def compute_msc(signal1, signal2, fs, nperseg=256, noverlap=128):
    """
    Compute magnitude squared coherence (MSC) between two signals using Welch's method.
    
    Parameters:
        signal1, signal2 : array-like
            Input signals (1D arrays).
        fs : float
            Sampling frequency.
        nperseg : int, optional
            Length of each segment for Welch's method.
        noverlap : int, optional
            Number of points to overlap between segments.
    
    Returns:
        f : ndarray
            Array of sample frequencies.
        msc : ndarray
            Magnitude squared coherence values for each frequency.
    """
    # Compute power spectral density for each signal
    f, Pxx = scipy.signal.welch(signal1, fs=fs, nperseg=nperseg, noverlap=noverlap)
    f, Pyy = scipy.signal.welch(signal2, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    # Compute cross-spectral density between the two signals
    f, Pxy = scipy.signal.csd(signal1, signal2, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    # Compute MSC using the formula: |Pxy|^2 / (Pxx * Pyy)
    msc = np.abs(Pxy)**2 / (Pxx * Pyy)
    return f, msc

# -------------------------
# Example usage:
# -------------------------
fs = 250        # Sampling frequency in Hz
n_samples = 1024  # Number of samples
np.random.seed(42)

# Generate two example signals:
# signal2 is partly coherent with signal1
signal1 = np.random.randn(n_samples)
signal2 = 0.7 * signal1 + 0.3 * np.random.randn(n_samples)

# Compute MSC
frequencies, msc_values = compute_msc(signal1, signal2, fs)

# Plot the MSC vs. Frequency
plt.figure(figsize=(8, 4))
plt.plot(frequencies, msc_values, label='MSC')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude Squared Coherence')
plt.title('Magnitude Squared Coherence between Signal 1 and Signal 2')
plt.legend()
plt.grid(True)
plt.show()

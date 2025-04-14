import numpy as np
import scipy.linalg

def compute_msc(Pxy, Pxx, Pyy):
    """
    Computes the Magnitude-Squared Coherence (MSC) based on the given formula.

    Parameters:
    - Pxy: (m, ) complex array, Cross-Power Spectral Density (CPSD) vector between inputs and output
    - Pxx: (m, m) complex array, Power Spectral Density (PSD) and CPSD matrix of the inputs
    - Pyy: scalar, PSD of the output signal

    Returns:
    - MSC value for a given frequency
    """
    Pxx_inv = np.linalg.pinv(Pxx)  # Compute the pseudo-inverse of Pxx
    numerator = np.conj(Pxy).T @ Pxx_inv @ Pxy  # Pxy^H * Pxx^-1 * Pxy
    msc = np.abs(numerator) / Pyy  # Final MSC calculation
    return msc

# Example PSD and CPSD values
m = 2  # Number of input signals
Pxy = np.array([1+2j, 2-1j])  # Example cross-power spectral density
Pxx = np.array([[2+1j, 0.5-0.3j], [0.5+0.3j, 3+0.5j]])  # Example PSD & CPSD matrix
Pyy = 4.5  # Example PSD of output

msc_value = compute_msc(Pxy, Pxx, Pyy)
print("Magnitude-Squared Coherence (MSC):", msc_value)

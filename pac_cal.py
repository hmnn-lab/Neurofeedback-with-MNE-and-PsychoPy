import pandas as pd
import numpy as np
from scipy.signal import butter, freqz
import mne
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Endpoint Corrected Hilbert Transform
def echt(xr, filt_lf, filt_hf, Fs, n=None):
    if not np.isrealobj(xr):
        logging.warning("Ignoring imaginary part of input signal.")
        xr = np.real(xr)
    if n is None:
        n = len(xr)
    
    # Validate inputs
    if not all(isinstance(x, (int, float)) for x in [filt_lf, filt_hf, Fs]):
        logging.error(f"Non-numeric inputs: filt_lf={filt_lf}, filt_hf={filt_hf}, Fs={Fs}")
        raise ValueError("Filter parameters must be numeric")
    if Fs <= 0:
        logging.error(f"Invalid sampling frequency: Fs={Fs}")
        raise ValueError("Sampling frequency must be positive")
    if filt_lf < 0 or filt_hf > Fs/2 or filt_lf >= filt_hf:
        logging.error(f"Invalid frequency range: filt_lf={filt_lf}, filt_hf={filt_hf}, Fs/2={Fs/2}")
        raise ValueError("Frequencies must satisfy 0 <= filt_lf < filt_hf <= Fs/2")
    
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
    try:
        b, a = butter(filt_order, [filt_lf / (Fs / 2), filt_hf / (Fs / 2)], btype='bandpass')
        if b is None or a is None:
            logging.error("Butterworth filter returned None")
            raise ValueError("Failed to create Butterworth filter")
    except Exception as e:
        logging.error(f"Error in butter filter: {str(e)}")
        raise
    T = 1 / Fs * n
    filt_freq = np.fft.fftfreq(n, d=1 / Fs)
    filt_coeff = freqz(b, a, worN=filt_freq, fs=Fs)[1]
    x *= filt_coeff
    analytic_signal = np.fft.ifft(x)
    phase = np.angle(analytic_signal)
    amplitude = np.abs(analytic_signal)
    return analytic_signal, phase, amplitude

import pandas as pd
import numpy as np
from scipy.signal import butter, freqz
import mne
import os
import logging
from datetime import datetime

def pac_cal(
    epoch,
    ch_name,
    epoch_count,
    band_name_1,
    band_name_2,
    low_freq_1,
    high_freq_1,
    low_freq_2,
    high_freq_2,
    output_path=None
):
    """
    Calculate Phase-Amplitude Coupling (PAC) using Endpoint Corrected Hilbert Transform (ECHT).

    Args:
        epoch (mne.Epochs): MNE Epochs object containing a single epoch of EEG data.
        ch_name (str): Name of the EEG channel to analyze.
        epoch_count (int): Epoch number for result tracking.
        band_name_1 (str): Name of the phase frequency band (e.g., 'Theta').
        band_name_2 (str): Name of the amplitude frequency band (e.g., 'Gamma').
        low_freq_1 (float): Lower frequency bound for phase band.
        high_freq_1 (float): Upper frequency bound for phase band.
        low_freq_2 (float): Lower frequency bound for amplitude band.
        high_freq_2 (float): Upper frequency bound for amplitude band.
        output_path (str, optional): Path to save Excel file with results. If None, results are not saved.

    Returns:
        float: Mean Vector Length (MVL) absolute value for PAC, or 0.0 if computation fails.
    """
    # Input validation
    if not isinstance(epoch, mne.Epochs) or len(epoch) != 1:
        logging.error(f"Expected single mne.Epochs object, got {len(epoch)} epochs")
        return 0.0
    if ch_name not in epoch.ch_names:
        logging.error(f"Invalid channel: {ch_name}. Available: {epoch.ch_names}")
        return 0.0
    if epoch.get_data().size == 0:
        logging.error("Empty epoch data")
        return 0.0
    if not (0 <= low_freq_1 < high_freq_1 <= 50 and 0 <= low_freq_2 < high_freq_2 <= 50):
        logging.error("Invalid frequency range. Must be 0 <= low < high <= 50 Hz")
        return 0.0

    # Extract data and sampling frequency
    data = epoch.get_data(picks=[ch_name])[0]
    sfreq = epoch.info['sfreq']
    if sfreq <= 0:
        logging.error(f"Invalid sampling frequency: sfreq={sfreq}")
        return 0.0
    
    # Compute phase and amplitude using ECHT
    try:
        _, low_phase, _ = echt(data[0], low_freq_1, high_freq_1, sfreq)
        _, _, high_amp = echt(data[0], low_freq_2, high_freq_2, sfreq)
    except ValueError as e:
        logging.error(f"ECHT failed: {str(e)}")
        return 0.0

    # MVL calculation
    N = len(low_phase)
    if N == 0 or len(high_amp) != N:
        logging.error("Invalid phase or amplitude data length")
        return 0.0
    complex_mvl = (1 / np.sqrt(N)) * np.sum(high_amp * np.exp(1j * low_phase)) / np.sqrt(np.sum(high_amp**2))
    mvl_abs = np.abs(complex_mvl)

    # Save results if output_path is provided
    if output_path:
        try:
            results = {
                "Epoch": epoch_count,
                "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Channel": ch_name,
                f"{band_name_1}-{band_name_2} MVL": mvl_abs
            }
            df = pd.DataFrame([results])
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if os.path.exists(output_path):
                existing_df = pd.read_excel(output_path)
                df = pd.concat([existing_df, df], ignore_index=True)
            df.to_excel(output_path, index=False)
            logging.info(f"Saved PAC results to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save results to {output_path}: {str(e)}")

    return mvl_abs
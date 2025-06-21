import os
import numpy as np
import pandas as pd
import mne
import logging

def compute_band_auc(
    epoch,
    ch_names,
    epoch_count,
    band_name="Alpha",
    low_freq=8.0,
    high_freq=12.0,
    output_path=None
):
    """
    Compute the Area Under the Curve (AUC) of PSD for a single-epoch mne.Epochs object.

    Parameters:
    -----------
    epoch : mne.Epochs
        Single epoch to compute PSD for.
    ch_names : list
        List of channel names to compute PSD for.
    epoch_count : int
        Current epoch number for logging.
    band_name : str
        Name of frequency band (e.g., "Alpha").
    low_freq : float
        Lower frequency bound for the band.
    high_freq : float
        Upper frequency bound for the band.
    output_path : str, optional
        Path to save PSD results. If None, results are not saved.

    Returns:
    --------
    power_change_values : float or list
        Power change (%) of band relative to total power (scalar for single channel, list for multiple).
    band_auc_values : list
        Absolute band AUC (power) for each channel.
    """
    band_freqs = {
        "Delta": (1.0, 4.0),
        "Theta": (4.0, 8.0),
        "Alpha": (8.0, 12.0),
        "Beta": (12.0, 30.0),
        "Gamma": (30.0, 100.0)
    }

    if band_name in band_freqs and low_freq == 8.0 and high_freq == 12.0:
        low_freq, high_freq = band_freqs[band_name]
        logging.info(f"Using default frequencies for {band_name}: {low_freq}-{high_freq} Hz")

    if not isinstance(epoch, mne.Epochs) or len(epoch) != 1:
        logging.error(f"Expected single mne.Epochs object, got {len(epoch)} epochs")
        return [], []

    if epoch.get_data().size == 0:
        logging.error("Empty epoch data")
        return [], []

    try:
        valid_channels = [ch for ch in ch_names if ch in epoch.ch_names]
        if not valid_channels:
            logging.error(f"No valid channels. Requested: {ch_names}, Available: {epoch.ch_names}")
            return [], []

        if not (0 <= low_freq < high_freq <= 50):
            logging.error(f"Invalid frequency range: {low_freq}-{high_freq} Hz. Must be 0 <= low < high <= 50 Hz")
            return [], []

        psd_obj = epoch.compute_psd(
            method='welch',
            fmin=0,
            fmax=50,
            picks=valid_channels,
            n_fft=256,
            n_per_seg=256,
            average=False
        )
        psd, freqs = psd_obj.get_data(return_freqs=True)
        logging.debug(f"PSD shape: {psd.shape}, Freq range: {freqs[0]:.2f}-{freqs[-1]:.2f} Hz")

        if psd.ndim == 3:
            power = psd[0]  # First epoch: (n_channels, n_freqs)
        else:
            power = np.squeeze(psd)
            if power.ndim == 1:
                power = power.reshape(1, -1)

        if power.shape[1] != len(freqs):
            logging.warning("Power freq dim mismatch, transposing power array")
            power = power.T

        power_change_values = []
        band_auc_values = []

        for ch_idx in range(len(valid_channels)):
            total_auc = np.trapz(power[ch_idx], freqs)
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_freqs = freqs[band_mask]

            if band_freqs.size == 0:
                logging.warning(f"No frequencies in band {low_freq}-{high_freq} Hz")
                band_auc = 0.0
            else:
                band_auc = np.trapz(power[ch_idx][band_mask], band_freqs)

            power_change = (band_auc / total_auc) * 100 if total_auc != 0 else 0.0
            power_change = 0.0 if np.isnan(power_change) else power_change

            power_change_values.append(power_change)
            band_auc_values.append(band_auc)

            logging.debug(
                f"Channel {valid_channels[ch_idx]}: Total AUC={total_auc:.4f}, "
                f"Band AUC={band_auc:.4f}, Power Change={power_change:.2f}%"
            )

        # Handle single-channel case
        if len(valid_channels) == 1:
            power_change_values = float(power_change_values[0])
            band_auc_values = [float(band_auc_values[0])]

        if output_path:
            try:
                power_data = {
                    'Epoch Count': [epoch_count],
                    **{f'Total power AUC ({ch})': [np.trapz(power[i], freqs)] for i, ch in enumerate(valid_channels)},
                    **{f'{band_name} band power AUC ({ch})': [band_auc_values[i] if isinstance(band_auc_values, list) else band_auc_values] for i, ch in enumerate(valid_channels)},
                    **{f'Power Change (%) ({ch})': [power_change_values[i] if isinstance(power_change_values, list) else power_change_values] for i, ch in enumerate(valid_channels)}
                }
                df = pd.DataFrame(power_data)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    existing_df = pd.read_excel(output_path, engine='openpyxl')
                    df = pd.concat([existing_df, df], ignore_index=True)
                df.to_excel(output_path, index=False)
                logging.info(f"Saved PSD data to {output_path}")
            except Exception as e:
                logging.error(f"Failed to save results to {output_path}: {str(e)}")

        return power_change_values, band_auc_values

    except Exception as e:
        logging.error(f"Error computing PSD: {str(e)}")
        return 0.0, 0.0
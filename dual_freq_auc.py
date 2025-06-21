import os
import numpy as np
import pandas as pd
import mne
import logging

def dual_freq_auc(
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
    Compute the power ratio (AUC band 1 / AUC band 2) for a single channel in a single-epoch mne.Epochs object.

    Parameters:
    -----------
    epoch : mne.Epochs
        Single epoch to compute PSD for.
    ch_name : str
        Single channel name to compute PSD for (from GUI selection).
    epoch_count : int
        Current epoch number for logging.
    band_name_1 : str
        Name of the first frequency band (e.g., "Alpha").
    band_name_2 : str
        Name of the second frequency band (e.g., "Beta").
    low_freq_1 : float
        Lower frequency bound for first band.
    high_freq_1 : float
        Upper frequency bound for first band.
    low_freq_2 : float
        Lower frequency bound for second band.
    high_freq_2 : float
        Upper frequency bound for second band.
    output_path : str, optional
        Path to save Excel results. If None, results are not saved.

    Returns:
    --------
    power_ratio : float
        Power ratio (band_auc_1 / band_auc_2) for the selected channel, or 0.0 if computation fails.
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

    try:
        # Compute PSD
        psd_obj = epoch.compute_psd(
            method='welch',
            fmin=0,
            fmax=50,
            picks=[ch_name],
            n_fft=256,
            n_per_seg=256,
            average=False
        )
        psd, freqs = psd_obj.get_data(return_freqs=True)
        logging.debug(f"PSD shape: {psd.shape}, Freq range: {freqs[0]:.2f}-{freqs[-1]:.2f} Hz")

        # Handle PSD data shape
        power = np.squeeze(psd)  # Shape: (n_channels=1, n_freqs) -> (n_freqs,)
        if power.ndim == 2:
            power = power[0]

        # Compute AUC for both bands
        # Band 1
        band_mask_1 = (freqs >= low_freq_1) & (freqs <= high_freq_1)
        band_freqs_1 = freqs[band_mask_1]
        band_auc_1 = np.trapz(power[band_mask_1], band_freqs_1) if band_freqs_1.size > 0 else 0.0
        if band_freqs_1.size == 0:
            logging.warning(f"No frequencies in band {band_name_1}: {low_freq_1}-{high_freq_1} Hz")

        # Band 2
        band_mask_2 = (freqs >= low_freq_2) & (freqs <= high_freq_2)
        band_freqs_2 = freqs[band_mask_2]
        band_auc_2 = np.trapz(power[band_mask_2], band_freqs_2) if band_freqs_2.size > 0 else 0.0
        if band_freqs_2.size == 0:
            logging.warning(f"No frequencies in band {band_name_2}: {low_freq_2}-{high_freq_2} Hz")

        # Compute power ratio
        power_ratio = (band_auc_1 / band_auc_2) if band_auc_2 != 0 else 0.0

        # Log values for debugging
        logging.debug(
            f"Epoch {epoch_count}: {band_name_1} AUC = {band_auc_1:.4f}, "
            f"{band_name_2} AUC = {band_auc_2:.4f}, Power Ratio = {power_ratio:.4f}"
        )

        # Save to Excel if output_path is provided
        if output_path:
            try:
                power_data = {
                    'Epoch Count': [epoch_count],
                    f'{band_name_1} band power AUC ({ch_name})': [band_auc_1],
                    f'{band_name_2} band power AUC ({ch_name})': [band_auc_2],
                    f'Power Ratio ({ch_name})': [power_ratio]
                }
                df = pd.DataFrame(power_data)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    existing_df = pd.read_excel(output_path, engine='openpyxl')
                    df = pd.concat([existing_df, df], ignore_index=True)
                df.to_excel(output_path, index=False)
                logging.info(f"Saved dual frequency ratio data to {output_path}")
            except Exception as e:
                logging.error(f"Failed to save results to {output_path}: {str(e)}")

        return power_ratio

    except Exception as e:
        logging.error(f"Error computing PSD: {str(e)}")
        return 0.0
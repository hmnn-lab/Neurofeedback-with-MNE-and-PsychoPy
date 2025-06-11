import os
import numpy as np
import pandas as pd
import mne
import logging

def compute_band_auc(
    epoch,
    ch_names,
    epoch_count,
    output_path=r"C:\Users\Admin\Documents\BCI\Neurofeedback Varsha\Current_working-code\psd_results\band_power_band1.xlsx",
    band_name="Alpha",
    low_freq=8.0,
    high_freq=12.0
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
    output_path : str
        Path to save PSD results.
    band_name : str
        Name of the frequency band (e.g., "Alpha").
    low_freq : float
        Lower frequency bound.
    high_freq : float
        Upper frequency bound.

    Returns:
    --------
    power_change : float
        Power change (%) of band relative to total power (first channel).
    band_auc : float
        Absolute band AUC (power) of the first channel.
    """
    # Map band names to default frequency ranges
    band_freqs = {
        "Delta": (1.0, 4.0),
        "Theta": (4.0, 8.0),
        "Alpha": (8.0, 12.0),
        "Beta": (12.0, 30.0),
        "Gamma": (30.0, 100.0)
    }
    
    # Use default frequencies for the band if low_freq/high_freq are default (8.0, 12.0)
    if band_name in band_freqs and (low_freq == 8.0 and high_freq == 12.0):
        low_freq, high_freq = band_freqs[band_name]
        print(f"Using default frequencies for {band_name}: {low_freq}-{high_freq} Hz")

    if len(epoch) != 1:
        print(f"Expected 1 epoch, got {len(epoch)}. Skipping.")
        return 0.0, 0.0

    if epoch.get_data().size == 0:
        print("Empty epoch data. Skipping.")
        return 0.0, 0.0

    try:
        # Validate channels
        valid_channels = [ch for ch in ch_names if ch in epoch.ch_names]
        if not valid_channels:
            print(f"No valid channels. Requested: {ch_names}, Available: {epoch.ch_names}")
            return 0.0, 0.0

        # Compute PSD
        psd_obj = epoch.compute_psd(
            method='welch',
            fmin=0,
            fmax=50,
            picks=ch_names,
            n_fft=256,
            n_per_seg=256,
            average=False
        )
        psd, freqs = psd_obj.get_data(return_freqs=True)
        logging.debug(f"PSD shape: {psd.shape}, Freq range: {freqs[0]:.2f}-{freqs[-1]:.2f} Hz")

        # Handle PSD data shape
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
                band_auc = 0
            else:
                band_auc = np.trapz(power[ch_idx][band_mask], band_freqs)

            power_change = (band_auc / total_auc) * 100 if total_auc != 0 else 0
            power_change = 0 if np.isnan(power_change) else power_change

            power_change_values.append(power_change)
            band_auc_values.append(band_auc)

            logging.debug(
                f"Channel {valid_channels[ch_idx]}: Total AUC={total_auc:.4f}, "
                f"Band AUC={band_auc:.4f}, Power Change={power_change:.2f}%"
            )

        # Convert lists to NumPy arrays
        #power_change_values = np.array(power_change_values)
        #band_auc_values = np.array(band_auc_values)

        # Save to Excel
        power_data = {
            'Epoch Count': [epoch_count],
            **{f'Total power AUC ({ch})': [np.trapz(power[i], freqs)] for i, ch in enumerate(valid_channels)},
            **{f'{band_name} band power AUC ({ch})': [band_auc_values[i]] for i, ch in enumerate(valid_channels)},
            **{f'Power Change (%) ({ch})': [power_change_values[i]] for i, ch in enumerate(valid_channels)}
        }

        df = pd.DataFrame(power_data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            existing_df = pd.read_excel(output_path, engine='openpyxl')
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_excel(output_path, index=False)
        logging.info(f"Appended PSD data to {output_path}")
        print(power_change_values, band_auc_values)
        return power_change_values, band_auc_values

    except Exception as e:
        logging.error(f"Error computing PSD: {str(e)}")
        return 0.0, 0.0
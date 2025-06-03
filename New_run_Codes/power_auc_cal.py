import numpy as np
import pandas as pd
import mne
import os

def compute_band_auc_epochs(
    epoch,
    ch_names,
    epoch_count,
    auc_values,
    output_path=r"C:\Users\varsh\NFB_Spyder\psd_results\band_power_band1.xlsx",
    band_name=None,
    low_freq=8.0,
    high_freq=12.0
):
    """
    Compute the Area Under the Curve (AUC) of PSD for a single epoch in the specified frequency band.
    
    Parameters:
    -----------
    epoch : mne.Epochs
        Single epoch to compute PSD for.
    ch_names : list
        List of channel names to compute PSD for (same as feed_ch_names).
    epoch_count : int
        Current epoch number for tracking.
    auc_values : list
        List to store power change values across epochs (updated in-place with first channel's power_change).
    output_path : str, optional
        Path to save PSD results as Excel file (default: "./psd_results/alpha_power.xlsx").
    band_name : str, optional
        Name of the frequency band (default: "Alpha").
    low_freq : float, optional
        Lower frequency bound (default: 8.0 Hz).
    high_freq : float, optional
        Upper frequency bound (default: 12.0 Hz).
        
    Returns:
    --------
    tuple
        (power_change, output_path) where power_change is the power change (%) for the first channel,
        and output_path is the Excel file path.
    """
    n_channels = len(ch_names)
    total_auc_values = []
    band_auc_values = []
    power_change_values = []

    # Compute PSD for the epoch
    psd = epoch.compute_psd(
        tmin=epoch.tmin, tmax=epoch.tmax, picks=ch_names, method='welch', average=False
    )
    power = np.squeeze(psd.get_data())  # Shape: (n_channels, n_freqs) or (n_freqs,) for single channel
    if power.ndim == 1:
        power = power.reshape(1, -1)  # Reshape to (1, n_freqs) for single channel
    freq = psd.freqs

    # Calculate AUC and power change for each channel
    power_changes = []
    for ch_idx in range(n_channels):
        total_auc = np.trapz(power[ch_idx], freq)
        band_mask = (freq >= low_freq) & (freq <= high_freq)
        band_freqs = freq[band_mask]
        band_auc = np.trapz(power[ch_idx][band_mask], band_freqs) if band_freqs.size > 0 else 0
        power_change = (band_auc / total_auc) * 100 if total_auc != 0 else 0
        power_change = power_change if not np.isnan(power_change) else 0

        total_auc_values.append(total_auc)
        band_auc_values.append(band_auc)
        power_change_values.append(power_change)
        power_changes.append(power_change)

    # Save results to Excel (append mode) for all channels
    if power_change_values:
        power_data = {
            'Epoch Count': [epoch_count],
            **{f'Total power AUC ({ch_name})': [total_auc_values[i]] for i, ch_name in enumerate(ch_names)},
            **{f'{band_name} band power AUC ({ch_name})': [band_auc_values[i]] for i, ch_name in enumerate(ch_names)},
            **{f'Power Change (%) ({ch_name})': [power_change_values[i]] for i, ch_name in enumerate(ch_names)}
        }
        df = pd.DataFrame(power_data)
        
        # Append to Excel file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            existing_df = pd.read_excel(output_path, engine='openpyxl')
            df = pd.concat([existing_df, df], ignore_index=True)
        df.to_excel(output_path, index=False)
        print(f"Data appended to {output_path}")

    # Update auc_values with the power change of the first channel
    power_change = power_changes[0] if power_changes else 0.0
    auc_values.append(power_change)

    return power_change, output_path

import os
import numpy as np
import pandas as pd
import mne

def compute_band_auc(
    epoch,
    ch_names,
    epoch_count,
    output_path=r"C:\Users\varsh\NFB_Spyder\psd_results\band_power_band1.xlsx",
    band_name="Alpha",
    low_freq=8.0,
    high_freq=12.0
):
    """
    Compute the Area Under the Curve (AUC) of PSD for a single-epoch mne.Epochs object.

    Returns:
    --------
    power_change : float
        Power change (%) of band relative to total power (first channel).
    band_auc : float
        Absolute band AUC (power) of the first channel.
    """

    if len(epoch) != 1:
        print(f"Warning: Expected 1 epoch, got {len(epoch)}. Skipping.")
        return 0.0, 0.0

    if epoch.get_data().size == 0:
        print("Warning: Empty epoch data. Skipping.")
        return 0.0, 0.0

    try:
        psd = epoch.compute_psd(method='welch', picks=ch_names, average=False)
        print(f"PSD shape: {psd.get_data().shape}, freq shape: {psd.freqs.shape}")
    except Exception as e:
        print(f"Error computing PSD: {e}")
        return 0.0, 0.0

    data = psd.get_data()  # shape (n_epochs, n_channels, n_freqs) or (n_channels, n_freqs)
    if data.ndim == 3:
        power = data[0]  # first epoch PSD (n_channels, n_freqs)
    else:
        power = np.squeeze(data)
        if power.ndim == 1:
            power = power.reshape(1, -1)

    freq = psd.freqs

    print(f"Epoch {epoch_count}: PSD shape {power.shape}, Freq range {freq[0]:.2f}-{freq[-1]:.2f} Hz")

    # If power shape doesn't match freq length, try transpose
    if power.shape[1] != freq.shape[0]:
        print("Power freq dim mismatch, transposing power array")
        power = power.T
        print(f"Power shape after transpose: {power.shape}")

    power_change_values = []
    band_auc_values = []

    for ch_idx in range(len(ch_names)):
        total_auc = np.trapz(power[ch_idx], freq)
        band_mask = (freq >= low_freq) & (freq <= high_freq)
        band_freqs = freq[band_mask]

        if band_freqs.size == 0:
            print(f"Warning: No frequencies found in band {low_freq}-{high_freq} Hz")
            band_auc = 0
        else:
            band_auc = np.trapz(power[ch_idx][band_mask], band_freqs)

        power_change = (band_auc / total_auc) * 100 if total_auc != 0 else 0
        power_change = 0 if np.isnan(power_change) else power_change

        power_change_values.append(power_change)
        band_auc_values.append(band_auc)

        print(f"Channel {ch_names[ch_idx]}: Total AUC={total_auc:.4f}, Band AUC={band_auc:.4f}, Power Change={power_change:.2f}%")

    # Save to Excel
    power_data = {
        'Epoch Count': [epoch_count],
        **{f'Total power AUC ({ch})': [np.trapz(power[i], freq)] for i, ch in enumerate(ch_names)},
        **{f'{band_name} band power AUC ({ch})': [band_auc_values[i]] for i, ch in enumerate(ch_names)},
        **{f'Power Change (%) ({ch})': [power_change_values[i]] for i, ch in enumerate(ch_names)}
    }

    df = pd.DataFrame(power_data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        existing_df = pd.read_excel(output_path, engine='openpyxl')
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_excel(output_path, index=False)
    print(f"Appended PSD data to {output_path}")

    return power_change_values[0], band_auc_values[0]

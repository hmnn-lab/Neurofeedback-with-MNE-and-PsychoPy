import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging
import mne

def compute_baseline_psd(raw_or_file):
    """
    Compute PSD AUC for frequency bands from an MNE Raw object or EEG FIF file,
    save results to Excel, and generate a PSD plot.

    Args:
        raw_or_file (str or mne.io.Raw): Path to EEG FIF file or MNE Raw object.

    Returns:
        tuple: (DataFrame, path to .xlsx file, path to plot file)
    """
    logging.basicConfig(level=logging.INFO)

    # Determine if input is file or Raw object
    if isinstance(raw_or_file, mne.io.BaseRaw):
        raw = raw_or_file.copy()
        base_name = "raw_object"
        save_dir = os.getcwd()
        logging.info("Processing raw MNE object in memory")
    elif isinstance(raw_or_file, str) and os.path.isfile(raw_or_file):
        logging.info(f"Processing file {raw_or_file}")
        raw = mne.io.read_raw_fif(raw_or_file, preload=True)
        raw.pick_types(eeg=True)
        base_name = os.path.splitext(os.path.basename(raw_or_file))[0]
        save_dir = os.path.dirname(os.path.abspath(raw_or_file))
    else:
        raise ValueError("Input must be an MNE Raw object or path to a valid FIF file.")

    raw.pick_types(eeg=True)
    ch_names = raw.info['ch_names']

    # Compute PSD using Welch
    psd = raw.compute_psd(method='welch', average='mean')
    power = psd.get_data()
    freq = psd.freqs

    # Define frequency bands
    bands = {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 45)
    }

    # Calculate AUC for each channel and frequency band
    results = []
    for ch_idx, ch_name in enumerate(ch_names):
        total_auc = np.trapz(power[ch_idx], freq)
        results.append({
            'Freq band name': 'Total',
            'Freq range (Hz)': 'All',
            'PSD (AUC)': total_auc,
            'Channel': ch_name
        })
        for band_name, (fmin, fmax) in bands.items():
            band_mask = (freq >= fmin) & (freq <= fmax)
            band_auc = np.trapz(power[ch_idx][band_mask], freq[band_mask]) if band_mask.any() else 0
            results.append({
                'Freq band name': band_name,
                'Freq range (Hz)': f'{fmin}-{fmax}',
                'PSD (AUC)': band_auc,
                'Channel': ch_name
            })

    df = pd.DataFrame(results)

    # Save Excel and plot only if we have a valid save directory (file input)
    if save_dir:
        df_path = os.path.join(save_dir, f"{base_name}.xlsx")
        df.to_excel(df_path, index=False)
        logging.info(f"Saved PSD AUC to {df_path}")

        # Plot PSD
        plt.figure(figsize=(10, 6))
        cmap = cm.get_cmap('tab20', len(ch_names))
        for ch_idx, ch_name in enumerate(ch_names):
            plt.plot(freq, power[ch_idx], label=ch_name, color=cmap(ch_idx), linewidth=1.5)

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (µV²/Hz)')
        plt.title(f'PSD for {base_name}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=8, loc='upper right', ncol=2)
        plt.tight_layout()

        plot_path = os.path.join(save_dir, f"{base_name}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logging.info(f"Saved PSD plot to {plot_path}")
    else:
        df_path = None
        plot_path = None

    return df, df_path, plot_path

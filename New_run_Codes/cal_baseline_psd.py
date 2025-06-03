import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import logging

def compute_psd_auc(fif_file, file_num):
    """
    Compute PSD AUC from an EEG FIF file and save results with consistent naming.

    Args:
        fif_file (str): Path to the EEG FIF file.
        file_num (str): File number (e.g., '001') for consistent naming.

    Returns:
        tuple: (DataFrame, path to .xlsx file, path to plot file)
    """
    logging.debug(f"Starting compute_psd_auc for {fif_file}")
    try:
        raw = mne.io.read_raw_fif(fif_file, preload=True)
        raw.pick_types(eeg=True)
        sfreq = raw.info['sfreq']
        ch_names = raw.info['ch_names']

        psd = raw.compute_psd(method='welch', average='mean')
        power = psd.get_data()
        if power.ndim == 1:
            power = power.reshape(len(ch_names), -1)
        freq = psd.freqs

        bands = {
            'Delta': (1, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 45)
        }

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
        script_dir = os.path.dirname(os.path.abspath(fif_file))
        df_path = os.path.join(script_dir, f"baseline_{file_num}.xlsx")
        df.to_excel(df_path, index=False)
        logging.debug(f"Saved PSD AUC to {df_path}")

        # Plot PSD
        plt.figure(figsize=(12, 6))
        cmap = cm.get_cmap('tab20', len(ch_names))
        for ch_idx, ch_name in enumerate(ch_names):
            plt.plot(freq, power[ch_idx], label=ch_name, color=cmap(ch_idx), linewidth=1.5)

        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.ylabel('Power Spectral Density (µV²/Hz)', fontsize=12)
        plt.title(f'PSD for baseline_{file_num}', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=8, loc='upper right', ncol=2)
        plt.tight_layout()

        plot_path = os.path.join(script_dir, f"baseline_{file_num}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logging.debug(f"Saved PSD plot to {plot_path}")

        return df, df_path, plot_path
    except Exception as e:
        logging.error(f"compute_psd_auc failed: {e}")
        raise
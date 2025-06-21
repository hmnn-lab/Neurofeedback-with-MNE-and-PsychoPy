import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt # REMOVE THIS IMPORT
import logging
import mne
from scipy.signal import hilbert

# Configure logging at the module level for consistency
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import helper functions
from power_auc_cal import compute_band_auc
from dual_freq_auc import dual_freq_auc
from pac_cal import pac_cal

# --- Constants ---
EEG_STANDARD_BANDS = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (12, 30),
    'Gamma': (30, 100)
}

def compute_baseline_modality(raw_or_file, modality, channels,
                              freq_bands_numeric,
                              channels_2=None,
                              band_1_dict=None,
                              band_2_dict=None,
                              phase_freq_dict=None,
                              amp_freq_dict=None,
                              user_data_dir=None):
    """
    Computes baseline values for specified EEG modalities (PSD AUC, PSD Ratio, PAC, Coherence).
    Results are saved to Excel. Plotting functionality is entirely removed.
    Output files are named 'baseline_<modality>_XX' with incrementing numbers
    and stored in the `user_data_dir`.

    Args:
        raw_or_file (str or mne.io.Raw): Path to the EEG FIF file or an MNE Raw object.
        modality (str): The type of analysis to perform ('psd_auc', 'psd_ratio', 'pac', 'coh').
        channels (list of str): Primary EEG channel name(s) for analysis.
                                Expected: [ch_name] for PSD AUC, PSD Ratio, PAC.
                                Expected: [ch_name_1, ch_name_2] for Coherence.
        freq_bands_numeric (list of lists/tuples): Numeric frequency ranges required for computation.
                                                   Format: [[low1, high1], [low2, high2], ...].
                                                   - For 'psd_auc' or 'coh': [[fmin, fmax]] (single band)
                                                   - For 'psd_ratio' or 'pac': [[fmin1, fmax1], [fmin2, fmax2]] (two bands)
        channels_2 (list of str, optional): Secondary EEG channel name(s) for modalities like coherence.
                                            Used in conjunction with `channels` for two-channel analyses.
        band_1_dict (dict, optional): Dictionary for the first frequency band, containing 'name', 'low', 'high'.
                                      Used primarily for display/logging when modality is 'psd_ratio'.
        band_2_dict (dict, optional): Dictionary for the second frequency band, containing 'name', 'low', 'high'.
                                      Used primarily for display/logging when modality is 'psd_ratio'.
        phase_freq_dict (dict, optional): Dictionary for the phase frequency band, containing 'name', 'low', 'high'.
                                          Used primarily for display/logging when modality is 'pac'.
        amp_freq_dict (dict, optional): Dictionary for the amplitude frequency band, containing 'name', 'low', 'high'.
                                        Used primarily for display/logging when modality is 'pac'.
        user_data_dir (str, optional): Directory path for saving output files.
                                       If None, files are not saved to disk.

    Returns:
        tuple: (DataFrame with results, path to .xlsx file, path to plot file (always None)).
               Returns (None, None, None) on error.
    """
    # 1. Input Validation and Data Loading
    try:
        if isinstance(raw_or_file, mne.io.BaseRaw):
            raw = raw_or_file.copy()
            base_name_for_files = "raw_object_in_memory"
            logging.info("Processing raw MNE object passed in memory.")
        elif isinstance(raw_or_file, str):
            if not os.path.isfile(raw_or_file) or not raw_or_file.endswith('.fif'):
                raise ValueError(f"Invalid FIF file path or extension: {raw_or_file}")
            if not os.access(raw_or_file, os.R_OK):
                raise PermissionError(f"Cannot read file: {raw_or_file}. Check permissions.")
            raw = mne.io.read_raw_fif(raw_or_file, preload=True, verbose='error')
            base_name_for_files = os.path.splitext(os.path.basename(raw_or_file))[0]
            logging.info(f"Loaded EEG data from file: {raw_or_file}")
        else:
            raise TypeError("`raw_or_file` must be an MNE Raw object or a path to a FIF file.")
    except Exception as e:
        logging.exception(f"Error loading raw data: {e}")
        return None, None, None

    raw.pick_types(eeg=True, exclude='bads')
    ch_names_in_raw = raw.info['ch_names']
    sfreq = raw.info['sfreq']

    # Validate primary channels
    if not isinstance(channels, list) or not channels:
        raise ValueError("`channels` must be a non-empty list of channel names.")
    if not all(ch in ch_names_in_raw for ch in channels):
        missing_chs = [ch for ch in channels if ch not in ch_names_in_raw]
        raise ValueError(f"Primary channel(s) not found in EEG data: {missing_chs}. Available: {ch_names_in_raw}")

    # Validate secondary channels for coherence if provided
    if modality == 'coh':
        if not isinstance(channels_2, list) or not channels_2 or len(channels_2) != 1:
            raise ValueError("For 'coh' modality, `channels_2` must be a list with exactly one channel name.")
        if not all(ch in ch_names_in_raw for ch in channels_2):
            missing_chs = [ch for ch in channels_2 if ch not in ch_names_in_raw]
            raise ValueError(f"Secondary channel(s) not found in EEG data: {missing_chs}. Available: {ch_names_in_raw}")
        all_channels_for_analysis = channels + channels_2
    else:
        all_channels_for_analysis = channels

    # 2. Validate and Prepare Frequency Bands
    if not isinstance(freq_bands_numeric, list) or not freq_bands_numeric:
        raise ValueError("`freq_bands_numeric` must be a non-empty list of [low, high] ranges.")

    validated_freq_ranges = []
    freq_band_names_for_display = []

    for i, fb_range in enumerate(freq_bands_numeric):
        if not (isinstance(fb_range, (list, tuple))) or len(fb_range) != 2:
            raise ValueError(f"Invalid frequency band format: Expected [low, high] or (low, high), got {fb_range}.")

        fmin, fmax = float(fb_range[0]), float(fb_range[1])

        if not (0 <= fmin < fmax):
            raise ValueError(f"Invalid frequency range: fmin ({fmin}) must be less than fmax ({fmax}) and non-negative for {fb_range}.")

        if fmax > sfreq / 2:
            raise ValueError(f"Frequency {fmax} Hz exceeds Nyquist limit ({sfreq/2} Hz) for band {fb_range}. Please ensure data is sampled correctly or band is within limits.")

        validated_freq_ranges.append((fmin, fmax))

        if modality == 'psd_ratio' and i == 0 and band_1_dict:
            freq_band_names_for_display.append(band_1_dict.get('name', f"{fmin}-{fmax}Hz"))
        elif modality == 'psd_ratio' and i == 1 and band_2_dict:
            freq_band_names_for_display.append(band_2_dict.get('name', f"{fmin}-{fmax}Hz"))
        elif modality == 'pac' and i == 0 and phase_freq_dict:
            freq_band_names_for_display.append(phase_freq_dict.get('name', f"Phase: {fmin}-{fmax}Hz"))
        elif modality == 'pac' and i == 1 and amp_freq_dict:
            freq_band_names_for_display.append(amp_freq_dict.get('name', f"Amplitude: {fmin}-{fmax}Hz"))
        else:
            found_name = next((name for name, val in EEG_STANDARD_BANDS.items() if val == (fmin, fmax)), f"{fmin}-{fmax}Hz")
            freq_band_names_for_display.append(found_name)

    # 3. Modality-Specific Validations
    expected_channel_count = 0
    expected_freq_band_count = 0

    if modality == 'psd_auc':
        expected_channel_count = 1
        expected_freq_band_count = 1
    elif modality == 'psd_ratio':
        expected_channel_count = 1
        expected_freq_band_count = 2
    elif modality == 'pac':
        expected_channel_count = 1
        expected_freq_band_count = 2
    elif modality == 'coh':
        expected_channel_count = 2
        expected_freq_band_count = 1
    else:
        raise ValueError(f"Unsupported or unrecognized modality: '{modality}'.")

    if len(channels) != 1 and modality not in ['coh']:
        raise ValueError(f"Modality '{modality}' requires exactly 1 primary channel, but {len(channels)} provided.")
    if modality == 'coh' and (len(channels) != 1 or len(channels_2) != 1):
        raise ValueError(f"Modality 'coh' requires exactly 1 primary channel and 1 secondary channel.")

    if len(validated_freq_ranges) != expected_freq_band_count:
        raise ValueError(f"Modality '{modality}' requires {expected_freq_band_count} frequency band(s), but {len(validated_freq_ranges)} provided.")

    # 4. Create MNE Epochs
    try:
        duration = raw.times[-1] - raw.times[0]
        min_f_for_duration_check = min(f[0] for f in validated_freq_ranges if f[0] > 0) if validated_freq_ranges else 1.0
        min_required_duration = 1.0 / min_f_for_duration_check if min_f_for_duration_check > 0 else 1.0

        if duration < min_required_duration:
            raise ValueError(f"Raw data duration ({duration:.2f} s) is too short for analysis. "
                             f"Minimum required duration based on lowest frequency ({min_f_for_duration_check} Hz) is {min_required_duration:.2f} s.")

        events = mne.make_fixed_length_events(raw, id=1, duration=duration)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=duration, baseline=None, preload=True, reject_by_annotation=False)
        logging.info(f"Created {len(epochs)} epoch(s) of {duration:.2f} seconds.")

        if len(raw.annotations) > 0:
            logging.warning(f"Raw data contains {len(raw.annotations)} annotations. Consider pre-processing them if they interfere with analysis.")

    except Exception as e:
        logging.exception(f"Error creating MNE epochs: {e}")
        return None, None, None

    # 5. Determine Unique File Name
    df_path = None
    plot_path = None # plot_path will always be None now
    file_name_prefix = None # Initialize to None in case user_data_dir is not set or fails

    if user_data_dir:
        try:
            os.makedirs(user_data_dir, exist_ok=True)
            file_counter = 1
            while True:
                current_file_name_prefix = f"baseline_{modality}_{file_counter:02d}"
                temp_df_path = os.path.join(user_data_dir, f"{current_file_name_prefix}.xlsx")
                # No temp_plot_path check needed as plots are removed
                if not os.path.exists(temp_df_path):
                    file_name_prefix = current_file_name_prefix
                    break
                file_counter += 1
            logging.info(f"Generated unique file prefix for saving: {file_name_prefix}")
        except Exception as e:
            logging.error(f"Failed to create or access output directory '{user_data_dir}': {e}. Files will not be saved.")
            user_data_dir = None # Disable saving if directory issue

    # 6. Modality-Specific Calculations and Result Storage
    results_list = []
    # generated_plot_fig = None # REMOVE THIS

    try:
        if modality == 'psd_auc':
            target_channel = channels[0]
            fmin, fmax = validated_freq_ranges[0]
            band_display_name = freq_band_names_for_display[0]

            auc_internal_output_path = os.path.join(user_data_dir, f"{file_name_prefix}_psd_auc_raw.xlsx") if user_data_dir and file_name_prefix else None

            _, band_auc_values = compute_band_auc(
                epoch=epochs,
                ch_names=[target_channel],
                epoch_count=1,
                band_name=band_display_name,
                low_freq=fmin,
                high_freq=fmax,
                output_path=auc_internal_output_path
            )
            if not band_auc_values:
                raise RuntimeError("No AUC values returned from `compute_band_auc`.")
            calculated_value = band_auc_values[0]

            results_list.append({
                'Modality': 'PSD AUC',
                'Channel': target_channel,
                'Frequency Band': band_display_name,
                'Frequency Range (Hz)': f"{fmin}-{fmax}",
                'Value': calculated_value
            })
            # generated_plot_fig = plot_psd(...) # REMOVE THIS CALL

        elif modality == 'psd_ratio':
            target_channel = channels[0]
            fmin1, fmax1 = validated_freq_ranges[0]
            fmin2, fmax2 = validated_freq_ranges[1]
            band_name_1_display = freq_band_names_for_display[0]
            band_name_2_display = freq_band_names_for_display[1]

            if fmin1 <= fmax2 and fmin2 <= fmax1:
                logging.warning(f"Frequency bands '{band_name_1_display}' ({fmin1}-{fmax1} Hz) "
                                f"and '{band_name_2_display}' ({fmin2}-{fmax2} Hz) overlap. "
                                "This might affect interpretation of ratio.")

            ratio_internal_output_path = os.path.join(user_data_dir, f"{file_name_prefix}_psd_ratio_raw.xlsx") if user_data_dir and file_name_prefix else None
            calculated_value = dual_freq_auc(
                epoch=epochs,
                ch_name=target_channel,
                epoch_count=1,
                band_name_1=band_name_1_display,
                band_name_2=band_name_2_display,
                low_freq_1=fmin1, high_freq_1=fmax1,
                low_freq_2=fmin2, high_freq_2=fmax2,
                output_path=ratio_internal_output_path
            )
            if not isinstance(calculated_value, (int, float)):
                raise RuntimeError(f"Invalid ratio type returned from `dual_freq_auc`: {type(calculated_value)}")

            results_list.append({
                'Modality': 'PSD Ratio',
                'Channel': target_channel,
                'Numerator Band': band_name_1_display,
                'Denominator Band': band_name_2_display,
                'Numerator Range (Hz)': f"{fmin1}-{fmax1}",
                'Denominator Range (Hz)': f"{fmin2}-{fmax2}",
                'Value': calculated_value
            })
            # generated_plot_fig = plot_psd(...) # REMOVE THIS CALL

        elif modality == 'pac':
            target_channel = channels[0]
            fmin_phase, fmax_phase = validated_freq_ranges[0]
            fmin_amp, fmax_amp = validated_freq_ranges[1]
            phase_band_display_name = freq_band_names_for_display[0]
            amplitude_band_display_name = freq_band_names_for_display[1]

            pac_internal_output_path = os.path.join(user_data_dir, f"{file_name_prefix}_pac_raw.xlsx") if user_data_dir and file_name_prefix else None
            calculated_value = pac_cal(
                epoch=epochs,
                ch_name=target_channel,
                epoch_count=1,
                band_name_1=phase_band_display_name,
                band_name_2=amplitude_band_display_name,
                low_freq_1=fmin_phase, high_freq_1=fmax_phase,
                low_freq_2=fmin_amp, high_freq_2=fmax_amp,
                output_path=pac_internal_output_path
            )
            if not isinstance(calculated_value, (int, float)):
                raise RuntimeError(f"Invalid PAC value type returned from `pac_cal`: {type(calculated_value)}")

            results_list.append({
                'Modality': 'PAC',
                'Channel': target_channel,
                'Phase Band': phase_band_display_name,
                'Amplitude Band': amplitude_band_display_name,
                'Phase Range (Hz)': f"{fmin_phase}-{fmax_phase}",
                'Amplitude Range (Hz)': f"{fmin_amp}-{fmax_amp}",
                'Value': calculated_value
            })
            # REMOVE PAC specific plot logic here as well
            # generated_plot_fig = fig_pac (or similar) # REMOVE THIS

        elif modality == 'coh':
            ch1_name = channels[0]
            ch2_name = channels_2[0]
            fmin, fmax = validated_freq_ranges[0]
            band_display_name = freq_band_names_for_display[0]

            coh_internal_output_path = os.path.join(user_data_dir, f"{file_name_prefix}_coh_raw.xlsx") if user_data_dir and file_name_prefix else None

            logging.warning("Coherence calculation ('coh') is not fully implemented. Returning dummy value.")
            calculated_value = 0.5 # Placeholder

            if not isinstance(calculated_value, (int, float)):
                raise RuntimeError(f"Invalid coherence value type returned: {type(calculated_value)}")

            results_list.append({
                'Modality': 'Coherence',
                'Channel 1': ch1_name,
                'Channel 2': ch2_name,
                'Frequency Band': band_display_name,
                'Frequency Range (Hz)': f"{fmin}-{fmax}",
                'Value': calculated_value
            })
            # generated_plot_fig = plot_psd(...) # REMOVE THIS CALL

    except Exception as e:
        logging.exception(f"Error during {modality.upper()} computation: {e}")
        # generated_plot_fig = None # REMOVE THIS, as no plot is generated
        return None, None, None # Ensure None, None, None is returned on calculation error

    # 7. Save Results (No Plot Saving)
    df = pd.DataFrame(results_list)
    if user_data_dir and file_name_prefix: # Check file_name_prefix as well
        try:
            df_path = os.path.join(user_data_dir, f"{file_name_prefix}.xlsx")
            df.to_excel(df_path, index=False)
            logging.info(f"Saved baseline {modality} results to {df_path}")
        except Exception as e:
            logging.exception(f"Failed to save output DataFrame to '{user_data_dir}': {e}")
            df_path = None
    else:
        df_path = None # Ensure df_path is None if not saved

    # plot_path is always None now
    plot_path = None
    # No plt.close() needed as matplotlib is not used for plotting

    return df, df_path, plot_path

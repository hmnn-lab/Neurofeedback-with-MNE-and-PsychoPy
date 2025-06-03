import mne
import numpy as np
from pyprep import NoisyChannels
from asrpy import ASR
from mne.preprocessing import ICA
from mne_icalabel import label_components

def preproc_flow(file_path, n_channels, montage_name='standard_1020', notch_freq=50, prob_threshold=0.9, max_chunk_size=1):
    """
    Preprocess EEG data through bad channel detection, artifact rejection, and ICA.

    Parameters:
    -----------
    file_path : str
        Path to the .fif file containing raw EEG data.
    n_channels : int
        Number of EEG channels (required for ICA component calculation).
    montage_name : str, optional
        Name of the standard montage (default: 'standard_1020').
    notch_freq : float, optional
        Frequency for notch filter (default: 50 Hz).
    prob_threshold : float, optional
        Probability threshold for identifying artifact components (default: 0.9).
    max_chunk_size : float, optional
        Maximum chunk size in seconds for RANSAC (default: 1).

    Returns:
    --------
    tuple
        Contains:
        - raw_cleaned (mne.io.Raw): The preprocessed EEG data with artifacts removed.
        - bad_channels (list): List of bad channel names.
        - asr (asrpy.ASR): Trained ASR object.
        - ica (mne.preprocessing.ICA): Trained ICA object.
        - artifact_components (list): List of ICA component indices identified as artifacts.
        - labels (dict): Component labels and probabilities from ICLabel.
    """
    # Load raw data
    try:
        raw = mne.io.read_raw_fif(file_path, preload=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {file_path}")
    
    print(len(raw.times))
    raw.crop(tmax=120)

    if n_channels != len(raw.ch_names):
        raise ValueError(f"n_channels ({n_channels}) does not match number of channels ({len(raw.ch_names)})")

    # Ensure all channels are EEG
    for ch in raw.ch_names:
        if raw.get_channel_types([ch])[0] not in ['eeg']:
            print(f"Setting channel {ch} to EEG type")
            raw.set_channel_types({ch: 'eeg'})

    # Rename channels to match standard_1020 montage (case-sensitive)
    channel_mapping = {
        'CZ': 'Cz', 'FP1': 'Fp1', 'FP2': 'Fp2', 'FPZ': 'Fpz',
        'FZ': 'Fz', 'PZ': 'Pz', 'OZ': 'Oz'
    }
    raw.rename_channels(channel_mapping)
    print(f"Renamed channels to: {raw.ch_names}")

    # Set montage
    try:
        montage = mne.channels.make_standard_montage(montage_name)
        raw.set_montage(montage, on_missing='warn')  # Warn if channels are missing
    except Exception as e:
        print(f"Failed to set montage: {e}")
        raise ValueError("Could not set montage. Check channel names and montage compatibility.")

    # Verify channel positions
    montage = raw.get_montage()
    if montage:
        pos = montage.get_positions()['ch_pos']
        nan_channels = [ch for ch, coord in pos.items() if ch in raw.ch_names and np.any(np.isnan(coord))]
        if nan_channels:
            print(f"Warning: NaN positions found for channels: {nan_channels}")
            print("Proceeding without RANSAC due to invalid channel positions.")
            bad_channels = []
        else:
            # Apply filters first
            raw.filter(1, 100).notch_filter(notch_freq)
            
            # Set reference to common average reference (CAR) BEFORE other processing
            raw.set_eeg_reference('average', projection=True)
            raw.apply_proj()  # Apply the projection

            # Bad channel detection using PREP pipeline RANSAC
            try:
                nd = NoisyChannels(raw, random_state=1337)
                nd.find_bad_by_ransac(channel_wise=True, max_chunk_size=max_chunk_size)
                bad_channels = nd.bad_by_ransac or []
                raw.info['bads'].extend(bad_channels)
                print(f"Bad channels detected: {bad_channels}")
            except Exception as e:
                print(f"RANSAC failed: {e}. Proceeding without bad channel detection.")
                bad_channels = []
    else:
        print("No montage set. Skipping RANSAC.")
        bad_channels = []
        # Still apply filters and reference even without montage
        raw.filter(1, 100).notch_filter(notch_freq)
        raw.set_eeg_reference('average', projection=True)
        raw.apply_proj()

    # Interpolate bad channels before ASR
    if bad_channels:
        raw.interpolate_bads(reset_bads=False)

    # Artifact Subspace Reconstruction (ASR)
    asr = ASR(sfreq=raw.info['sfreq'])
    asr.fit(raw)
    raw = asr.transform(raw)

    # ICA for artifact detection and removal - use extended infomax
    n_components = int(0.9 * n_channels)
    ica = ICA(
        n_components=n_components, 
        method='infomax', 
        fit_params=dict(extended=True),  # Use extended infomax as recommended
        max_iter='auto', 
        random_state=42
    )
    ica.fit(raw)

    # Label components using iclabel
    labels = label_components(raw, ica, 'iclabel')
    component_labels = labels['labels']
    component_probs = labels['y_pred_proba']

    # Identify artifact components
    artifact_components = [i for i, (prob, label) in enumerate(zip(component_probs, component_labels))
                          if prob >= prob_threshold and label in ['muscle', 'eye']]
    print(f"Flagged artifact components: {artifact_components}")

    # Apply ICA to remove artifacts
    raw_cleaned = ica.apply(raw.copy(), exclude=artifact_components)

    # Return 6 values to match expected unpacking
    return raw_cleaned, bad_channels, asr, ica, artifact_components, labels
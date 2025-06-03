import mne
import numpy as np
from asrpy import ASR
import mne_icalabel

def preprocess_realtime_stream(data, client_info, rename_dict, bad_channels, asr, ica, artifact_components, montage_name='standard_1020', notch_freq=50):
    """
    Apply preprocessing steps to a real-time EEG stream.
    
    Parameters:
    -----------
    data : ndarray
        Real-time EEG data array (shape: n_channels x n_samples).
    client_info : mne.Info
        MNE Info object containing channel information.
    rename_dict : dict
        Dictionary mapping old channel names to new standard montage names.
    bad_channels : list
        List of bad channel names to mark.
    asr : asrpy.ASR
        Trained ASR object for artifact subspace reconstruction.
    ica : mne.preprocessing.ICA
        Trained ICA object for artifact removal.
    artifact_components : list
        List of ICA component indices to exclude (artifacts).
    montage_name : str, optional
        Name of the standard montage (default: 'standard_1020').
    notch_freq : float, optional
        Frequency for notch filter (default: 50 Hz).

    Returns:
    --------
    raw_realtime_processed : mne.io.Raw
        Preprocessed real-time EEG data.
    """
    # Validate inputs
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array (n_channels x n_samples)")
    n_channels, n_samples = data.shape
    if n_channels != len(client_info['ch_names']):
        raise ValueError(f"Data channels ({n_channels}) do not match client_info channels ({len(client_info['ch_names'])})")
    if asr.sfreq != client_info['sfreq']:
        raise ValueError(f"ASR sampling rate ({asr.sfreq} Hz) does not match client_info ({client_info['sfreq']} Hz)")
    if ica.n_components > n_channels:
        raise ValueError(f"ICA components ({ica.n_components}) exceed number of channels ({n_channels})")

    # Create RawArray from real-time data
    try:
        raw_realtime = mne.io.RawArray(data, client_info)
    except Exception as e:
        raise ValueError(f"Failed to create RawArray: {e}")

    # Rename channels
    try:
        raw_realtime.rename_channels(rename_dict)
        print(f"Renamed channels to: {raw_realtime.ch_names}")
    except Exception as e:
        raise ValueError(f"Channel renaming failed: {e}")

    # Set montage
    try:
        montage = mne.channels.make_standard_montage(montage_name)
        raw_realtime.set_montage(montage, on_missing='warn')
    except Exception as e:
        print(f"Warning: Failed to set montage: {e}")

    # Apply filters and reference - IMPORTANT: Apply CAR reference first
    try:
        raw_realtime.filter(1, 100, verbose=False).notch_filter(notch_freq, verbose=False)
        raw_realtime.set_eeg_reference('average', projection=True)
        raw_realtime.apply_proj()  # Apply the projection
    except Exception as e:
        raise RuntimeError(f"Filtering/referencing failed: {e}")

    # Mark and interpolate bad channels
    if bad_channels:
        raw_realtime.info['bads'].extend(bad_channels)
        print(f"Interpolating bad channels: {bad_channels}")
        try:
            raw_realtime.interpolate_bads(reset_bads=False)
        except Exception as e:
            print(f"Warning: Bad channel interpolation failed: {e}")

    # Apply ASR transformation
    try:
        raw_realtime_asr = asr.transform(raw_realtime)
    except Exception as e:
        raise RuntimeError(f"ASR transformation failed: {e}")

    # Apply ICA to remove artifact components
    try:
        raw_realtime_processed = ica.apply(raw_realtime_asr.copy(), exclude=artifact_components)
    except Exception as e:
        raise RuntimeError(f"ICA application failed: {e}")

    return raw_realtime_processed
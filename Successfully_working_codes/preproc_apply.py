import mne
from mne.preprocessing import ICA
from channel_utils import rename_eeg_channels

def preprocess_realtime_stream(data, client_info, bad_channels, asr, ica, artifact_components):
    """
    Apply preprocessing steps to a real-time EEG stream using saved preprocessing parameters.
    
    Parameters:
    -----------
    data : ndarray
        Real-time EEG data array (shape: n_channels x n_samples).
    client_info : mne.Info
        MNE Info object containing channel information.
    bad_channels : list
        List of bad channel names to mark.
    asr : asrpy.ASR
        Trained ASR object for artifact subspace reconstruction.
    ica : mne.preprocessing.ICA
        Trained ICA object for artifact removal.
    artifact_components : list
        List of ICA component indices to exclude (artifacts).
        
    Returns:
    --------
    raw_realtime_processed : mne.io.Raw
        Preprocessed real-time EEG data.
    """
    # Step 1: Create RawArray from real-time data
    raw_realtime = mne.io.RawArray(data, client_info)

    # Step 2: Determine if renaming is needed
    standard_montage = mne.channels.make_standard_montage('standard_1020')
    if not all(ch in standard_montage.ch_names for ch in raw_realtime.ch_names):
        raw_realtime, _ = rename_eeg_channels(raw_realtime, rename_dict=None)
    
    print(f"Renamed channels: {raw_realtime.ch_names}")  # Debug

    # Step 3: Apply bandpass and notch filters
    raw_realtime.notch_filter(50.0).filter(0.5, 40.0)
    print("Applied notch and bandpass filters")  # Debug

    # Step 4: Re-reference to average
    raw_realtime.set_eeg_reference('average')
    print("Re-referenced to average")  # Debug
    

    # Step 5: Set montage for spatial consistency
    raw_realtime.set_montage(standard_montage)
    print("Montage to standard_1020 set")  # Debug

    # Step 6: Mark bad channels
    raw_realtime.info['bads'].extend(bad_channels)
    print(f"Marked bad channels: {raw_realtime.info['bads']}")  # Debug

    # Step 7: Apply ASR
    raw_asr = asr.transform(raw_realtime)
    print("Applied ASR")  # Debug

    # Step 8: Apply ICA to remove identified artifact components
    raw_clean = ica.apply(raw_asr, exclude=artifact_components)
    print(f"Applied ICA, excluded components: {artifact_components}")  # Debug

    return raw_clean

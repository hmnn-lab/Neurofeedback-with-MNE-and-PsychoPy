import mne

def preprocess_realtime_stream(data, client_info, rename_dict, bad_channels, asr, ica, artifact_components):
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
        
    Returns:
    --------
    raw_realtime_processed : mne.io.Raw
        Preprocessed real-time EEG data.
    """
    # Create RawArray from real-time data
    raw_realtime = mne.io.RawArray(data, client_info)
    
    # Rename channels
    raw_realtime.rename_channels(rename_dict)
    
    # Mark bad channels
    raw_realtime.info['bads'].extend(bad_channels)
    
    # Apply ASR transformation
    raw_realtime_asr = asr.transform(raw_realtime)
    
    # Apply ICA to remove artifact components
    raw_realtime_processed = ica.apply(raw_realtime_asr, exclude=artifact_components)
    
    return raw_realtime_processed
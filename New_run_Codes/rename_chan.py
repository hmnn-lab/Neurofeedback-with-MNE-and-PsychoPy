def rename_eeg_channels(raw, rename_dict=None):
    """
    Rename EEG channels in a raw object according to a provided dictionary or default montage.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        The raw EEG data object to rename channels for.
    rename_dict : dict, optional
        Dictionary mapping old channel names to new standard montage names.
        If None, a default dictionary is used.
        
    Returns:
    --------
    raw : mne.io.Raw
        The raw object with renamed channels.
    n_channels : int
        The number of channels renamed.
    """
    # Default dictionary if none provided
    if rename_dict is None:
        rename_dict = {
            'EEG 001': 'Pz',
            'EEG 002': 'T8',
            'EEG 003': 'P3',
            'EEG 004': 'C3',
            'EEG 005': 'Fp1',
            'EEG 006': 'Fp2',
            'EEG 007': 'O1',
            'EEG 008': 'P4',
            'EEG 009': 'Fz',
            'EEG 010': 'F7',
            'EEG 011': 'C4',
            'EEG 012': 'O2',
            'EEG 013': 'F3',
            'EEG 014': 'F4',
            'EEG 015': 'Cz',
            'EEG 016': 'T3',
            'FZ': 'Fz',
            'CZ': 'Cz',
            'PZ': 'Pz',
            'OZ': 'Oz',
            'FP1': 'Fp1',
            'FP2': 'Fp2',
            'FPZ': 'Fpz'
        }
    
    # Rename channels
    raw.rename_channels(rename_dict)
    
    # Store number of channels
    n_channels = len(rename_dict.keys())
    
    return raw, n_channels
import mne

def load_and_filter_base():
    """
    Prompt the user for a baseline.fif file path, load the EEG data, and apply basic filtering.
    
    Returns:
    --------
    raw : mne.io.Raw
        The filtered raw EEG data object with notch filter, bandpass filter,
        average reference, and standard 10-20 montage applied.
    """
    # Prompt user to input the file path for baseline.fif
    file_path = input("Please enter the file path for the baseline.fif file: ")
    
    # Load raw EEG data
    raw = mne.io.read_raw_fif(file_path, preload=True)
    
    # Apply notch filter and bandpass filter
    raw.notch_filter(50, picks='eeg').filter(l_freq=0.1, h_freq=40)
    
    # Apply average re-referencing
    raw.set_eeg_reference('average')
    
    # Set standard 10-20 montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    
    return raw
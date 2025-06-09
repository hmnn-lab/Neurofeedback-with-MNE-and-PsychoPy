import mne

def rename_eeg_channels(raw, rename_dict=None):
    """
    Rename EEG channels in a raw object according to a provided dictionary.

    Parameters:
    -----------
    raw : mne.io.Raw
        The raw EEG data object to rename channels for.
    rename_dict : dict, optional
        Dictionary mapping old channel names to new names.
        If None, a default dictionary for 16 EEG channels is used.

    Returns:
    --------
    raw : mne.io.Raw
        The raw object with renamed channels.
    n_channels : int
        Number of channels successfully renamed.
    """

    # Default rename dictionary (matches your EEG 001–016 layout)
    if rename_dict is None:
        rename_dict = {
            'EEG 001': 'C3',
            'EEG 002': 'C4',
            'EEG 003': 'Cz',
            'EEG 004': 'F3',
            'EEG 005': 'F4',
            'EEG 006': 'F7',
            'EEG 007': 'F8',
            'EEG 008': 'Fp1',
            'EEG 009': 'Fp2',
            'EEG 010': 'Fpz',
            'EEG 011': 'Fz',
            'EEG 012': 'Pz',
            'EEG 013': 'T4',
            'EEG 014': 'O1',
            'EEG 015': 'O2',
            'EEG 016': 'Oz'
        }

    # Filter the rename_dict to include only keys present in raw.ch_names
    filtered_dict = {k: v for k, v in rename_dict.items() if k in raw.ch_names}
    if not filtered_dict:
        raise ValueError(f"None of the channels in rename_dict match raw.ch_names: {raw.ch_names}")

    # Apply renaming
    raw.rename_channels(filtered_dict)

    # Return raw and number of renamed channels
    return raw, len(filtered_dict)

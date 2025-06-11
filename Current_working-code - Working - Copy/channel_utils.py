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
            '0': 'Pz',
            '1': 'F8',
            '2': 'P3',
            '3': 'C3',
            '4': 'Fp1',
            '5': 'Fp2',
            '6': 'O1',
            '7': 'P4',
            '8': 'Fz',
            '9': 'F7',
            '10': 'C4',
            '11': 'O2',
            '12': 'F3',
            '13': 'F4',
            '14': 'Cz',
            '15': 'T3'
        }

        # rename_dict = {
        #     '0': 'C3',
        #     '1': 'C4',
        #     '2': 'Cz',
        #     '3': 'F3',
        #     '4': 'F4',
        #     '5': 'F7',
        #     '6': 'F8',
        #     '7': 'Fp1',
        #     '8': 'Fp2',
        #     '9': 'Fpz',
        #     '10': 'Fz',
        #     '11': 'Pz',
        #     '12': 'T4',
        #     '13': 'O1',
        #     '14': 'O2',
        #     '15': 'Oz'
        # }

    # Filter the rename_dict to include only keys present in raw.ch_names
    filtered_dict = {k: v for k, v in rename_dict.items() if k in raw.ch_names}
    if not filtered_dict:
        raise ValueError(f"None of the channels in rename_dict match raw.ch_names: {raw.ch_names}")

    # Apply renaming
    raw.rename_channels(filtered_dict)

    # Return raw and number of renamed channels
    return raw, len(filtered_dict)

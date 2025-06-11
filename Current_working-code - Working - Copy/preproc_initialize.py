import mne
import pyprep
import asrpy
from pyprep import NoisyChannels
from asrpy import ASR
from mne.preprocessing import ICA
from mne_icalabel import label_components
from channel_utils import rename_eeg_channels

def preproc_flow(baseline_file_path, n_channels, channel_names=None):
    """
    Preprocess EEG data through bad channel detection, artifact rejection, and ICA.
    
    Parameters:
    -----------
    baseline_file_path : str
        File path to the baseline .fif EEG recording.
    n_channels : int
        Number of EEG channels (required for ICA component calculation).
    channel_names : list, optional
        List of new channel names (e.g., ['Fz', 'Cz', 'Pz']) for montage.
    progress_callback : callable, optional
        Function to call with progress updates (0.0 to 1.0).
        
    Returns:
    --------
    tuple
        Contains:
        - raw_cleaned (mne.io.Raw): The preprocessed EEG data with artifacts removed.
        - bad_channels (list): List of bad channel names.
        - asr (asrpy.ASR): Trained ASR object.
        - ica (mne.preprocessing.ICA): Trained ICA object.
        - artifact_components (list): List of ICA component indices identified as artifacts.
    """

    # Load raw data
    raw = mne.io.read_raw_fif(baseline_file_path, preload=True)
    
    # Validate inputs
    if n_channels != len(raw.ch_names):
        raise ValueError(f"Number of channels ({n_channels}) does not match raw data channels ({len(raw.ch_names)})")

    print(f"Original channels: {raw.ch_names}")  # Debug

    # Rename channels using provided dictionary or default montage
    raw, renamed_n_channels = rename_eeg_channels(raw, rename_dict=None)
    print(f"Renamed channels: {raw.ch_names}")  # Debug
    # if channel_names:
    #     if len(channel_names) != renamed_n_channels:
    #         raise ValueError(f"Number of provided channel names ({len(channel_names)}) does not match renamed channels ({renamed_n_channels})")
    #     raw.rename_channels({old: new for old, new in zip(raw.ch_names, channel_names)})
    # else:
    #     # Use default montage if no channel names provided
    #     raw.rename_channels({
    #         'EEG 001': 'C3', 'EEG 002': 'C4', 'EEG 003': 'Cz',
    #         'EEG 004': 'F3', 'EEG 005': 'F4', 'EEG 006': 'F7',
    #         'EEG 007': 'F8', 'EEG 008': 'Fp1', 'EEG 009': 'Fp2',
    #         'EEG 010': 'Fpz', 'EEG 011': 'Fz', 'EEG 012': 'Pz',
    #         'EEG 013': 'T4', 'EEG 014': 'O1', 'EEG 015': 'O2',
    #         'EEG 016': 'Oz'
    #     })  
    
    # Apply notch and bandpass filters
    raw.notch_filter(50).filter(0.5, 40.0)
     

    # Set montage for spatial consistency
    raw.set_montage("standard_1020", match_case=False)  # match_case can help with capitalization mismatches
    
    # Update info with montage  
    print(raw.info['sfreq'])  # Ensure sfreq is set correctly
    
    # Detect bad channels using PREP pipeline (includes RANSAC)
    nd = NoisyChannels(raw, random_state=1337)
    nd.find_bad_by_ransac(channel_wise=True, max_chunk_size=1, sample_prop=0.5)
    bad_channels = nd.bad_by_ransac
    raw.info['bads'].extend(bad_channels)

    print('Ransac Completed')

    # Apply ASR
    asr = ASR(sfreq=raw.info['sfreq'])
    asr.fit(raw)
    raw = asr.transform(raw)

    # ICA for artifact detection and removal
    ica = ICA(n_components=n_channels - len(bad_channels), method='infomax', 
              max_iter='auto', random_state=42)
    ica.fit(raw)

    # Label components
    labels = label_components(raw, ica, 'iclabel')
    component_labels = labels['labels']
    component_probs = labels['y_pred_proba']

    # Identify artifact components (muscle or eye, with high confidence)
    artifact_components = []
    for i, prob in enumerate(component_probs):
        if prob >= 0.9 and component_labels[i] in ['muscle', 'eye']:
            print(f"Component {i} is likely an artifact")
            artifact_components.append(i)

    print("Flagged artifact components:", artifact_components)

    # Apply ICA to remove artifact components
    raw_cleaned = ica.apply(raw.copy(), exclude=artifact_components)

    return raw_cleaned, bad_channels, asr, ica, artifact_components

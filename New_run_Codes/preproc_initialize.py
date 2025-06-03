import mne
from pyprep import NoisyChannels
from asrpy import ASR
from mne.preprocessing import ICA
from mne_icalabel import label_components

def preproc_flow(raw, n_channels):
    """
    Preprocess EEG data through bad channel detection, artifact rejection, and ICA.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        The raw EEG data object (preprocessed with notch filter, bandpass filter,
        average reference, and montage, e.g., from load_and_filter_base()).
    n_channels : int
        Number of EEG channels (required for ICA component calculation).
        
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
    raw = mne.io.read_raw_fif(r'C:\Users\varsh\NFB_Spyder\New_Runs\Baseline_codes\baseline_recordings\baseline_002.fif', preload=True)
    #raw.load_data()
    raw.filter(0.5, 40).notch_filter(50)
    # Bad channel detection using PREP pipeline RANSAC
    nd = NoisyChannels(raw, random_state=1337)
    nd.find_bad_by_ransac(channel_wise=True, max_chunk_size=1)
    bad_channels = nd.bad_by_ransac
    raw.info['bads'].extend(bad_channels)
    
    # Artifact Subspace Reconstruction (ASR)
    asr = ASR(sfreq=raw.info['sfreq'])
    asr.fit(raw)
    raw = asr.transform(raw)
    
    # ICA for artifact detection and removal
    ica = ICA(n_components=n_channels - len(bad_channels), method='infomax', 
              max_iter=500, random_state=42)
    ica.fit(raw)
    
    # Label components using iclabel
    labels = label_components(raw, ica, 'iclabel')
    component_labels = labels['labels']
    component_probs = labels['y_pred_proba']
    
    # Identify artifact components
    artifact_components = []
    for i, prob in enumerate(component_probs):
        if prob >= 0.9 and prob <= 1 and component_labels[i] in ['muscle', 'eye']:
            print(f"Component {i} is likely an artifact")
            artifact_components.append(i)
    
    # Print ICA weights and flagged components
    ica_weights = ica.unmixing_matrix_
    ica_inverse_weights = ica.mixing_matrix_
    print("Flagged artifact components:", artifact_components)
    
    # Apply ICA to remove artifacts
    raw_cleaned = ica.apply(raw.copy(), exclude=artifact_components)
    
    return raw_cleaned, bad_channels, asr, ica, artifact_components


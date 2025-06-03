import mne
from pyprep import NoisyChannels
from asrpy import ASR
from mne.preprocessing import ICA
from mne_icalabel import label_components
import numpy as np

class EEGPreproc:
    """A class for preprocessing EEG data, supporting both offline and real-time processing."""
    
    def __init__(self, sfreq, n_channels, random_state=42):
        """
        Initialize the EEGPreprocessor with sampling frequency and number of channels.
        
        Parameters:
        -----------
        sfreq : float
            Sampling frequency of the EEG data (Hz).
        n_channels : int
            Number of EEG channels.
        random_state : int, optional
            Random seed for reproducibility (default: 42).
        """
        self.sfreq = sfreq
        self.n_channels = n_channels
        self.random_state = random_state
        self.bad_channels = []
        self.asr = None
        self.ica = None
        self.artifact_components = []
        
    def preproc_offline(self, raw, rename_dict=None):
        """
        Preprocess EEG data through bad channel detection, artifact rejection, and ICA.
        
        Parameters:
        -----------
        raw : mne.io.Raw
            The raw EEG data object (preprocessed with notch filter, bandpass filter,
            average reference, and montage).
        rename_dict : dict, optional
            Dictionary mapping old channel names to new standard montage names.
            
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
        # Apply channel renaming if provided
        if rename_dict:
            raw.rename_channels(rename_dict)
        
        # Bad channel detection using PREP pipeline RANSAC
        nd = NoisyChannels(raw, random_state=self.random_state)
        nd.find_bad_by_ransac(channel_wise=True, max_chunk_size=1)
        self.bad_channels = nd.bad_by_ransac
        raw.info['bads'].extend(self.bad_channels)
        
        # Artifact Subspace Reconstruction (ASR)
        self.asr = ASR(sfreq=self.sfreq)
        self.asr.fit(raw)
        raw = self.asr.transform(raw)
        
        # ICA for artifact detection and removal
        self.ica = ICA(n_components=self.n_channels - len(self.bad_channels), 
                      method='infomax', max_iter=500, random_state=self.random_state)
        self.ica.fit(raw)
        
        # Label components using iclabel
        labels = label_components(raw, self.ica, 'iclabel')
        component_labels = labels['labels']
        component_probs = labels['y_pred_proba']
        
        # Identify artifact components
        self.artifact_components = [
            i for i, prob in enumerate(component_probs)
            if prob >= 0.9 and prob <= 1 and component_labels[i] in ['muscle', 'eye']
        ]
        
        # Print ICA weights and flagged components
        print("Flagged artifact components:", self.artifact_components)
        
        # Apply ICA to remove artifacts
        raw_cleaned = self.ica.apply(raw.copy(), exclude=self.artifact_components)
        
        return raw_cleaned, self.bad_channels, self.asr, self.ica, self.artifact_components
    
    def preproc_realtime(self, data, client_info, rename_dict=None):
        """
        Apply preprocessing steps to a real-time EEG stream.
        
        Parameters:
        -----------
        data : ndarray
            Real-time EEG data array (shape: n_channels x n_samples).
        client_info : mne.Info
            MNE Info object containing channel information.
        rename_dict : dict, optional
            Dictionary mapping old channel names to new standard montage names.
            
        Returns:
        --------
        raw_realtime_processed : mne.io.Raw
            Preprocessed real-time EEG data.
        """
        # Create RawArray from real-time data
        raw_realtime = mne.io.RawArray(data, client_info)
        
        # Rename channels if provided
        if rename_dict:
            raw_realtime.rename_channels(rename_dict)
        
        # Mark bad channels
        raw_realtime.info['bads'].extend(self.bad_channels)
        
        # Apply ASR transformation
        if self.asr is None:
            raise ValueError("ASR model not trained. Run preprocess_offline first.")
        raw_realtime_asr = self.asr.transform(raw_realtime)
        
        # Apply ICA to remove artifact components
        if self.ica is None:
            raise ValueError("ICA model not trained. Run preprocess_offline first.")
        raw_realtime_processed = self.ica.apply(raw_realtime_asr, exclude=self.artifact_components)
        
        return raw_realtime_processed
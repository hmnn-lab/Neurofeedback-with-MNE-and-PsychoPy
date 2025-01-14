# NEUROFEEDBACK USING MNE PYTHON
This is a software for realtime neurofeeback training based on MNE and PsychoPy for visualization.
It uses modules and libraries from MNE to stream EEG signal in real time through LSL (Lab Streaming Layer), do basic preprocessing, 
automatically rejection bad channels and perform ICA for baseline calibration. The feedback is visualized through PsychoPy tools. 

## Dependencies
1. Python version 3.9 
2. MNE Python version 1.7 or higher (https://mne.tools/stable/install/index.html)
   + mne-realtime (https://mne.tools/mne-realtime/)
   + numpy             1.24.4
   + scipy             1.11.4
   + matplotlib        3.8.4 (backend=QtAgg)
   + mne-icalabel      0.7.0 (requires sklearn version 1.5.1)
3. LSL (https://github.com/sccn/liblsl) and pylsl (https://github.com/chkothe/pylsl)
4. PsychoPy version 2024.2.1 (https://www.psychopy.org/download.html)


## Folders
1. Offline codes: 
   + Contains the codes for realtime mock LSL streaming of raw EEG, PSD and relative PSD. No          prepropressing done here. 
   + Also contains the basic paradigms running on mock LSL stream
2. Final codes:
   + contains the codes that work on Real-time LSL stream of EEG signal with preprocessing
   + also contains preprocessing file 
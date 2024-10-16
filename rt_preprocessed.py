# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:46:37 2024

@author: varsh
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg') 

import mne_realtime
from mne_realtime import LSLClient, MockLSLStream

import pylsl

import pyxdf
import mne
mne.viz.set_browser_backend('qt')

#import threading

# this is the host id that identifies our stream on LSL
host = 'mne_stream'
# the max wait time in seconds until client connection
wait_max = 5


# Load a file to stream raw data
xdf_fpath = "C:/Users/varsh/.spyder-py3/random-block-1.xdf"

streams, header = pyxdf.load_xdf(xdf_fpath)

eeg_stream = streams[0]
data = streams[0]["time_series"].T
ch_names = ['Ch1','Ch2','Ch3','Ch4','Ch5','Ch6','Ch7','Ch8']
sfreq = float(streams[0]["info"]["nominal_srate"][0])
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

raw = mne.io.RawArray(data, info)
raw.apply_function(lambda x: x*1e-6, picks="eeg")
# Renaming the channels according to the standards to set the monatge
rename_dict = {
    'Ch1': 'F3',
    'Ch2': 'O1',
    'Ch3': 'CPz',
    'Ch4': 'AFz',
    'Ch5': 'Fz',
    'Ch6': 'Cz',
    'Ch7': 'Pz',
    'Ch8': 'Fp1',}
raw.rename_channels(rename_dict)
#Creating a standard EEG montage
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

# Applying Notch filter and bandpass filter
raw.notch_filter(50, picks='eeg').filter(l_freq=0.1, h_freq=40)
#baseline correction - baseline.rescale() is not compatible with RawArray()
#baseline correction is being done in the main loop on the epochs

# ICA (automatic artifact detection and removal of ICs)
from mne.preprocessing import ICA
ica = ICA(n_components=None, random_state=97, max_iter='auto', method = 'infomax')
ica.fit(raw)

from mne_icalabel import label_components
ica_labels = label_components(raw, ica,'iclabel')  # Label ICA components using ICLabel
print(ica_labels)  # Print ICLabel classifications (components will be labeled as 'brain', 'eye', 'heart', etc.)

# Automatically rejects eye and heart components
import numpy as np
artifact_inds = np.where(np.logical_or(
    np.array(ica_labels['labels']) == 'eye',
    np.array(ica_labels['labels']) == 'heart'
))[0]

print(f"Removing components: {artifact_inds}")
ica.exclude = artifact_inds  # Mark these components for removal
raw = ica.apply(raw)  # Apply ICA to remove the components


# Initializing the mock LSL stream.
#n_epochs = 10
running = True
epoch_count = 1

_, ax = plt.subplots(1)
# main function to enable script as own program
# in such way a child process can be started (primarily for Windows)
if __name__ == '__main__':
    with MockLSLStream(host, raw, 'eeg'):
        with LSLClient(info=raw.info, host=host, wait_max=wait_max) as client:
            client_info = client.get_measurement_info()
            sfreq = int(client_info['sfreq'])
            # let's loop the data as realtime stream
            while running:
                print(f'Got epoch {epoch_count}' )
                plt.cla()
                ax.set_title('Raw data stream realtime')
                
                epoch = client.get_data_as_epoch(n_samples=sfreq)
                # Baseline correction: subtract the mean of the first 100 ms
                epoch.apply_baseline(baseline=(0, None))  # (None, 0) corrects using the full range before time 0
                epoch.average().plot(axes=ax)
                plt.pause(0.1) #
                
                epoch_count += 1
            plt.draw()           
print('Streams closed')

print(__doc__)


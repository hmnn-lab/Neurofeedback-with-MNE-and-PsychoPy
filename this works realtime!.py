# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:47:47 2024

@author: varsh
"""
# Import the required libraries 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg') 

import mne_realtime
from mne_realtime import LSLClient, MockLSLStream

import pylsl

import pyxdf
import mne
mne.viz.set_browser_backend('qt')

# this is the host id that identifies your stream on LSL
host = 'mne_stream'
# this is the max wait time in seconds until client connection
wait_max = 5


# Load a file to stream raw data
xdf_fpath = "C:/Users/varsh/.spyder-py3/random-block-1.xdf"

streams, header = pyxdf.load_xdf(xdf_fpath)

eeg_stream = streams[0]
data = streams[0]["time_series"].T
ch_names = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8']
sfreq = float(streams[0]["info"]["nominal_srate"][0])
info = mne.create_info(ch_names, sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(data, info)
raw.apply_function(lambda x: x*1e-6, picks="eeg") # to convert the microvolts to volts
# Apply notch filter and bandpass filter
raw.notch_filter(50, picks='eeg').filter(l_freq=0.1, h_freq=40)

# raw.plot(scalings=dict(eeg=20e-6), title="Eye closed rest")

# Creating the mock LSL stream.
#n_epochs = 10
running = True
epoch_count = 1
_, ax = plt.subplots(1)
# main function is necessary here to enable script as own program
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
                epoch.average().plot(axes=ax)
                plt.pause(0.1) # the time (in seconds) to update the epoch frame
                
                epoch_count += 1
            plt.draw()           
print('Streams closed')

print(__doc__)


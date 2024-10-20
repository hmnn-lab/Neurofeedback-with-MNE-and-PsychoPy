# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 16:21:28 2024

@author: varsh
"""
# Import the necessary libraries
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

import mne_realtime
from mne_realtime import LSLClient, MockLSLStream

import pylsl

import pyxdf
import mne
mne.viz.set_browser_backend('qt')

# this is the host id that identifies your stream on LSL
host = 'HMNN-LSL'
# this is the max wait time in seconds until client connection
wait_max = 5

# Load a file to stream raw data
xdf_fpath = "C:/Users/varsh/.spyder-py3/eye-open-mc.xdf"

streams, header = pyxdf.load_xdf(xdf_fpath)
eeg_stream = streams[0]

#User input
step = 0.1 #in seconds #changed from 0.25
time_window = 1 #in seconds
n_channels = 2
ch_names = ['Ch1', 'Ch2']
low_freq = 4; # for theta band (4-7 Hz)
high_freq = 7;
data = streams[0]["time_series"].T
data = data[:n_channels]

sfreq = float(streams[0]["info"]["nominal_srate"][0])
info = mne.create_info(ch_names, sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(data, info)
raw.apply_function(lambda x: x*1e-6, picks="eeg")

raw.notch_filter(50, picks='eeg').filter(l_freq=0.1, h_freq=40)

# raw.plot(scalings=dict(eeg=20e-6), title="Eye closed rest")

# use the mock LSL stream.

#n_epochs = 10
running = True
epoch_count = 1 # Initializing the epoch counter
time_points = [] # Initializing the list to save the time points of update
pow_change_values = [] # initializing the list to save the relative power values

_, ax = plt.subplots(1)
# main function is necessary here to enable script as own program
# in such way a child process can be started (primarily for Windows)
if __name__ == '__main__':
    with MockLSLStream(host, raw, 'eeg', step):
        with LSLClient(info=raw.info, host=host, wait_max=wait_max) as client:
            client_info = client.get_measurement_info()
            sfreq = int(client_info['sfreq'])
            # Let's observe ten seconds of data
            while running:
                    print(f'Got epoch {epoch_count}')
                    epoch = client.get_data_as_epoch(n_samples=int(time_window * sfreq))
                    #Clear the previous plot
                    ax.clear()
                    # set the plot features 
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Relative Power Change (%)')
                    ax.set_title('Relative theta Power Change')
                    
                    # calculate the power 
                    psd = epoch.compute_psd(tmin=0, tmax=250, picks="eeg", method='welch', average=False)
                    power = (psd._data).reshape(n_channels, max((psd._data).shape))
                    power_whole = power.sum(axis = 1)
                    freq = psd._freqs
                    low_freq_ind = (abs(freq - low_freq)).argmin()
                    high_freq_ind = (abs(freq - high_freq)).argmin()
                    power_range = (power[:,low_freq_ind:high_freq_ind]).sum(axis=1)
                    power_change = (power_range/power_whole)*100 # relative power change
                    # Append the current epoch count and power_change to the lists
                    time_points.append(epoch_count)
                    pow_change_values.append(power_change)
                    ax.plot(time_points, pow_change_values, label=ch_names)
                    ax.legend(loc='upper right')
                    
                    #Draw the updated plot
                    plt.draw()
                    #Pause to update the plot in realtime
                    plt.pause(0.1) #changed from previously 0.25 to 0.1
                    epoch_count += 1
            plt.show()
           
print('Streams closed')

 
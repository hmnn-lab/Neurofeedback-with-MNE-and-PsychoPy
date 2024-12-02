# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:08:15 2024

@author: varsh
"""

import mne
from mne_realtime import LSLClient
from mne import create_info
import numpy as np

#Define the stream name 
stream_name = '' #Replace with your actual stream name
n_channels = 16 #Adjust according to your stream
sampling_rate = 250 #Adjust based on your stream

# Create info object
info = create_info(ch_names=[f'EEG {i}'for i in range(n_channels)], 
                   sfreq=sampling_rate,
                   ch_type='eeg')
#Create an LSL Client and start pulling data
with LSLClient(info=info, host=stream_name) as client:
    client.start_acquisition()
    
    #Prepare to store the data
    all_data = []
    
    #Continuously pull and store the data
    try:
        for _ in range(1000): # Adjust based on how long the data is to be captured
            raw_data, _ = client.get_data_as_epoch(n_samples=100) # Fetch 100 samples at a time
            if raw_data is not None:
                all_data.append(raw_data.get_data())
                
    except KeyboardInterrupt:
        print("Stopping data collection.")
        
    # Stop aquisition
    client.stop_acquisition()
    
    # Combine all collected data
    all_data = np.concatenate(all_data, axis=-1)
    
    # Create a raw object 
    raw = mne.io.RawArray(all_data, info)
    
    #Save the data to a file
    raw.save('lsl_eeg_data.fif', overwrite=True)    


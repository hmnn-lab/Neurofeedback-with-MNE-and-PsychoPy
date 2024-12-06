import psychopy
from psychopy import visual, core, event
from psychopy.hardware import keyboard
import pyxdf
import numpy as np
import pandas as pd  # Import pandas for Excel saving

import matplotlib.pyplot as plt
import mne_realtime
from mne_realtime import LSLClient, MockLSLStream
import pylsl
import mne

# Create a mywindow
mywin = visual.Window([500, 500], monitor="TestMonitor", color=[0, 0, 0], fullscr=False, units="deg")
fixation = visual.ShapeStim(mywin, vertices=((0, -1), (0, 1), (0, 0), (-1, 0), (1, 0)), lineWidth=2, closeShape=False, lineColor="white")

# Create square and circle stimuli
sqr = visual.Rect(win=mywin, size=[20, 20], fillColor='black', pos=(0.0, 0.0))
mycir = visual.Circle(win=mywin, radius=5.0, edges=128, lineWidth=1.5, lineColor='white', fillColor='red', pos=(0, 0))

# Parameters for smooth radius change
max_radius = 10
min_radius = 2

# Create keyboard component
kb = keyboard.Keyboard()

# this is the host id that identifies your stream on LSL
host = 'HMNN-LSL'
wait_max = 5

# Load EEG data (for demo, assuming your xdf file is working properly)
xdf_fpath = "C:/Users/varsh/.spyder-py3/eye-open-rest.xdf"
streams, header = pyxdf.load_xdf(xdf_fpath)
eeg_stream = streams[0]

# EEG parameters
step = 0.01  # in seconds
time_window = 5  # in seconds
n_channels = 2
ch_names = ['Ch1', 'Ch2']
low_freq = 8 #alpha band
high_freq = 12
data = streams[0]["time_series"].T
data = data[:n_channels]

sfreq = float(streams[0]["info"]["nominal_srate"][0])
info = mne.create_info(ch_names, sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(data, info)
raw.apply_function(lambda x: x * 1e-6, picks="eeg")
raw.notch_filter(50, picks='eeg').filter(l_freq=0.1, h_freq=40)

# Timer for updating the circle radius every 0.75 seconds
update_interval = 0.5
update_timer = core.CountdownTimer(update_interval)

# Main loop variables
running = True
epoch_count = 1
time_points = []
pow_change_values = []
feedback_values = []  # List to store feedback values (circle radius)

if __name__ == '__main__':
    with MockLSLStream(host, raw, 'eeg', step):
        with LSLClient(info=raw.info, host=host, wait_max=wait_max) as client:
            client_info = client.get_measurement_info()
            sfreq = int(client_info['sfreq'])
            
            while running:
                keys = event.getKeys()
                if 'escape' in keys:
                    running = False
                    # Save the feedback values to an Excel file
                    feedback_df = pd.DataFrame(feedback_values, columns=['Feedback Radius'])
                    feedback_df.to_excel(r"C:\Users\varsh\OneDrive\Desktop\Neurofeedback MNE-PsychoPy\feedback_values.xlsx", index=False)
                    print("Exiting and saving data")
                    break
                
                # Check if we need to update
                if update_timer.getTime() <= 0:
                    print(f'Got epoch {epoch_count}')
                    
                    # Get epoch data and compute power spectrum
                    epoch = client.get_data_as_epoch(n_samples=int(time_window * sfreq))
                    psd = epoch.compute_psd(tmin=0, tmax=250, picks="eeg", method='welch', average=False)
                    
                    # Reshape power data
                    power = (psd._data).reshape(n_channels, max((psd._data).shape))
                    power_whole = power.sum(axis=1)
                    
                    # Frequency indices and boundaries
                    freq = psd._freqs
                    low_freq_ind = (abs(freq - low_freq)).argmin()
                    high_freq_ind = (abs(freq - high_freq)).argmin()
                    
                    # Sum power in the chosen frequency band
                    power_range = (power[:, low_freq_ind:high_freq_ind]).sum(axis=1)
                    
                    # Compute relative power change 
                    power_change = (power_range / power_whole) * 100
                    time_points.append(epoch_count)
                    pow_change_values.append(power_change)
                    
                    # Reset update timer
                    update_timer.reset(update_interval)
                    epoch_count += 1
                
                # Update circle's radius
                if len(pow_change_values) > 0:
                    current_power_change = pow_change_values[-1]
                    new_radius = np.interp(np.mean(current_power_change), [0, 100], [min_radius, max_radius])
                    mycir.radius = new_radius
                    feedback_values.append(new_radius)  # Store the feedback value
                    print(f'The feedback value is: {new_radius}')
                
                # Draw stimuli
                sqr.draw()  # Draw square
                mycir.draw()  # Draw circle
                fixation.draw()  # Draw fixation cross
                
                mywin.flip()  # Flip window to reflect changes

mywin.close()
core.quit()
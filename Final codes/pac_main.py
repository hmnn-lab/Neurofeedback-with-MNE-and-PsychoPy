import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
from mne import EpochsArray
from mne.baseline import rescale
from mne_realtime import LSLClient
import matplotlib 
matplotlib.use('QtAgg')
from pyprep.find_noisy_channels import NoisyChannels
from asrpy import ASR
from scipy.signal import butter, freqz
from mne_realtime import LSLClient
import psychopy
from psychopy import visual, core, event
from psychopy.hardware import keyboard
from datetime import datetime

# Function for Endpoint Corrected Hilbert Transform
def echt(xr, filt_lf, filt_hf, Fs, n=None):
    if not np.isrealobj(xr):
        print("Warning: Ignoring imaginary part of input signal.")
        xr = np.real(xr)
    if n is None:
        n = len(xr)
    x = np.fft.fft(xr, n=n)
    h = np.zeros(n, dtype=float)
    if n > 0 and n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:n // 2] = 2
    elif n > 0:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    x *= h
    filt_order = 2
    b, a = butter(filt_order, [filt_lf / (Fs / 2), filt_hf / (Fs / 2)], btype='bandpass')
    T = 1 / Fs * n
    filt_freq = np.fft.fftfreq(n, d=1 / Fs)
    filt_coeff = freqz(b, a, worN=filt_freq, fs=Fs)[1]
    x = np.fft.fftshift(x)
    x *= filt_coeff
    x = np.fft.ifftshift(x)
    analytic_signal = np.fft.ifft(x)
    phase = np.angle(analytic_signal)
    amplitude = np.abs(analytic_signal)
    return analytic_signal, phase, amplitude

###### BASELINE CALIBRATION #####
# The host id that identifies the stream of interest on LSL
host = 'openbcigui'
# This is the max wait time in seconds until client connection
wait_max = 5

raw = mne.io.read_raw_fif(r"C:\Users\Admin\Desktop\Varsha\mne-psychopy-codes\baseline-data-rashi4-raw.fif", preload=True) # Load the raw EEG baseline calibration
# Apply notch filter to remove power-line noise and bandpass filter the signal
raw.notch_filter(50, picks='eeg').filter(l_freq=0.1, h_freq=40)
# Apply EEG re-referencing
raw.set_eeg_reference('average')
# Create a dictionary for renaming the electrodes to fit according to standard montage
# rename_dict = {
#     'EEG 001': 'Pz',
#     'EEG 002': 'T8',
#     'EEG 003': 'P3',
#     'EEG 004': 'C3',
#     'EEG 005': 'Fp1',
#     'EEG 006': 'Fp2',
#     'EEG 007': 'O1',
#     'EEG 008': 'P4',
#     'EEG 009': 'Fz',
#     'EEG 010': 'F7',
#     'EEG 011': 'C4',
#     'EEG 012': 'O2',
#     'EEG 013': 'F3',
#     'EEG 014': 'F4',
#     'EEG 015': 'Cz',
#     'EEG 016': 'T3',
#     }
rename_dict = {
    'EEG 001': 'O1',
    'EEG 002': 'O2',
    'EEG 003': 'F3',
    'EEG 004': 'C3',
    'EEG 005': 'Fp1',
    'EEG 006': 'Fp2',
    'EEG 007': 'P3',
    'EEG 008': 'P4',
}
raw.rename_channels(rename_dict)
# Store the number of channels 
n_channels = len(rename_dict.keys())
# Set the montage 
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

# Apply notch filter to remove powerline noise and bandpass filter
raw.notch_filter(50, picks='eeg').filter(l_freq=0.1, h_freq=40)

####### PREPROCESSING #######
# Bad channels detection and rejection using PREP pipeline RANSAC algorithm
nd = NoisyChannels(raw, random_state=1337)

nd.find_bad_by_ransac(channel_wise=True, max_chunk_size=1, sample_prop=0.5)
bad_channels = nd.bad_by_ransac
raw.info['bads'].extend(bad_channels)

# Artifact Subspace Reconstruction (ASR) to detect and reject non-bio artifacts
asr = ASR(sfreq=raw.info['sfreq']) 
asr.fit(raw)
raw = asr.transform(raw)

# ICA to detect and remove independent components like eye-blinks, ECG, muscle artifacts
ica = ICA(n_components=n_channels-len(bad_channels), method='infomax', max_iter=500, random_state=42)
ica.fit(raw)

labels = label_components(raw, ica, 'iclabel')
component_labels = labels['labels']
component_probs = labels['y_pred_proba']
artifact_components = []
for i, prob in enumerate(component_probs):
    if prob >= 0.9 and prob <= 1 and component_labels[i] in ['muscle', 'eye']:
        print(f"Component (i) is likely an artifact")
        artifact_components.append(i)

ica_weights = ica.unmixing_matrix_
ica_inverse_weights = ica.mixing_matrix_
print("flagged artifact components: ", artifact_components)

raw_cleaned = ica.apply(raw.copy(), exclude=artifact_components)

# Loop variables for visual output of the clean raw EEG data in realtime
ch_names = list(rename_dict.values())

fig, axs = plt.subplots(len(ch_names), 1, figsize=(10,12), sharex=True)
plt.subplots_adjust(hspace=0.5)

######## VISUALIZATION #######
# Create a window
mywin = visual.Window([500, 500], monitor="TestMonitor", color=[-1,-1,-1], fullscr=False, units="cm")
# Create fixation cross
fixation = visual.ShapeStim(mywin, vertices=((0, -1), (0, 1), (0, 0), (-1, 0), (1, 0)), lineWidth=2, closeShape=False, lineColor="white")
# Create a dynamic circle for visualization
dynamic_circle = visual.Circle(win=mywin, radius=5, fillColor="cyan", lineColor="white", pos=(0, 0))

# Parameters for smooth radius change
max_radius = 10
min_radius = 2 # Minimum visual radius of the circle

# Creating keyboard component
kb = keyboard.Keyboard()

# Timer for updating the circle radius every 0.5 seconds
update_interval = 0.5
update_timer = core.CountdownTimer(update_interval)

# EEG parameters (user input)
step = 0.01 # in seconds
time_window = 5 # in seconds
n_channels = 2
feed_ch_names = ['O1', 'F3']
freq_band_1 = (4, 7) # First freq band (theta)
freq_band_2 = (8, 12) # second freq band (alpha)
frequency_bands = [freq_band_1, freq_band_2]

# Frequency band names
freq_band_1_name = visual.TextStim(win=mywin, text="Theta (4-7 Hz)", pos=(0, -30), color=(1, 1, 1), opacity=0.75, anchorHoriz='left', anchorVert='bottom', height=10, ori=0.0)
freq_band_2_name = visual.TextStim(win=mywin, text="Alpha (8-12 Hz)", pos=(0, -30), color=(1, 1, 1), opacity=0.75, anchorHoriz='left', anchorVert='bottom', height=10, ori=0.0)

# Main loop variable
epoch_count = 1  # Initialize epoch counter
running = True
results = []  # Initialize feedback data list

if __name__ == '__main__':
    try:
        with LSLClient(info=None, host=host, wait_max=wait_max) as client:
            client_info = client.get_measurement_info()
            sfreq = int(client_info['sfreq'])
            print("LSLClient started, sampling frequency:", sfreq)
            
            while running:
            # Check for keyboard input
                keys = kb.getKeys()
                if 'escape' in keys:
                    running = False
                    break

            while running:
                if update_timer.getTime() <= 0:
                    print(f"Fetching epoch {epoch_count}")
                    #Fetch the epoch data
                    epoch = client.get_data_as_epoch(n_samples=sfreq)
                    print(f"Received epoch {epoch_count} with {epoch.get_data().shape[1]} samples")

                    data, times = epoch.get_data(picks="eeg")[0], epoch.times                    
                    # Apply baseline correction to the epoch data
                    epoch.apply_baseline(baseline=(0, None))
                    
                    # Transform the epoch data to a RawArray for further processing
                    data = np.squeeze(epoch.get_data())
                    raw_realtime = mne.io.RawArray(data,client_info)
                    #raw_realtime.rename_channels(rename_dict)
                    raw_realtime.notch_filter(50, picks='eeg').filter(l_freq=0.1, h_freq=40)
                    raw_realtime.info['bads'].extend(bad_channels)
                    raw_realtime_asr = asr.transform(raw_realtime)
                    raw_realtime_asr_ica = ica.apply(raw_realtime_asr,exclude = artifact_components)
            
                    # Use echt for phase and amplitude extraction
                    _, low_phase, _ = echt(data[0], freq_band_1[0], freq_band_1[1], sfreq)
                    _, _, high_amp = echt(data[0], freq_band_2[0], freq_band_2[1], sfreq)

                    #MVL calculation
                    N = len(low_phase)
                    complex_mvl = (1 / np.sqrt(N)) * np.sum(high_amp * np.exp(1j * low_phase)) / np.sqrt(np.sum(high_amp**2))
                    mvl_abs = np.abs(complex_mvl)

                    # Scale the radius based on mvl_abs
                    scaled_radius = min_radius + (mvl_abs * (max_radius - min_radius))

                    # Update the circle's radius and redraw
                    dynamic_circle.radius = scaled_radius
                    mywin.clearBuffer()
                    dynamic_circle.draw()
                    fixation.draw()
                    mywin.flip()

                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    results.append({
                        "Epoch": epoch_count,
                        "Timestamp": timestamp,
                        "MVL Absolute Value": mvl_abs,
                    })

                    update_timer.reset(update_interval)
                    epoch_count += 1

                if 'escape' in event.getKeys():
                    running = False

    except Exception as e:
        print("Error occurred:", e)

    finally:
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_excel(r"C:\Users\varsh\NFB_Spyder\real_time_mvl_results.xlsx", index=False)
            print("Results saved to 'real_time_mvl_results.xlsx'.")
        mywin.close()
        core.quit()
        print("Streams closed")



import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg') 

import psychopy
from psychopy import visual, core, event
from psychopy.hardware import keyboard

import mne_realtime
from mne_realtime import LSLClient, MockLSLStream
#from mne.time_frequency import 
import pylsl
import pyxdf
from pylsl import resolve_stream, StreamInlet
import mne
mne.viz.set_browser_backend('qt')
import numpy as np
import time
import os
import pandas as pd
from pyprep.find_noisy_channels import NoisyChannels


# this is the host id that identifies your stream on LSL
host = 'Signal_generator_EEG_16_250.0_float32_Vgram'
# this is the max wait time in seconds until client connection
wait_max = 5

# Load a file to stream raw data


ch_names = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8', 'Ch9', 'Ch10', 'Ch11', 'Ch12', 'Ch13', 'Ch14', 'Ch15', 'Ch16']
#ch_names = ['Pz','T8','P3','C3','Fp1','Fp2','O1','P4','Fz','F7','C4','O2','F3','F4','Cz','T3']
running = True
duration = 120 #in seconds
epoch_count = 1

streams = resolve_stream()  # Resolves all available streams
for stream in streams:
    print(f"Stream Name: {stream.name()}, Type: {stream.type()}")

streams = resolve_stream('EEG')

if not streams:
    raise RuntimeError("No EEG streams found. Make sure the LSL stream is running.")

# Create an inlet to pull data
inlet = StreamInlet(streams[0])
print("Successfully connected to the EEG stream!")

# Create a mywindow
mywin = visual.Window([500, 500], monitor="TestMonitor", color=[0, 0, 0], fullscr=False, units="deg")
fixation = visual.ShapeStim(mywin, vertices=((0, -1), (0, 1), (0, 0), (-1, 0), (1, 0)), lineWidth=2, closeShape=False, lineColor="white")

# Create keyboard component
kb = keyboard.Keyboard()

# Create subplots: One row per channel
fig, axs = plt.subplots(len(ch_names), 1, figsize=(10, 12), sharex=True)
plt.subplots_adjust(hspace=0.5)

# Initialize data storage
all_data = []
start_time = time.time()

if __name__ == '__main__':
    try:
        with LSLClient(host=host, wait_max=wait_max) as client:
            client_info = client.get_measurement_info()
            sfreq = client_info['sfreq']
            ch_names_stream = [ch['ch_name'] for ch in client_info['chs']]
            
            # Validate channel names
            if ch_names_stream != ch_names:
                print(f"Warning: Stream channels {ch_names_stream} differ from expected {ch_names}")
                ch_names = ch_names_stream  # Use stream channels to avoid mismatch
            
            # Create MNE Info object
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
            
            while running:
                elapsed_time = time.time() - start_time
                if elapsed_time >= duration:
                    print("Recording duration reached.")
                    break
                
                print(f'Got epoch {epoch_count}')
                
                # Get one second of data (epoch)
                epoch = client.get_data_as_epoch(n_samples=int(sfreq))
                data = epoch.get_data()  # Shape: (n_epochs, n_channels, n_samples)
                all_data.append(data[0])  # Append single epoch data (n_channels, n_samples)
                
                epoch_count += 1
                
                # Update visual display
                fixation.draw()
                mywin.flip()
                
                # Check for keyboard interrupt
                keys = kb.getKeys(['escape'])
                if 'escape' in keys:
                    print("Experiment stopped by user.")
                    break
            
            # Concatenate all data
            if all_data:
                data = np.concatenate(all_data, axis=1)  # Shape: (n_channels, total_samples)
                
                # Create Raw object
                raw = mne.io.RawArray(data, info, verbose=False)
                
                # Save baseline recording
                baseline_path = r'C:\Users\varsh\OneDrive\Desktop\NFB-MNE-Psy\dummy-baseline.fif'
                raw.save(baseline_path, overwrite=True)
                print(f"Baseline saved to {baseline_path}")
                
                # ---------- PSD CALCULATION ----------
                psds, freqs = mne.time_frequency.psd_array_welch(
                    raw.get_data(), sfreq=sfreq, n_fft=2048, n_jobs=1, verbose=False
                )
                psds_db = 10 * np.log10(psds)
                
                bands = {
                    'Delta (0.5–4 Hz)': (0.5, 4),
                    'Theta (4–8 Hz)': (4, 8),
                    'Alpha (8–13 Hz)': (8, 13),
                    'Beta (13–30 Hz)': (13, 30),
                    'Gamma (30–100 Hz)': (30, 100),
                }
                
                band_powers = []
                plot_folder = 'psd_plots'
                os.makedirs(plot_folder, exist_ok=True)
                
                # Calculate band powers and generate plots
                for ch_idx, ch_name in enumerate(raw.info['ch_names']):
                    band_power = {'Channel': ch_name}
                    for band_name, (fmin, fmax) in bands.items():
                        idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
                        mean_power = psds_db[ch_idx, idx_band].mean()
                        band_power[band_name] = mean_power
                    band_powers.append(band_power)
                    
                    # Plot PSD for channel
                    plt.figure(figsize=(6, 4))
                    plt.plot(freqs, psds_db[ch_idx, :], label=ch_name)
                    plt.title(f'PSD - {ch_name}')
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('Power Spectral Density (dB)')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plot_path = os.path.join(plot_folder, f'{ch_name}_psd.png')
                    plt.savefig(plot_path)
                    plt.close()
                
                # Save to Excel
                df = pd.DataFrame(band_powers)
                excel_path = r'C:\Users\varsh\OneDrive\Desktop\NFB-MNE-Psy\Baseline_PSD.xlsx'
                
                with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                    # Write band powers
                    df.to_excel(writer, sheet_name='Band_Powers', index=False)
                    
                    # Add PSD plots
                    worksheet = writer.book.add_worksheet('PSD_Plots')
                    row = 0
                    for ch_idx, ch_name in enumerate(raw.info['ch_names']):
                        img_path = os.path.join(plot_folder, f'{ch_name}_psd.png')
                        worksheet.insert_image(
                            row, 0, img_path, {'x_scale': 0.6, 'y_scale': 0.6}
                        )
                        row += 15  # Adjusted for better spacing
                        
                print(f'PSD results and plots saved to {excel_path}')
                
            else:
                print("No data collected.")
                
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        # Cleanup
        mywin.close()
        plt.close('all')
# import mne
# import pyxdf

# # Load the .xdf file
# try:
#     streams, header = pyxdf.load_xdf(r"C:\Users\varsh\OneDrive\Desktop\NFB-MNE-Psy\dummy_eeg.xdf")
# except ValueError as e:
#     print(f"Error loading XDF file: {e}")
#     exit()

# # Extract data and info (assuming the first stream is EEG data)
# data = streams[0]['time_series'].T
# sfreq = float(streams[0]['info']['nominal_srate'][0])
# channel_names = [ch['label'][0] for ch in streams[0]['info']['desc'][0]['channels'][0]['channel']]
# channel_types = ['eeg'] * len(channel_names)

# # Create MNE Info object
# info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types)

# # Create Raw object
# raw = mne.io.RawArray(data, info)

# # Save as .fif
# raw.save(r"C:\Users\varsh\OneDrive\Desktop\NFB-MNE-Psy\dummy-baseline.fif", overwrite=True)

# import mne

# raw = mne.io.read_raw_fif(r"C:\Users\varsh\OneDrive\Desktop\NFB-MNE-Psy\base.fif", preload=True)
# print(raw.info)

# from psychopy import visual, core, event

# # Create a PsychoPy window
# win = visual.Window(size=(800, 600), color="black", units="pix")

# # Create a visual stimulus (e.g., a moving bar)
# bar = visual.Rect(win, width=50, height=50, fillColor="red", pos=(0, -100))

# # Create a text stimulus for feedback
# feedback_text = visual.TextStim(win, text="Feedback: 0.00", color="white", pos=(0, 200), height=30)

# # Dummy feedback values (e.g., computed from EEG)
# feedback_values = [0.1, 0.3, 0.5, 0.7, 1.0]  # Example values changing over time

# # Main loop for stimulus presentation
# for feedback in feedback_values:
#     # Update feedback text
#     feedback_text.text = f"Feedback: {feedback:.2f}"

#     # Move the bar's height based on feedback
#     bar.height = 50 + (feedback * 100)  # Scaling height

#     # Draw everything
#     feedback_text.draw()
#     bar.draw()
#     win.flip()  # Update the screen

#     core.wait(1)  # Wait for 1 second before updating

#     # Exit if 'q' is pressed
#     if event.getKeys(["q"]):
#         break

# # Close the window
# win.close()
# core.quit()

import mne
builtin_montages = mne.channels.get_builtin_montages(descriptions=True)
for montage_name, montage_description in builtin_montages:
    print(f"{montage_name}: {montage_description}")

